"""Execution layer: PaperBroker (simulated), LiveBroker (spot), FuturesBroker.

Fill model (all brokers return the same dict keys):
  buy  -> {qty, price, fee, order_id, quote_amt, margin_used}
  sell -> {qty, price, fee, order_id, gross, quote_amt, margin_freed}

Fees: 0.10% spot taker, 0.05% futures taker (conservative defaults).
Paper fills happen at ``price_hint`` (live loop passes the latest close);
use ``RiskCfg.slippage_bps`` in the backtester for slippage studies.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Optional

from .api import BinanceClient

log = logging.getLogger("glmbot.broker")

TAKER_FEE = 0.001       # 0.10% spot taker
FUT_TAKER_FEE = 0.0005  # 0.05% futures taker


class Broker(ABC):
    fee_rate: float = TAKER_FEE

    @abstractmethod
    def balance_quote(self) -> float:
        """Free quote-currency balance available for new entries."""

    def buy_market(self, symbol: str, quote_amount: float, price_hint: float) -> Dict:
        """Market buy of ``quote_amount`` notional. Returns fill dict."""
        raise NotImplementedError

    def sell_market(self, symbol: str, qty: float, price_hint: float,
                    entry_price: Optional[float] = None) -> Dict:
        """Market sell of ``qty`` base. Returns fill dict."""
        raise NotImplementedError


class PaperBroker(Broker):
    """Simulated fills at current market price with taker fees.

    Futures-aware: ``margin = notional / leverage`` is locked on open and
    ``margin + pnl − fee`` is released on close. Spot-like when leverage=1.
    """

    def __init__(self, client: BinanceClient, starting_cash: float,
                 leverage: int = 1, symbol_filter=None):
        self.client = client
        self.cash = float(starting_cash)
        self.leverage = max(1, int(leverage))
        self.fee_rate = FUT_TAKER_FEE if getattr(client, "market", "spot") == "futures" else TAKER_FEE

    def balance_quote(self) -> float:
        return self.cash

    def buy_market(self, symbol: str, quote_amount: float, price_hint: float) -> Dict:
        if price_hint <= 0:
            raise ValueError(f"{symbol}: invalid price_hint {price_hint}")
        if quote_amount <= 0:
            raise ValueError(f"{symbol}: invalid quote_amount {quote_amount}")
        price = float(price_hint)
        if self.leverage <= 1:
            fee_quote = quote_amount * self.fee_rate
            if quote_amount + fee_quote > self.cash:
                raise ValueError(
                    f"{symbol}: insufficient paper cash "
                    f"(need {quote_amount + fee_quote:.2f}, have {self.cash:.2f})"
                )
            qty = quote_amount / price
            self.cash -= quote_amount + fee_quote
            margin = quote_amount
        else:
            notional = quote_amount * self.leverage
            fee_quote = notional * self.fee_rate
            margin = quote_amount
            if margin + fee_quote > self.cash:
                raise ValueError(
                    f"{symbol}: insufficient paper margin "
                    f"(need {margin + fee_quote:.2f}, have {self.cash:.2f})"
                )
            qty = notional / price
            self.cash -= margin + fee_quote
        return {
            "qty": qty, "price": price, "fee": fee_quote, "order_id": None,
            "quote_amt": quote_amount, "margin_used": margin,
        }

    def sell_market(self, symbol: str, qty: float, price_hint: float,
                    entry_price: Optional[float] = None) -> Dict:
        if qty <= 0:
            raise ValueError(f"{symbol}: invalid qty {qty}")
        gross = qty * float(price_hint)
        fee = gross * self.fee_rate
        if entry_price is not None and self.leverage > 1:
            margin = qty * entry_price / self.leverage
            pnl = qty * (price_hint - entry_price)
            self.cash += margin + pnl - fee
        else:
            self.cash += gross - fee
        return {
            "qty": qty, "price": float(price_hint), "fee": fee, "order_id": None,
            "gross": gross, "quote_amt": gross - fee, "margin_freed": gross,
        }


class LiveBroker(Broker):
    """Real market orders on Binance spot (testnet or mainnet per client)."""

    def __init__(self, client: BinanceClient):
        self.client = client
        self.fee_rate = TAKER_FEE

    def balance_quote(self) -> float:
        return self.balance_quote_for("USDT")

    def _filters(self, symbol: str):
        return self.client.get_filter(symbol)

    def balance_quote_for(self, quote_asset: str) -> float:
        try:
            balances = self.client.account().get("balances", [])
        except Exception as e:
            log.error("balance fetch failed: %s", e)
            raise
        for b in balances:
            if b.get("asset") == quote_asset:
                return float(b.get("free", 0) or 0)
        return 0.0

    def buy_market(self, symbol: str, quote_amount: float, price_hint: float) -> Dict:
        f = self._filters(symbol)
        price = Decimal(str(price_hint))
        if price <= 0:
            raise ValueError(f"{symbol}: invalid price {price_hint}")
        quote = Decimal(str(quote_amount)).quantize(Decimal("0.01"))
        qty = f.clamp_qty(f.round_qty(quote / price), price)
        if qty is None:
            raise ValueError(
                f"{symbol}: order below exchange minimums "
                f"(minQty={f.min_qty}, minNotional={f.min_notional}, "
                f"wanted {quote_amount:.2f} USDT @ {price_hint:.6g})"
            )
        res = self.client.place_order(symbol=symbol, side="BUY",
                                      order_type="MARKET", quantity=fmt_dec(qty))
        price_avg, fee = self._avg_price_fee(res)
        executed_qty = float(res.get("executedQty", qty))
        fill_price = price_avg if price_avg else float(price)
        return {
            "qty": executed_qty, "price": fill_price, "fee": fee,
            "order_id": res.get("orderId"),
            "quote_amt": executed_qty * fill_price, "margin_used": 0.0,
        }

    def sell_market(self, symbol: str, qty: float, price_hint: float,
                    entry_price: Optional[float] = None) -> Dict:
        f = self._filters(symbol)
        qty_d = f.round_qty(Decimal(str(qty)))
        if qty_d <= 0:
            raise ValueError(f"{symbol}: qty {qty} rounds to 0 (step={f.step_size}) - dust; skipping")
        res = self.client.place_order(symbol=symbol, side="SELL",
                                      order_type="MARKET", quantity=fmt_dec(qty_d))
        price_avg, fee = self._avg_price_fee(res)
        executed = float(res.get("executedQty", qty_d))
        fill_price = price_avg if price_avg else float(price_hint)
        gross = executed * fill_price
        return {
            "qty": executed, "price": fill_price, "fee": fee,
            "order_id": res.get("orderId"), "gross": gross,
            "quote_amt": gross, "margin_freed": 0.0,
        }

    @staticmethod
    def _avg_price_fee(res: Dict) -> tuple:
        try:
            cq = Decimal(str(res.get("cummulativeQuoteQty", "0")))
            eq = Decimal(str(res.get("executedQty", "0")))
            price = float(cq / eq) if eq else 0.0
        except Exception:
            price = 0.0
        fee = 0.0
        for fl in res.get("fills", []) or []:
            try:
                if fl.get("commissionAsset") in ("USDT", "BNB", "BTC", "ETH"):
                    fee += float(fl.get("commission", 0) or 0)
            except (ValueError, TypeError):
                continue
        return price, fee


class FuturesBroker(LiveBroker):
    """USD-M futures: one-way mode, fixed leverage, market orders.

    BUY opens a long; SELL with ``reduceOnly`` closes it. No shorts in v1
    (strategy signals are long-only; SELL means exit).
    """

    def __init__(self, client: BinanceClient, leverage: int = 1):
        super().__init__(client)
        if client.market != "futures":
            raise ValueError("FuturesBroker requires a futures BinanceClient")
        self.leverage = max(1, int(leverage))
        self.fee_rate = FUT_TAKER_FEE
        self._setup_done: set = set()

    def _ensure_setup(self, symbol: str) -> None:
        """One-way position mode + leverage per symbol (idempotent)."""
        if symbol in self._setup_done:
            return
        try:
            self.client.change_position_mode(hedge=False)
        except Exception as e:
            log.debug("position mode: %s", e)
        try:
            self.client.set_leverage(symbol, self.leverage)
        except Exception as e:
            log.warning("set leverage %s %dx failed: %s", symbol, self.leverage, e)
        self._setup_done.add(symbol)

    def balance_quote_for(self, quote_asset: str = "USDT") -> float:
        try:
            return float(self.client.futures_balance()["free"])
        except Exception as e:
            log.error("futures balance fetch failed: %s", e)
            raise

    def buy_market(self, symbol: str, quote_amount: float, price_hint: float) -> Dict:
        self._ensure_setup(symbol)
        f = self._filters(symbol)
        price = Decimal(str(price_hint))
        notional = Decimal(str(quote_amount)) * self.leverage
        qty = f.clamp_qty(f.round_qty(notional / price), price)
        if qty is None:
            raise ValueError(
                f"{symbol}: order below futures minimums (minQty={f.min_qty}, "
                f"minNotional={f.min_notional}, wanted notional {notional:.2f} USDT)"
            )
        res = self.client.place_order(symbol=symbol, side="BUY",
                                      order_type="MARKET", quantity=fmt_dec(qty))
        price_avg, fee = self._avg_price_fee(res)
        executed = float(res.get("executedQty", qty))
        fill_price = price_avg if price_avg else float(price)
        notional_fill = executed * fill_price
        return {
            "qty": executed, "price": fill_price, "fee": fee,
            "order_id": res.get("orderId"), "quote_amt": notional_fill,
            "margin_used": notional_fill / self.leverage,
        }

    def sell_market(self, symbol: str, qty: float, price_hint: float,
                    entry_price: Optional[float] = None) -> Dict:
        self._ensure_setup(symbol)
        f = self._filters(symbol)
        qty_d = f.round_qty(Decimal(str(qty)))
        if qty_d <= 0:
            raise ValueError(f"{symbol}: qty {qty} rounds to 0 (step={f.step_size}) - dust; skipping")
        res = self.client.place_order(symbol=symbol, side="SELL", order_type="MARKET",
                                      quantity=fmt_dec(qty_d), reduce_only=True)
        price_avg, fee = self._avg_price_fee(res)
        executed = float(res.get("executedQty", qty_d))
        fill_price = price_avg if price_avg else float(price_hint)
        gross = executed * fill_price
        return {
            "qty": executed, "price": fill_price, "fee": fee,
            "order_id": res.get("orderId"), "gross": gross,
            "quote_amt": gross, "margin_freed": 0.0,
        }


def fmt_dec(d: Decimal) -> str:
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"
