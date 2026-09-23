"""Event-driven backtester over historical klines (same strategies as live).

Fidelity notes (what matches live, what is simplified):
  MATCHES live
    - Strategy code paths (identical ``evaluate()`` calls, no look-ahead:
      signal at bar ``i`` uses rows ``0..i`` only... plus one-bar execution
      delay - fills happen at the *next* bar's close, like a real market order
      placed after the signal bar closes).
    - Entry consensus: ``min_votes`` BUY threshold + SELL veto (same as trader).
    - Risk exits: SL -> TP -> trailing ratchet (same order as ``RiskManager``).
    - ATR stops when ``risk.atr_stops`` is enabled.
  SIMPLIFIED
    - Fees: taker fee per side (spot 0.10% / futures 0.05%); no maker rebates.
    - Slippage: optional ``risk.slippage_bps`` added against the trader.
    - Futures: leverage-scaled notional; no funding payments, no liquidation
      modeling (stops are assumed to fill - optimistic for wide-stop configs).
    - One position per symbol at a time (matches live engine v1).

Metrics per symbol: trades, win rate, PnL%, max drawdown, fees, Sharpe,
profit factor, expectancy, avg win/loss, exposure %, buy-and-hold delta.
"""

from __future__ import annotations

import contextlib
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import BotConfig
from .indicators import atr as atr_fn
from .klines import Klines
from .metrics import max_drawdown as md_fn
from .metrics import profit_factor, sharpe_ratio, sortino_ratio
from .strategies import BUY, SELL, Signal, build_strategies

log = logging.getLogger("glmbot.backtest")

SPOT_TAKER_FEE = 0.001
FUT_TAKER_FEE = 0.0005


@dataclass
class BTTrade:
    symbol: str
    side: str  # BUY | SELL
    ts: int  # open_time in ms
    price: float
    qty: float
    reason: str
    proceeds: float = 0.0  # net quote received (SELL only)
    fee: float = 0.0


@dataclass
class BTResult:
    symbol: str
    trades: list[BTTrade] = field(default_factory=list)
    final_equity: float = 0.0
    start_equity: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    total_fees: float = 0.0
    # extended metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    exposure_pct: float = 0.0
    buy_hold_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    @property
    def pnl_pct(self) -> float:
        return (self.final_equity / self.start_equity - 1) * 100 if self.start_equity else 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "win_rate": round(self.win_rate, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_fees": round(self.total_fees, 2),
            "profit_factor": round(self.profit_factor, 2)
            if self.profit_factor != float("inf")
            else None,
            "expectancy": round(self.expectancy, 2),
            "sharpe_15m": round(self.sharpe, 2),
            "sortino_15m": round(self.sortino, 2),
            "exposure_pct": round(self.exposure_pct, 1),
            "buy_hold_pct": round(self.buy_hold_pct, 2),
            "final_equity": round(self.final_equity, 2),
        }


class Backtester:
    """Walk-forward backtester: iterate candles, same rules as live."""

    def __init__(
        self, cfg: BotConfig, klines: dict[str, Klines], starting_equity: float | None = None
    ):
        self.cfg = cfg
        self.klines = klines
        self.risk_cfg = cfg.risk
        self.strategies = build_strategies(cfg.strategies, cfg.strategy_params)
        self.starting = starting_equity or cfg.risk.quote_budget
        self.is_futures = cfg.market == "futures"
        self.leverage = max(1, cfg.leverage) if self.is_futures else 1
        self.fee_rate = FUT_TAKER_FEE if self.is_futures else SPOT_TAKER_FEE
        self.slip = max(0.0, cfg.risk.slippage_bps) / 10_000.0

    # ---------------- public ----------------
    def run(self) -> dict[str, BTResult]:
        results: dict[str, BTResult] = {}
        for symbol, k in self.klines.items():
            try:
                results[symbol] = self._run_symbol(symbol, k)
            except Exception as e:
                log.error("backtest failed for %s: %s", symbol, e)
                r = BTResult(symbol=symbol, start_equity=self.starting, final_equity=self.starting)
                results[symbol] = r
        return results

    def export_trades_csv(self, results: dict[str, BTResult], path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "side", "ts_ms", "price", "qty", "fee", "proceeds", "reason"])
            for sym in sorted(results):
                for t in results[sym].trades:
                    w.writerow(
                        [
                            t.symbol,
                            t.side,
                            t.ts,
                            f"{t.price:.8f}",
                            f"{t.qty:.8f}",
                            f"{t.fee:.4f}",
                            f"{t.proceeds:.2f}",
                            t.reason,
                        ]
                    )
        return path

    # ---------------- engine ----------------
    def _run_symbol(self, symbol: str, k: Klines) -> BTResult:
        res = BTResult(symbol=symbol, start_equity=self.starting)
        cash = self.starting
        qty = 0.0
        entry_price = 0.0
        entry_time: int | None = None  # open_time ms of the fill bar
        margin_locked = 0.0
        stop_loss: float | None = None
        take_profit: float | None = None
        trail_high: float | None = None
        equity_curve: list[float] = []
        bars_in_pos = 0

        closes = k.close
        times = k.open_time
        n = len(k)
        if n == 0:
            res.final_equity = cash
            return res
        buy_hold = (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0.0
        res.buy_hold_pct = buy_hold

        # Precompute ATR series once (used only when atr_stops enabled).
        atr_s: list[float | None] = [None] * n
        if self.risk_cfg.atr_stops:
            try:
                atr_s = atr_fn(k.high, k.low, k.close, self.risk_cfg.atr_period)
            except Exception:
                atr_s = [None] * n

        min_rows = max(60, self._warmup_rows())
        pending_entry: dict | None = None  # signal bar -> fill next bar

        for i in range(min_rows, n):
            window = Klines(k.rows(i + 1))
            close = float(closes[i])

            # ---- execute pending entry at this bar's close (+slippage) ----
            if pending_entry is not None and qty == 0:
                fill_price = close * (1 + self.slip)
                budget = min(self.starting * self.risk_cfg.per_trade_pct / 100.0, cash)
                if budget >= 10:
                    if self.is_futures:
                        notional = budget * self.leverage
                        fee = notional * self.fee_rate
                        if budget + fee <= cash:
                            qty = notional / fill_price
                            margin_locked = budget
                            cash -= budget + fee
                            res.total_fees += fee
                            entry_price = fill_price
                            entry_time = times[i]
                            stop_loss, take_profit = self._levels(fill_price, atr_s[i])
                            trail_high = fill_price
                            res.trades.append(
                                BTTrade(
                                    symbol,
                                    "BUY",
                                    times[i],
                                    fill_price,
                                    qty,
                                    pending_entry["reason"],
                                    fee=fee,
                                )
                            )
                    else:
                        fee = budget * self.fee_rate
                        qty = (budget - fee) / fill_price
                        cash -= budget
                        res.total_fees += fee
                        entry_price = fill_price
                        entry_time = times[i]
                        stop_loss, take_profit = self._levels(fill_price, atr_s[i])
                        trail_high = fill_price
                        res.trades.append(
                            BTTrade(
                                symbol,
                                "BUY",
                                times[i],
                                fill_price,
                                qty,
                                pending_entry["reason"],
                                fee=fee,
                            )
                        )
                pending_entry = None

            # ---- manage open position exits (at this bar's close) ----
            if qty > 0:
                bars_in_pos += 1
                exit_reason: str | None = None
                # 0. breakeven lock (mirrors RiskManager.check_exit)
                be_trig = self.risk_cfg.breakeven_trigger_pct
                if be_trig > 0 and entry_price > 0 and close >= entry_price * (1 + be_trig / 100):
                    be_sl = entry_price * (1 + self.risk_cfg.breakeven_buffer_pct / 100)
                    if stop_loss is None or be_sl > stop_loss:
                        stop_loss = be_sl
                if stop_loss is not None and close <= stop_loss:
                    exit_reason = "stop-loss"
                elif take_profit is not None and close >= take_profit:
                    exit_reason = "take-profit"
                elif self.risk_cfg.trailing_stop_pct > 0 and trail_high is not None:
                    if close > trail_high:
                        trail_high = close
                        new_sl = trail_high * (1 - self.risk_cfg.trailing_stop_pct / 100)
                        if stop_loss is None or new_sl > stop_loss:
                            stop_loss = new_sl
                    if stop_loss is not None and trail_high > entry_price and close <= stop_loss:
                        exit_reason = "trailing-stop"
                if (
                    exit_reason is None
                    and self.risk_cfg.max_hold_min > 0
                    and entry_time is not None
                    and (times[i] - entry_time) / 60000 >= self.risk_cfg.max_hold_min
                ):
                    exit_reason = "time-stop"
                if exit_reason is None:
                    # unanimous-SELL strategy exit (matches live trader)
                    votes = self._votes(symbol, window)
                    if votes and all(v.side == SELL for v in votes):
                        exit_reason = f"signal exit: {votes[0].reason}"
                if exit_reason is not None:
                    t = self._sell(
                        res, symbol, times[i], close, qty, entry_price, margin_locked, exit_reason
                    )
                    cash += t.proceeds
                    qty, margin_locked = 0.0, 0.0
                    entry_time = None
                    stop_loss = take_profit = trail_high = None
                    res.trades.append(t)

            # ---- look for entry signal (fills NEXT bar) ----
            if qty == 0 and pending_entry is None:
                votes = self._votes(symbol, window)
                buys = [v for v in votes if v.side == BUY]
                sells = [v for v in votes if v.side == SELL]
                m = len(self.strategies)
                need = self.risk_cfg.min_votes if self.risk_cfg.min_votes > 0 else m
                need = min(max(need, 1), m)
                if len(buys) >= need and len(sells) == 0:
                    pending_entry = {"reason": buys[0].reason}

            if self.is_futures and qty > 0:
                # margin + unrealized (same honest accounting as the live trader)
                equity_curve.append(margin_locked + qty * (close - entry_price) + cash)
            else:
                equity_curve.append(cash + qty * close)

        # close residual at last close
        if qty > 0:
            t = self._sell(
                res,
                symbol,
                times[-1],
                float(closes[-1]),
                qty,
                entry_price,
                margin_locked,
                "backtest-end",
            )
            cash += t.proceeds
            res.trades.append(t)
            equity_curve.append(cash)
            qty = 0.0

        res.equity_curve = equity_curve
        res.final_equity = equity_curve[-1] if equity_curve else cash
        self._finalize_metrics(res, bars_in_pos, n - min_rows)
        return res

    # ---------------- helpers ----------------
    def _warmup_rows(self) -> int:
        need = 30
        p = self.cfg.strategy_params
        need = max(need, int(p.get("ema_cross", {}).get("slow", 21)) + 2)
        need = max(
            need,
            int(p.get("macd", {}).get("slow", 26)) + int(p.get("macd", {}).get("signal", 9)) + 2,
        )
        need = max(need, int(p.get("bollinger", {}).get("period", 20)) + 2)
        need = max(need, int(p.get("vwap_trend", {}).get("period", 20)) + 2)
        need = max(
            need,
            int(p.get("stoch_rsi_cross", {}).get("rsi_period", 14))
            + int(p.get("stoch_rsi_cross", {}).get("stoch_period", 14))
            + 1,
        )
        need = max(
            need,
            int(p.get("bollinger_squeeze", {}).get("period", 20))
            + int(p.get("bollinger_squeeze", {}).get("lookback", 50)),
        )
        need = max(need, int(p.get("trend_momentum", {}).get("slow", 50)) + 2)
        need = max(need, int(p.get("donchian_breakout", {}).get("period", 20)) + 2)
        need = max(need, int(p.get("supertrend", {}).get("period", 10)) + 3)
        if self.risk_cfg.atr_stops:
            need = max(need, self.risk_cfg.atr_period + 2)
        return need

    def _votes(self, symbol: str, window: Klines) -> list[Signal]:
        votes: list[Signal] = []
        for strat in self.strategies:
            with contextlib.suppress(Exception):
                votes.append(strat.evaluate(symbol, window))
        return votes

    def _levels(self, price: float, atr_val: float | None):
        if self.risk_cfg.atr_stops and atr_val and atr_val > 0:
            sl = price - self.risk_cfg.atr_sl_mult * atr_val
            tp = price + self.risk_cfg.atr_tp_mult * atr_val
            if sl <= 0 or sl >= price:
                sl = price * (1 - self.risk_cfg.stop_loss_pct / 100)
            if tp <= price:
                tp = price * (1 + self.risk_cfg.take_profit_pct / 100)
        else:
            sl = price * (1 - self.risk_cfg.stop_loss_pct / 100)
            tp = price * (1 + self.risk_cfg.take_profit_pct / 100)
        return sl, tp

    def _sell(
        self,
        res: BTResult,
        symbol: str,
        ts,
        price: float,
        qty: float,
        entry_price: float,
        margin_locked: float,
        reason: str,
    ) -> BTTrade:
        fill_price = price * (1 - self.slip)
        gross = qty * fill_price
        fee = gross * self.fee_rate
        res.total_fees += fee
        if self.is_futures:
            pnl = qty * (fill_price - entry_price)
            net = margin_locked + pnl - fee
        else:
            net = gross - fee
        t = BTTrade(symbol, "SELL", ts, fill_price, qty, reason, proceeds=net, fee=fee)
        # pair with entry for win/loss accounting
        buys = [x for x in res.trades if x.side == "BUY"]
        if buys:
            entry = buys[-1]
            # futures: compare margin-relative; spot: notional-relative
            pnl_trade = net - (margin_locked if self.is_futures else entry.price * entry.qty)
            if pnl_trade > 0:
                res.gross_profit += pnl_trade
            else:
                res.gross_loss += abs(pnl_trade)
        return t

    def _finalize_metrics(self, res: BTResult, bars_in_pos: int, total_bars: int) -> None:
        sells = [t for t in res.trades if t.side == "SELL"]
        res.n_trades = len(sells)
        # wins: SELL proceeds vs paired BUY cost
        wins = 0
        win_pnls: list[float] = []
        loss_pnls: list[float] = []
        open_buy: BTTrade | None = None
        for t in res.trades:
            if t.side == "BUY":
                open_buy = t
            elif t.side == "SELL" and open_buy is not None:
                cost = (
                    (open_buy.qty * open_buy.price)
                    if not self.is_futures
                    else (open_buy.qty * open_buy.price / self.leverage)
                )
                pnl = t.proceeds - cost
                if pnl > 0:
                    wins += 1
                    win_pnls.append(pnl)
                else:
                    loss_pnls.append(pnl)
                open_buy = None
        # fallback: price-based when cost pairing unavailable
        if not win_pnls and not loss_pnls and sells:
            open_buy = None
            for t in res.trades:
                if t.side == "BUY":
                    open_buy = t
                elif t.side == "SELL" and open_buy is not None:
                    (win_pnls if t.price > open_buy.price else loss_pnls).append(
                        abs(t.price - open_buy.price) * t.qty
                    )
                    open_buy = None
            wins = len(win_pnls)
        res.n_wins = wins
        res.win_rate = (wins / len(sells) * 100) if sells else 0.0
        res.avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        res.avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
        res.profit_factor = profit_factor(res.gross_profit, res.gross_loss)
        from .metrics import expectancy as exp_fn

        res.expectancy = exp_fn(res.win_rate, res.avg_win, res.avg_loss)
        if res.equity_curve:
            res.max_drawdown_pct = md_fn(res.equity_curve)
            res.sharpe = sharpe_ratio(res.equity_curve)
            res.sortino = sortino_ratio(res.equity_curve)
        res.exposure_pct = (bars_in_pos / total_bars * 100) if total_bars > 0 else 0.0
