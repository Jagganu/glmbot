"""Core trading engine: data -> strategies -> risk -> execution -> journal/notify.

Cycle order (every ``update_interval_sec``):
  1. Fetch quotes (latest 15m close per symbol; batch where possible).
  2. Manage exits first: hard SL -> TP -> trailing -> unanimous-SELL signal exit.
  3. Scan entries: vote consensus (``min_votes`` + SELL veto) -> risk gates ->
     ATR-aware SL/TP -> broker fill -> journal + notify.
  4. Snapshot equity + evaluate the daily-loss kill switch.

Ops features: startup preflight (keys/balance/filters/leverage), graceful
SIGINT/SIGTERM shutdown, heartbeat file (``data/glmbot.heartbeat``), cycle
timing stats, stale-quote detection, restart cooldown restore.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from .api import BinanceClient
from .broker import Broker, FuturesBroker, PaperBroker
from .config import BotConfig
from .indicators import atr as atr_fn
from .klines import Klines
from .notifier import Notifier, format_buy, format_sell
from .risk import RiskManager
from .storage import Store, utcnow
from .strategies import BUY, SELL, Signal, build_strategies

log = logging.getLogger("glmbot.trader")

CANDLE_INTERVAL = "15m"
HEARTBEAT_FILE = Path("data/glmbot.heartbeat")
STALE_QUOTE_SEC = 15 * 60 + 120  # a 15m close older than this is suspicious


class Trader:
    def __init__(
        self,
        cfg: BotConfig,
        store: Store,
        client: BinanceClient,
        broker: Broker,
        notifier: Notifier,
    ):
        self.cfg = cfg
        self.store = store
        self.client = client
        self.broker = broker
        self.notifier = notifier
        self.strategies = build_strategies(cfg.strategies, cfg.strategy_params)
        self.risk = RiskManager(cfg.risk, store)
        self._stop = False
        self._cycles = 0
        self._cycle_ms: list[float] = []
        self._sync_notified: set[tuple[str, float]] = set()  # (symbol, trigger) alerted
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(OSError, ValueError):  # Windows / non-main-thread
                signal.signal(sig, self._handle_stop)

    # ---------------- lifecycle ----------------
    def _handle_stop(self, *_a) -> None:
        if not self._stop:
            log.info("shutdown signal received - finishing current cycle...")
        self._stop = True

    def preflight(self) -> list[str]:
        """Startup checks. Returns warnings (empty = clean). Raises on fatal."""
        warnings: list[str] = []
        # 1. clock sync
        try:
            offset = self.client.sync_time()
            if abs(offset) > 5000:
                warnings.append(f"clock offset {offset}ms is large - NTP sync recommended")
        except Exception as e:
            warnings.append(f"time sync failed: {e}")
        # 2. connectivity
        if not self.client.ping():
            raise RuntimeError(
                "Binance REST unreachable (ping failed) - check network/VPN/geo-block"
            )
        # 3. keys in live mode
        if self.cfg.mode == "live" and (
            not self.cfg.api_key or self.cfg.api_key.startswith("YOUR_")
        ):
            raise RuntimeError("live mode requires api.key (or GLMBOT_API_KEY)")
        # 4. balance sanity
        try:
            cash = self._cash()
            if cash <= 0:
                warnings.append(
                    f"quote balance is {cash:.2f} {self.cfg.quote_asset} - deposits needed?"
                )
            elif cash < self.cfg.risk.quote_budget:
                warnings.append(
                    f"exchange balance {cash:.2f} < configured quote_budget "
                    f"{self.cfg.risk.quote_budget:.2f} - sizing uses the budget; "
                    "entries may fail on insufficient funds"
                )
        except Exception as e:
            warnings.append(f"balance check failed: {e}")
        # 5. filters for every symbol (fail fast on delisted/typo'd symbols)
        for sym in self.cfg.symbols:
            try:
                self.client.get_filter(sym)
            except Exception as e:
                warnings.append(f"{sym}: exchange filter lookup failed: {e}")
        # 6. futures leverage handshake (best effort; warns, doesn't fail)
        if self.cfg.market == "futures" and self.cfg.mode == "live":
            try:
                from .broker import FuturesBroker

                if isinstance(self.broker, FuturesBroker):
                    for sym in self.cfg.symbols[:3]:
                        self.client.set_leverage(sym, self.cfg.leverage)
                    break_marker = True
                    assert break_marker
            except Exception as e:
                warnings.append(f"leverage preset failed (will retry per-symbol): {e}")
        return warnings

    # ---------------- data ----------------
    def _df(self, symbol: str) -> Klines:
        """Closed candles only (#11): the live feed's last bar is still
        forming - strategies evaluating it would repaint and diverge from
        the backtest. Quotes/exits intentionally keep using latest prices."""
        kl = self.client.klines(symbol, CANDLE_INTERVAL, limit=300)
        if not kl:
            raise RuntimeError(f"empty kline response for {symbol}")
        k = Klines(kl).closed()
        if not k:
            raise RuntimeError(f"no closed candles for {symbol}")
        issues = k.validate()
        if issues:
            log.warning("%s kline quality: %s", symbol, "; ".join(issues[:2]))
        return k

    def _atr(self, k: Klines) -> float | None:
        try:
            period = self.cfg.risk.atr_period
            if len(k.close) < period + 1:
                return None
            return atr_fn(k.high, k.low, k.close, period)[-1]
        except Exception as e:
            log.warning("ATR calc failed: %s", e)
            return None

    def _market_stats(self, symbol: str, k: Klines | None = None) -> dict:
        """Volatility / volume / chandelier context for advanced risk filters.

        Returns {atr, atr_pct, volume_ratio, highest_high}. Missing data is
        None (filters fail open). One kline fetch is reused by the caller.
        """
        try:
            kk = k if k is not None else self._df(symbol)
        except Exception:
            return {"atr": None, "atr_pct": None, "volume_ratio": None, "highest_high": None}
        atr_v = self._atr(kk)
        price = float(kk.close[-1]) if kk.close else 0.0
        atr_pct = (atr_v / price * 100.0) if atr_v and price else None
        volume_ratio = None
        try:
            if len(kk.volume) >= 21 and sum(kk.volume[-20:]) > 0:
                sma20 = sum(kk.volume[-20:]) / 20.0
                volume_ratio = (kk.volume[-1] / sma20) if sma20 > 0 else None
        except Exception:
            volume_ratio = None
        highest_high = None
        try:
            if self.cfg.risk.chandelier_enabled and len(kk.high) >= self.cfg.risk.chandelier_period:
                highest_high = max(kk.high[-self.cfg.risk.chandelier_period :])
        except Exception:
            highest_high = None
        return {
            "atr": atr_v,
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio,
            "highest_high": highest_high,
            "klines": kk,
        }

    def _price(self, symbol: str) -> float:
        kl = self.client.klines(symbol, CANDLE_INTERVAL, limit=2)
        if not kl:
            raise RuntimeError(f"no klines for {symbol}")
        age_ms = int(time.time() * 1000) - int(kl[-1].get("close_time", kl[-1]["open_time"]))
        if age_ms > STALE_QUOTE_SEC * 1000:
            log.warning("%s quote is stale (%ds old)", symbol, age_ms // 1000)
        return float(kl[-1]["close"])

    def _consensus(self, symbol: str, df: Klines | None = None) -> Signal | None:
        """Vote-based entry signal.

        ``min_votes=0`` (default) requires ALL strategies to agree;
        ``min_votes=N`` requires >=N BUY with zero SELL dissent.
        Unanimous SELL is returned as an exit signal regardless.
        """
        df = df if df is not None else self._df(symbol)
        votes: list[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.evaluate(symbol, df)
                votes.append(sig)
                with contextlib.suppress(Exception):
                    self.store.log_signal(sig)
            except Exception as e:
                log.error("strategy %s failed on %s: %s", strat.name, symbol, e)
        if not votes:
            return None
        buys = [v for v in votes if v.side == BUY]
        sells = [v for v in votes if v.side == SELL]
        n = len(self.strategies)
        need = self.cfg.risk.min_votes if self.cfg.risk.min_votes > 0 else n
        need = min(max(need, 1), n)
        if len(buys) >= need and len(sells) == 0:
            return buys[0]
        if sells and len(sells) == n:
            return sells[0]
        return None

    def _quotes(self) -> dict[str, float]:
        # Batch path first (one HTTP call), per-symbol fallback for resilience.
        try:
            if hasattr(self.client, "ticker_prices"):
                return self.client.ticker_prices(list(self.cfg.symbols))
        except Exception as e:
            log.debug("batch price fetch failed, falling back per-symbol: %s", e)
        out: dict[str, float] = {}
        for s in self.cfg.symbols:
            try:
                out[s] = self._price(s)
            except Exception as e:
                log.warning("price fetch failed for %s: %s", s, e)
        return out

    def position_value(self, pos: dict, price: float) -> float:
        """Honest mark-to-market value of one open position.

        Spot: full notional (qty x price) - cash paid it in full.
        Futures: locked margin + unrealized PnL - only the margin left
        the wallet, so counting full notional would double-count.
        """
        qty = float(pos["qty"])
        entry = float(pos["entry_price"])
        if self.cfg.market == "futures":
            margin = qty * entry / max(1, self.cfg.leverage)
            return margin + qty * (price - entry)
        return qty * price

    # ---------------- main loop ----------------
    def run_once(self) -> dict[str, int]:
        """One full sense->decide->act cycle. Returns cycle counters (for tests/ops)."""
        t0 = time.time()
        mode = self.cfg.mode
        stats = {"exits": 0, "entries": 0, "errors": 0}
        quotes = self._quotes()
        if not quotes:
            log.warning("no prices fetched (offline?) - exits unmanaged this cycle")

        # 1) manage exits on open positions (risk stops first, then signal exits)
        for pos in self.store.open_positions(mode):
            symbol = pos["symbol"]
            price = quotes.get(symbol)
            if price is None:
                continue
            ch_high, ch_atr = None, None
            if self.cfg.risk.chandelier_enabled:
                try:
                    ctx = self._market_stats(symbol)
                    ch_high, ch_atr = ctx["highest_high"], ctx["atr"]
                except Exception:
                    ch_high, ch_atr = None, None
            try:
                plan = self.risk.check_exit(pos, price, ch_high, ch_atr)
            except Exception as e:
                log.error("exit check failed for %s: %s", symbol, e)
                stats["errors"] += 1
                continue
            if plan.action == "exit":
                self._close_position(pos, price, plan.reason)
                stats["exits"] += 1
                continue
            # Position survives: ratchets (trailing/breakeven) may have raised
            # the journal SL above the armed exchange trigger - sync it up.
            self._maybe_sync_exchange_stop(pos)
            try:
                sig = self._consensus(symbol)
            except Exception as e:
                log.error("signal eval failed for %s: %s", symbol, e)
                stats["errors"] += 1
                continue
            if sig is not None and sig.side == SELL:
                self._close_position(pos, price, f"signal exit: {sig.reason}")
                stats["exits"] += 1

        # 1b) per-cycle reconciliation (live futures): journal vs exchange.
        # Catches stops that fired while away and unknown manual positions.
        self._reconcile_cycle(quotes)

        # 2) entries (consensus -> regime filter -> risk gates -> fill)
        halted, why_halt = self.risk.halted(mode)
        if halted:
            log.warning("kill switch active for today - no new entries (%s)", why_halt)
        else:
            for symbol in self.cfg.symbols:
                price = quotes.get(symbol)
                if price is None:
                    continue
                try:
                    if self.store.get_open_position(symbol, mode):
                        continue
                    sig = self._consensus(symbol)
                    if sig is None or sig.side != BUY:
                        continue
                    ctx = self._market_stats(symbol)
                    if self.cfg.regime_filter:
                        try:
                            from .regime import detect_regime, regime_allows_entries

                            kk = ctx.get("klines")
                            if kk is None:
                                kk = self._df(symbol)
                            info = detect_regime(kk)
                            ok_reg, why_reg = regime_allows_entries(
                                info["regime"], self.cfg.regime_allow_chop
                            )
                            if not ok_reg:
                                log.info("skip entry %s: %s", symbol, why_reg)
                                continue
                        except Exception as e:
                            log.debug("regime filter bypassed for %s: %s", symbol, e)
                    ok, why = self.risk.can_open(
                        symbol,
                        mode,
                        atr_pct=ctx.get("atr_pct"),
                        volume_ratio=ctx.get("volume_ratio"),
                    )
                    if not ok:
                        log.info("skip entry %s: %s", symbol, why)
                        continue
                    if self._open_position(symbol, price, sig):
                        stats["entries"] += 1
                except Exception as e:
                    log.error("entry scan failed for %s: %s", symbol, e)
                    stats["errors"] += 1

        # 3) equity snapshot + kill switch
        try:
            cash = self._cash()
        except Exception as e:
            log.error("balance fetch failed; snapshotting with last-known cash: %s", e)
            cash = 0.0
        pos_val = sum(
            self.position_value(pos, quotes.get(pos["symbol"], pos["entry_price"]))
            for pos in self.store.open_positions(mode)
        )
        total = cash + pos_val
        try:
            self.store.snapshot_equity(mode, cash, pos_val)
        except Exception as e:
            log.error("equity snapshot failed: %s", e)
        if self.risk.check_daily_loss(mode, total):
            self.notifier.kill_switch(
                f"[{mode.upper()}] DAILY LOSS CAP HIT - trading halted until tomorrow (UTC) "
                f"(equity {total:.2f} {self.cfg.quote_asset})"
            )
        if self.risk.check_daily_profit(mode, total):
            self.notifier.kill_switch(
                f"[{mode.upper()}] DAILY PROFIT LOCK +{self.cfg.risk.daily_profit_lock_pct:g}% - "
                f"banking the green day, no new entries until tomorrow (UTC) "
                f"(equity {total:.2f} {self.cfg.quote_asset})"
            )
        if self.risk.check_consecutive_losses(mode):
            self.notifier.kill_switch(
                f"[{mode.upper()}] CONSECUTIVE LOSS HALT - trading halted until tomorrow (UTC)"
            )
        if self.risk.check_drawdown_halt(mode, total):
            self.notifier.kill_switch(
                f"[{mode.upper()}] DRAWDOWN HALT - trading halted until tomorrow (UTC) "
                f"(equity {total:.2f} {self.cfg.quote_asset})"
            )
        dt_ms = (time.time() - t0) * 1000
        self._cycle_ms.append(dt_ms)
        if len(self._cycle_ms) > 50:
            self._cycle_ms.pop(0)
        log.info(
            "cycle done: cash=%.2f positions=%.2f total=%.2f (+%d/-%d, %.0fms)",
            cash,
            pos_val,
            total,
            stats["entries"],
            stats["exits"],
            dt_ms,
        )
        self._heartbeat()
        return stats

    def _cash(self) -> float:
        if isinstance(self.broker, PaperBroker):
            return float(self.broker.cash)
        return float(self.broker.balance_quote_for(self.cfg.quote_asset))

    # ---------------- execution ----------------
    def _open_position(self, symbol: str, price: float, sig: Signal) -> bool:
        mode = self.cfg.mode
        try:
            cash = self._cash()
        except Exception as e:
            log.error("BUY %s skipped: balance unavailable: %s", symbol, e)
            return False
        # ATR fetch serves both ATR stops and risk-based sizing (one fetch).
        atr_val = None
        if self.cfg.risk.atr_stops or self.cfg.risk.risk_per_trade_pct > 0:
            try:
                atr_val = self._atr(self._df(symbol))
            except Exception as e:
                log.warning("ATR fetch failed for %s: %s (using fixed %%)", symbol, e)
        amount = self.risk.position_size(price, cash)
        if self.cfg.risk.risk_per_trade_pct > 0:
            preview = self.risk.entry_levels(price, atr_val)
            budget_cap = min(self.cfg.risk.quote_budget * self.cfg.risk.per_trade_pct / 100.0, cash)
            sized = self.risk.risk_size(
                self.cfg.risk,
                price,
                preview["stop_loss"],
                cash,
                leverage=self.cfg.leverage if self.cfg.market == "futures" else 1,
                max_margin=budget_cap,
            )
            if sized is not None:
                # Broker amount semantics: margin for futures, notional for spot.
                amount = sized["margin"] if self.cfg.market == "futures" else sized["notional"]
                log.info(
                    "%s risk-sized: risk %.2f%% -> qty %.6f (margin %.2f)",
                    symbol,
                    self.cfg.risk.risk_per_trade_pct,
                    sized["qty"],
                    sized["margin"],
                )
        if amount <= 0:
            log.info("skip entry %s: insufficient funds (cash %.2f)", symbol, cash)
            return False
        try:
            fill = self.broker.buy_market(symbol, amount, price)
        except Exception as e:
            log.error("BUY %s failed: %s", symbol, e)
            self.notifier.error(f"[{mode}] BUY {symbol} FAILED: {e}")
            return False
        levels = self.risk.entry_levels(fill["price"], atr_val)
        try:
            pos_id = self.store.open_position(
                {
                    "symbol": symbol,
                    "strategy": sig.strategy,
                    "entry_price": fill["price"],
                    "qty": fill["qty"],
                    "stop_loss": levels["stop_loss"],
                    "take_profit": levels["take_profit"],
                    "mode": mode,
                }
            )
        except Exception as e:
            # Fill happened but journaling failed - alert loudly (position exists
            # on exchange but not in DB). Operator must reconcile manually.
            log.error("JOURNAL FAILED after BUY %s fill @ %s: %s", symbol, fill["price"], e)
            self.notifier.error(f"[{mode}] BUY {symbol} filled but JOURNAL FAILED: {e}")
            return False
        self._arm_exchange_stops(symbol, levels["stop_loss"], levels["take_profit"], pos_id)
        self.risk.note_entry(symbol)
        try:
            self.store.insert_trade(
                {
                    "ts": utcnow(),
                    "mode": mode,
                    "symbol": symbol,
                    "side": "BUY",
                    "qty": fill["qty"],
                    "price": fill["price"],
                    "quote_amt": fill["quote_amt"],
                    "fee": fill["fee"],
                    "reason": sig.reason,
                    "exchange_order_id": str(fill.get("order_id") or ""),
                    "raw": "",
                }
            )
        except Exception as e:
            log.error("trade journal failed (BUY %s): %s", symbol, e)
        msg = format_buy(
            mode,
            symbol,
            fill["qty"],
            fill["price"],
            fill["quote_amt"],
            self.cfg.quote_asset,
            levels["stop_loss"],
            levels["take_profit"],
            sig.reason,
            self.cfg.leverage,
            self.cfg.market,
        )
        log.info(msg)
        self.notifier.trade(msg)
        if isinstance(self.broker, PaperBroker):
            with contextlib.suppress(Exception):
                self.store.set_paper_state(self.broker.cash)
        return True

    # ---------------- exchange safety net ----------------
    def _arm_exchange_stops(
        self, symbol: str, stop_loss: float | None, take_profit: float | None, pos_id: int
    ) -> None:
        """Place exchange-native STOP/TP (live futures only). Non-fatal:
        on failure the position stays protected by bot-side risk only."""
        if not self.cfg.exchange_stops or self.cfg.mode != "live":
            return
        if not isinstance(self.broker, FuturesBroker):
            return
        try:
            ids = self.broker.place_protection_orders(symbol, stop_loss or 0.0, take_profit or 0.0)
        except Exception as e:
            log.error("exchange stops FAILED for %s (bot-side only): %s", symbol, e)
            self.notifier.error(f"[{self.cfg.mode}] {symbol} exchange stops FAILED: {e}")
            return
        try:
            self.store.update_protection_orders(
                pos_id,
                ids.get("stop_order_id"),
                ids.get("take_order_id"),
                stop_trigger=ids.get("stop_trigger"),
            )
        except Exception as e:
            log.error("protection journal failed for %s: %s", symbol, e)

    def _disarm_exchange_stops(self, pos: dict) -> None:
        """Best-effort cancel of a position's exchange stops (post-close)."""
        if not isinstance(self.broker, FuturesBroker):
            return
        oids = [pos.get("stop_order_id"), pos.get("take_order_id")]
        if not any(oids):
            return
        try:
            self.broker.cancel_protection_orders(pos["symbol"], oids)
        except Exception as e:
            log.warning("cancel protection failed for %s: %s", pos["symbol"], e)

    def _maybe_sync_exchange_stop(self, snapshot: dict) -> None:
        """Push a ratcheted journal SL up to the exchange (live futures only).

        Compares the fresh journal SL against the stored armed trigger; on a
        meaningful raise, cancels the old STOP then places the new one (the
        endpoint rejects overlapping closePosition stops, so place-first is
        impossible - the window is ~one round trip). Failures are non-fatal:
        bot-side SL still enforces, and the next cycle retries.
        """
        if not self.cfg.exchange_stops or self.cfg.mode != "live":
            return
        if not isinstance(self.broker, FuturesBroker):
            return
        try:
            fresh = self.store.get_open_position(snapshot["symbol"], self.cfg.mode)
        except Exception as e:
            log.warning("sync read failed for %s: %s", snapshot.get("symbol"), e)
            return
        if not fresh or not fresh.get("stop_order_id"):
            return
        try:
            journal_sl = float(fresh["stop_loss"]) if fresh.get("stop_loss") else 0.0
        except (TypeError, ValueError):
            return
        if journal_sl <= 0:
            return
        old_trigger = fresh.get("exchange_stop_price")
        try:
            old_trigger = float(old_trigger) if old_trigger is not None else 0.0
        except (TypeError, ValueError):
            old_trigger = 0.0
        base = max(old_trigger, float(snapshot.get("stop_loss") or 0.0))
        if journal_sl <= base + 1e-9:
            return  # no meaningful raise since the armed trigger
        try:
            res = self.broker.replace_protection_stop(
                fresh["symbol"], fresh.get("stop_order_id"), journal_sl
            )
        except Exception as e:
            log.error(
                "exchange stop sync FAILED for %s (bot-side SL %.6g still guards): %s",
                fresh["symbol"],
                journal_sl,
                e,
            )
            key = (fresh["symbol"], round(journal_sl, 8))
            if key not in self._sync_notified:
                self._sync_notified.add(key)
                self.notifier.error(
                    f"[{self.cfg.mode}] {fresh['symbol']} exchange stop sync FAILED "
                    f"(bot-side SL {journal_sl:.6g} guards, retrying): {e}"
                )
            return
        try:
            self.store.update_protection_orders(
                fresh["id"],
                res.get("algo_id"),
                fresh.get("take_order_id"),
                stop_trigger=res.get("trigger"),
            )
        except Exception as e:
            log.error("sync journal failed for %s: %s", fresh["symbol"], e)
            return
        log.info(
            "%s exchange stop synced %.6g -> %.6g (algo %s)",
            fresh["symbol"],
            old_trigger,
            res.get("trigger"),
            res.get("algo_id"),
        )

    # Fresh fills need a grace window: testnet fill reporting lags, so a
    # position younger than this is never reconciled away.
    RECONCILE_GRACE_MIN = 10.0

    def _reconcile_cycle(self, quotes: dict[str, float]) -> None:
        """Per-cycle journal-vs-exchange audit (live futures only, #12).

        - Journal-open but exchange-flat (and older than grace): reconcile-close.
        - Exchange-open but journal-flat on a watchlist symbol: alert once/day
          (manual position - never auto-managed, never auto-closed).
        """
        if self.cfg.mode != "live" or not isinstance(self.broker, FuturesBroker):
            return
        try:
            risks = self.client.futures_position_risk()
        except Exception as e:
            log.warning("reconcile scan failed: %s", e)
            return
        amts: dict[str, float] = {}
        for r in risks or []:
            try:
                amt = abs(float(r.get("positionAmt", 0) or 0))
            except (TypeError, ValueError):
                continue
            if r.get("symbol"):
                amts[r["symbol"]] = amt
        mode = self.cfg.mode
        journal_syms = set()
        for pos in self.store.open_positions(mode):
            sym = pos["symbol"]
            journal_syms.add(sym)
            if amts.get(sym, 0.0) > 0:
                continue
            age = RiskManager._position_age_min(pos)
            if age is not None and age < self.RECONCILE_GRACE_MIN:
                continue
            price = quotes.get(sym, pos["entry_price"])
            if self._reconcile_flat(pos, price):
                log.info("%s ghost position reconciled (exchange flat)", sym)
        for sym, amt in amts.items():
            if amt <= 0 or sym in journal_syms or sym not in self.cfg.symbols:
                continue
            key = f"unknownpos:{mode}:{sym}:{self.risk._today_key()}"
            try:
                if self.store.get_meta(key) == "1":
                    continue
                self.store.set_meta(key, "1")
            except Exception:
                pass
            self.notifier.send(
                f"[{mode.upper()}] UNKNOWN exchange position {sym} amt={amt:g} "
                "(manual entry? NOT managed, NOT auto-closed)",
                event="warning",
            )

    def _reconcile_flat(self, pos: dict, price: float) -> bool:
        """Journal-close when the exchange shows no position.

        Happens when our exchange stop fired while the bot was away (or a
        manual close): the close order then fails, but economically we are
        flat. Returns True when reconciled.
        """
        if not isinstance(self.broker, FuturesBroker):
            return False
        symbol = pos["symbol"]
        try:
            risks = self.client.futures_position_risk(symbol=symbol)
        except Exception as e:
            log.warning("reconcile query failed for %s: %s", symbol, e)
            return False
        amt = 0.0
        for r in risks or []:
            try:
                amt = max(amt, abs(float(r.get("positionAmt", 0) or 0)))
            except (TypeError, ValueError):
                continue
        if amt > 0:
            return False  # genuinely still open - retry next cycles
        exit_price = float(price)
        qty = float(pos["qty"])
        pnl = (exit_price - float(pos["entry_price"])) * qty  # fee unknown here
        reason = "reconciled: exchange already flat (stop fired while away), fee excluded"
        try:
            self.store.close_position(pos["id"], exit_price, pnl)
            self.store.insert_trade(
                {
                    "ts": utcnow(),
                    "mode": self.cfg.mode,
                    "symbol": symbol,
                    "side": "SELL",
                    "qty": qty,
                    "price": exit_price,
                    "quote_amt": qty * exit_price,
                    "fee": 0.0,
                    "reason": reason,
                    "exchange_order_id": "",
                    "raw": "",
                }
            )
        except Exception as e:
            log.error("reconcile journal failed for %s: %s", symbol, e)
            return False
        pct = (exit_price / pos["entry_price"] - 1) * 100 if pos["entry_price"] else 0.0
        msg = format_sell(
            self.cfg.mode,
            symbol,
            qty,
            exit_price,
            pnl,
            self.cfg.quote_asset,
            pct,
            reason,
            self.cfg.leverage,
            self.cfg.market,
        )
        log.info(msg)
        self.notifier.trade(msg)
        return True

    def _close_position(self, pos: dict, price: float, reason: str) -> bool:
        mode = self.cfg.mode
        symbol = pos["symbol"]
        try:
            fill = self.broker.sell_market(
                symbol, pos["qty"], price, entry_price=pos["entry_price"]
            )
        except Exception as e:
            log.error("SELL %s failed: %s", symbol, e)
            if self._reconcile_flat(pos, price):
                return True
            self.notifier.error(f"[{mode}] SELL {symbol} FAILED: {e}")
            return False
        # Our fill closed it - withdraw the exchange safety net (best effort).
        self._disarm_exchange_stops(pos)
        exit_price = float(fill["price"])
        pnl = (
            float(fill["gross"])
            - float(pos["entry_price"]) * float(pos["qty"])
            - float(fill["fee"])
        )
        try:
            self.store.close_position(pos["id"], exit_price, pnl)
            self.store.insert_trade(
                {
                    "ts": utcnow(),
                    "mode": mode,
                    "symbol": symbol,
                    "side": "SELL",
                    "qty": fill["qty"],
                    "price": exit_price,
                    "quote_amt": fill["gross"],
                    "fee": fill["fee"],
                    "reason": reason,
                    "exchange_order_id": str(fill.get("order_id") or ""),
                    "raw": "",
                }
            )
        except Exception as e:
            log.error("JOURNAL FAILED after SELL %s: %s", symbol, e)
            self.notifier.error(f"[{mode}] SELL {symbol} filled but JOURNAL FAILED: {e}")
            return False
        pct = (exit_price / pos["entry_price"] - 1) * 100 if pos["entry_price"] else 0.0
        msg = format_sell(
            mode,
            symbol,
            fill["qty"],
            exit_price,
            pnl,
            self.cfg.quote_asset,
            pct,
            reason,
            self.cfg.leverage,
            self.cfg.market,
        )
        log.info(msg)
        self.notifier.trade(msg)
        if isinstance(self.broker, PaperBroker):
            with contextlib.suppress(Exception):
                self.store.set_paper_state(self.broker.cash)
        return True

    # ---------------- ops ----------------
    def _heartbeat(self) -> None:
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        except Exception:
            pass

    def cycle_stats(self) -> dict[str, float]:
        ms = self._cycle_ms
        return {
            "cycles": float(self._cycles),
            "avg_ms": sum(ms) / len(ms) if ms else 0.0,
            "max_ms": max(ms) if ms else 0.0,
        }

    def _restore_state(self) -> None:
        positions = self.store.open_positions(self.cfg.mode)
        for pos in positions:
            opened = pos.get("opened_ts")
            if not opened:
                continue
            try:
                dt = datetime.strptime(opened, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                self.risk.note_entry_ts(pos["symbol"], dt.timestamp())
            except Exception:
                pass
        if positions:
            log.info("restored %d open position(s) + cooldowns from journal", len(positions))
        # Arm exchange stops for restored positions that predate the feature
        # (or whose placement failed): the journal SL/TP becomes the backstop.
        if self.cfg.mode == "live" and isinstance(self.broker, FuturesBroker):
            for pos in positions:
                if pos.get("stop_order_id") or pos.get("take_order_id"):
                    continue
                if not pos.get("stop_loss"):
                    continue
                log.info("arming exchange stops for restored %s", pos["symbol"])
                self._arm_exchange_stops(
                    pos["symbol"], pos.get("stop_loss"), pos.get("take_profit"), pos["id"]
                )

    def run_forever(self) -> None:
        log.info(
            "starting glmbot: mode=%s market=%s symbols=%s strategies=%s interval=%ss",
            self.cfg.mode,
            self.cfg.market,
            self.cfg.symbols,
            [s.name for s in self.strategies],
            self.cfg.update_interval_sec,
        )
        warnings = []
        try:
            warnings = self.preflight()
        except RuntimeError as e:
            log.error("preflight FAILED: %s", e)
            self.notifier.error(f"glmbot preflight failed: {e}")
            raise
        for w in warnings:
            log.warning("preflight: %s", w)
        self._restore_state()
        self.notifier.send(
            f"glmbot started | {self.cfg.env_label} | "
            f"symbols={','.join(self.cfg.symbols)} | "
            f"strategies={','.join(s.name for s in self.strategies)}",
            event="startup",
        )
        offline_cycles = 0
        while not self._stop:
            start = time.time()
            try:
                quotes = self._quotes()
                if not quotes and offline_cycles >= 3:
                    log.warning("offline for %d cycles - exits unmanaged, retrying", offline_cycles)
                self.run_once()
                self._cycles += 1
                offline_cycles = 0 if quotes else offline_cycles + 1
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.exception("cycle error: %s", e)
                with contextlib.suppress(Exception):
                    self.notifier.error(f"glmbot cycle error: {e}")
            if self._stop:
                break
            elapsed = time.time() - start
            sleep_s = max(1.0, self.cfg.update_interval_sec - elapsed)
            log.debug("sleeping %.1fs", sleep_s)
            # interruptible sleep (1s slices) so SIGTERM stops promptly
            for _ in range(int(sleep_s)):
                if self._stop:
                    break
                time.sleep(1.0)
            remainder = sleep_s - int(sleep_s)
            if remainder > 0 and not self._stop:
                time.sleep(remainder)
        log.info("glmbot stopped after %d cycles", self._cycles)
        self.notifier.send("glmbot stopped", event="shutdown")
