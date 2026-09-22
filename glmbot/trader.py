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

import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .api import BinanceClient
from .broker import Broker, PaperBroker
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
    def __init__(self, cfg: BotConfig, store: Store, client: BinanceClient,
                 broker: Broker, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.client = client
        self.broker = broker
        self.notifier = notifier
        self.strategies = build_strategies(cfg.strategies, cfg.strategy_params)
        self.risk = RiskManager(cfg.risk, store)
        self._stop = False
        self._cycles = 0
        self._cycle_ms: List[float] = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle_stop)
            except (OSError, ValueError):
                pass  # Windows / non-main-thread

    # ---------------- lifecycle ----------------
    def _handle_stop(self, *_a) -> None:
        if not self._stop:
            log.info("shutdown signal received - finishing current cycle...")
        self._stop = True

    def preflight(self) -> List[str]:
        """Startup checks. Returns warnings (empty = clean). Raises on fatal."""
        warnings: List[str] = []
        # 1. clock sync
        try:
            offset = self.client.sync_time()
            if abs(offset) > 5000:
                warnings.append(f"clock offset {offset}ms is large - NTP sync recommended")
        except Exception as e:
            warnings.append(f"time sync failed: {e}")
        # 2. connectivity
        if not self.client.ping():
            raise RuntimeError("Binance REST unreachable (ping failed) - check network/VPN/geo-block")
        # 3. keys in live mode
        if self.cfg.mode == "live" and (not self.cfg.api_key or self.cfg.api_key.startswith("YOUR_")):
            raise RuntimeError("live mode requires api.key (or GLMBOT_API_KEY)")
        # 4. balance sanity
        try:
            cash = self._cash()
            if cash <= 0:
                warnings.append(f"quote balance is {cash:.2f} {self.cfg.quote_asset} - deposits needed?")
            elif cash < self.cfg.risk.quote_budget:
                warnings.append(
                    f"exchange balance {cash:.2f} < configured quote_budget "
                    f"{self.cfg.risk.quote_budget:.2f} - sizing uses the budget; "
                    "entries may fail on insufficient funds")
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
        kl = self.client.klines(symbol, CANDLE_INTERVAL, limit=300)
        if not kl:
            raise RuntimeError(f"empty kline response for {symbol}")
        k = Klines(kl)
        issues = k.validate()
        if issues:
            log.warning("%s kline quality: %s", symbol, "; ".join(issues[:2]))
        return k

    def _atr(self, k: Klines) -> Optional[float]:
        try:
            period = self.cfg.risk.atr_period
            if len(k.close) < period + 1:
                return None
            return atr_fn(k.high, k.low, k.close, period)[-1]
        except Exception as e:
            log.warning("ATR calc failed: %s", e)
            return None

    def _price(self, symbol: str) -> float:
        kl = self.client.klines(symbol, CANDLE_INTERVAL, limit=2)
        if not kl:
            raise RuntimeError(f"no klines for {symbol}")
        age_ms = int(time.time() * 1000) - int(kl[-1].get("close_time", kl[-1]["open_time"]))
        if age_ms > STALE_QUOTE_SEC * 1000:
            log.warning("%s quote is stale (%ds old)", symbol, age_ms // 1000)
        return float(kl[-1]["close"])

    def _consensus(self, symbol: str, df: Optional[Klines] = None) -> Optional[Signal]:
        """Vote-based entry signal.

        ``min_votes=0`` (default) requires ALL strategies to agree;
        ``min_votes=N`` requires >=N BUY with zero SELL dissent.
        Unanimous SELL is returned as an exit signal regardless.
        """
        df = df if df is not None else self._df(symbol)
        votes: List[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.evaluate(symbol, df)
                votes.append(sig)
                try:
                    self.store.log_signal(sig)
                except Exception:
                    pass
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

    def _quotes(self) -> Dict[str, float]:
        # Batch path first (one HTTP call), per-symbol fallback for resilience.
        try:
            if hasattr(self.client, "ticker_prices"):
                return self.client.ticker_prices(list(self.cfg.symbols))
        except Exception as e:
            log.debug("batch price fetch failed, falling back per-symbol: %s", e)
        out: Dict[str, float] = {}
        for s in self.cfg.symbols:
            try:
                out[s] = self._price(s)
            except Exception as e:
                log.warning("price fetch failed for %s: %s", s, e)
        return out

    # ---------------- main loop ----------------
    def run_once(self) -> Dict[str, int]:
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
            try:
                plan = self.risk.check_exit(pos, price)
            except Exception as e:
                log.error("exit check failed for %s: %s", symbol, e)
                stats["errors"] += 1
                continue
            if plan.action == "exit":
                self._close_position(pos, price, plan.reason)
                stats["exits"] += 1
                continue
            try:
                sig = self._consensus(symbol)
            except Exception as e:
                log.error("signal eval failed for %s: %s", symbol, e)
                stats["errors"] += 1
                continue
            if sig is not None and sig.side == SELL:
                self._close_position(pos, price, f"signal exit: {sig.reason}")
                stats["exits"] += 1

        # 2) entries
        if self.risk.tripped_today(mode):
            log.warning("kill switch active for today - no new entries")
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
                    ok, why = self.risk.can_open(symbol, mode)
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
            pos["qty"] * quotes.get(pos["symbol"], pos["entry_price"])
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
        dt_ms = (time.time() - t0) * 1000
        self._cycle_ms.append(dt_ms)
        if len(self._cycle_ms) > 50:
            self._cycle_ms.pop(0)
        log.info("cycle done: cash=%.2f positions=%.2f total=%.2f (+%d/-%d, %.0fms)",
                 cash, pos_val, total, stats["entries"], stats["exits"], dt_ms)
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
        amount = self.risk.position_size(price, cash)
        if amount <= 0:
            log.info("skip entry %s: insufficient funds (cash %.2f)", symbol, cash)
            return False
        atr_val = None
        if self.cfg.risk.atr_stops:
            try:
                atr_val = self._atr(self._df(symbol))
            except Exception as e:
                log.warning("ATR fetch failed for %s: %s (using fixed %%)", symbol, e)
        try:
            fill = self.broker.buy_market(symbol, amount, price)
        except Exception as e:
            log.error("BUY %s failed: %s", symbol, e)
            self.notifier.error(f"[{mode}] BUY {symbol} FAILED: {e}")
            return False
        levels = self.risk.entry_levels(fill["price"], atr_val)
        try:
            self.store.open_position({
                "symbol": symbol, "strategy": sig.strategy,
                "entry_price": fill["price"], "qty": fill["qty"],
                "stop_loss": levels["stop_loss"], "take_profit": levels["take_profit"],
                "mode": mode,
            })
        except Exception as e:
            # Fill happened but journaling failed - alert loudly (position exists
            # on exchange but not in DB). Operator must reconcile manually.
            log.error("JOURNAL FAILED after BUY %s fill @ %s: %s", symbol, fill["price"], e)
            self.notifier.error(f"[{mode}] BUY {symbol} filled but JOURNAL FAILED: {e}")
            return False
        self.risk.note_entry(symbol)
        try:
            self.store.insert_trade({
                "ts": utcnow(), "mode": mode, "symbol": symbol, "side": "BUY",
                "qty": fill["qty"], "price": fill["price"], "quote_amt": fill["quote_amt"],
                "fee": fill["fee"], "reason": sig.reason,
                "exchange_order_id": str(fill.get("order_id") or ""), "raw": "",
            })
        except Exception as e:
            log.error("trade journal failed (BUY %s): %s", symbol, e)
        msg = format_buy(mode, symbol, fill["qty"], fill["price"], fill["quote_amt"],
                         self.cfg.quote_asset, levels["stop_loss"], levels["take_profit"],
                         sig.reason, self.cfg.leverage, self.cfg.market)
        log.info(msg)
        self.notifier.trade(msg)
        if isinstance(self.broker, PaperBroker):
            try:
                self.store.set_paper_state(self.broker.cash)
            except Exception:
                pass
        return True

    def _close_position(self, pos: Dict, price: float, reason: str) -> bool:
        mode = self.cfg.mode
        symbol = pos["symbol"]
        try:
            fill = self.broker.sell_market(symbol, pos["qty"], price,
                                           entry_price=pos["entry_price"])
        except Exception as e:
            log.error("SELL %s failed: %s", symbol, e)
            self.notifier.error(f"[{mode}] SELL {symbol} FAILED: {e}")
            return False
        exit_price = float(fill["price"])
        pnl = float(fill["gross"]) - float(pos["entry_price"]) * float(pos["qty"]) - float(fill["fee"])
        try:
            self.store.close_position(pos["id"], exit_price, pnl)
            self.store.insert_trade({
                "ts": utcnow(), "mode": mode, "symbol": symbol, "side": "SELL",
                "qty": fill["qty"], "price": exit_price, "quote_amt": fill["gross"],
                "fee": fill["fee"], "reason": reason,
                "exchange_order_id": str(fill.get("order_id") or ""), "raw": "",
            })
        except Exception as e:
            log.error("JOURNAL FAILED after SELL %s: %s", symbol, e)
            self.notifier.error(f"[{mode}] SELL {symbol} filled but JOURNAL FAILED: {e}")
            return False
        pct = (exit_price / pos["entry_price"] - 1) * 100 if pos["entry_price"] else 0.0
        msg = format_sell(mode, symbol, fill["qty"], exit_price, pnl,
                          self.cfg.quote_asset, pct, reason,
                          self.cfg.leverage, self.cfg.market)
        log.info(msg)
        self.notifier.trade(msg)
        if isinstance(self.broker, PaperBroker):
            try:
                self.store.set_paper_state(self.broker.cash)
            except Exception:
                pass
        return True

    # ---------------- ops ----------------
    def _heartbeat(self) -> None:
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        except Exception:
            pass

    def cycle_stats(self) -> Dict[str, float]:
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
                dt = datetime.strptime(opened, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc)
                self.risk.note_entry_ts(pos["symbol"], dt.timestamp())
            except Exception:
                pass
        if positions:
            log.info("restored %d open position(s) + cooldowns from journal",
                     len(positions))

    def run_forever(self) -> None:
        log.info("starting glmbot: mode=%s market=%s symbols=%s strategies=%s interval=%ss",
                 self.cfg.mode, self.cfg.market, self.cfg.symbols,
                 [s.name for s in self.strategies], self.cfg.update_interval_sec)
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
                    log.warning("offline for %d cycles - exits unmanaged, retrying",
                                offline_cycles)
                self.run_once()
                self._cycles += 1
                offline_cycles = 0 if quotes else offline_cycles + 1
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.exception("cycle error: %s", e)
                try:
                    self.notifier.error(f"glmbot cycle error: {e}")
                except Exception:
                    pass
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
