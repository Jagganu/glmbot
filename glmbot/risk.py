"""Position sizing, entry gating and exit management.

Order of exit checks (first hit wins):
  0. breakeven lock (once trigger reached, SL ratchets to entry + buffer)
  1. hard stop-loss -> 2. take-profit -> 3. trailing stop -> 4. time stop.

Trailing stop *ratchets*: as price climbs, ``stop_loss`` is raised to
``trail_high * (1 - trailing_pct)`` and persisted via ``Store.update_trail``.
It never moves down. Breakeven is the same ratchet idea, applied once.

Entry gates (in order): max positions -> daily kill switch -> daily trade
budget -> per-symbol cooldown.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from .config import RiskCfg
from .storage import Store

log = logging.getLogger("glmbot.risk")

MIN_NOTIONAL_QUOTE = 10.0  # Binance practical minimum; mirrors position_size floor


@dataclass
class ExitPlan:
    action: str  # "exit" | "hold"
    reason: str
    trigger_price: float | None = None

    @property
    def should_exit(self) -> bool:
        return self.action == "exit"


class RiskManager:
    """Stateless-except-cooldowns risk engine backed by :class:`Store`."""

    def __init__(self, cfg: RiskCfg, store: Store):
        self.cfg = cfg
        self.store = store
        self._last_entry_ts: dict[str, float] = {}

    # ---------------- entries ----------------
    def halted(self, mode: str) -> tuple[bool, str]:
        """Any latched halt for today: daily cap, consecutive losses, drawdown."""
        if self.tripped_today(mode):
            return True, "daily loss cap tripped - trading halted until tomorrow (UTC)"
        if self._latched(f"consloss:{mode}"):
            return True, (
                f"{self.cfg.consecutive_loss_halt} consecutive losses - "
                "trading halted until tomorrow (UTC)"
            )
        if self._latched(f"drawdown:{mode}"):
            return True, (
                f"peak drawdown {self.cfg.max_drawdown_halt_pct:g}% hit - "
                "trading halted until tomorrow (UTC)"
            )
        return False, ""

    def _latch_key(self, stem: str) -> str:
        return f"{stem}:{self._today_key()}"

    def _latched(self, stem: str) -> bool:
        return self.store.get_meta(self._latch_key(stem)) == "1"

    def _latch(self, stem: str) -> None:
        self.store.set_meta(self._latch_key(stem), "1")

    def can_open(self, symbol: str, mode: str) -> tuple[bool, str]:
        """Return (allowed, human-readable reason). Empty reason when allowed."""
        open_pos = self.store.open_positions(mode)
        if len(open_pos) >= self.cfg.max_open_positions:
            return (
                False,
                f"max_open_positions reached ({len(open_pos)}/{self.cfg.max_open_positions})",
            )
        halted, why = self.halted(mode)
        if halted:
            return False, why
        if self.cfg.max_daily_trades > 0:
            today = self._today_key()
            n_today = self._count_entries_today(mode, today)
            if n_today >= self.cfg.max_daily_trades:
                return False, f"daily trade budget used ({n_today}/{self.cfg.max_daily_trades})"
        last = self._last_entry_ts.get(symbol)
        if last and time.time() - last < self.cfg.cooldown_min * 60:
            wait = int(self.cfg.cooldown_min - (time.time() - last) / 60) + 1
            return False, f"cooldown: {wait}m left for {symbol} ({self.cfg.cooldown_min}m)"
        return True, ""

    def _count_entries_today(self, mode: str, today: str) -> int:
        try:
            trades = self.store.trades(mode=mode, limit=1000)
            return sum(
                1 for t in trades if t.get("side") == "BUY" and str(t.get("ts", ""))[:10] == today
            )
        except Exception:
            return 0

    # ---------------- daily loss cap (kill switch) ----------------
    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def tripped_today(self, mode: str) -> bool:
        if self.cfg.daily_loss_cap_pct <= 0:
            return False
        return self.store.get_meta(f"killswitch:{mode}:{self._today_key()}") == "1"

    def check_daily_loss(self, mode: str, current_total: float) -> bool:
        """Trip the kill switch if drawdown vs today's first snapshot >= cap.

        Returns True exactly when the switch *trips on this call*.
        """
        cap = self.cfg.daily_loss_cap_pct
        if cap <= 0:
            return False
        baseline = self._day_baseline(mode)
        if baseline is None or baseline <= 0:
            return False
        loss_pct = (1 - current_total / baseline) * 100
        if loss_pct >= cap:
            self.store.set_meta(f"killswitch:{mode}:{self._today_key()}", "1")
            log.warning(
                "DAILY LOSS CAP HIT: -%.2f%% (cap %.2f%%, baseline %.2f -> now %.2f) "
                "- halting new entries until tomorrow (UTC)",
                loss_pct,
                cap,
                baseline,
                current_total,
            )
            return True
        return False

    def _day_baseline(self, mode: str) -> float | None:
        """First equity snapshot of the current UTC day, else latest."""
        rows = self.store.equity_history(mode, limit=500)
        if not rows:
            return None
        today = self._today_key()
        todays = [r for r in rows if str(r.get("ts") or "").startswith(today)]
        if todays:
            return float(todays[0]["total"])
        return float(rows[-1]["total"])

    # ---------------- stronger kill switches ----------------
    def consecutive_losses(self, mode: str) -> int:
        """Straight losing closes, most-recent first (unknown PnL breaks streak)."""
        streak = 0
        for pos in self.store.closed_positions(mode):
            pnl = pos.get("pnl_quote")
            if pnl is None:
                break
            try:
                if float(pnl) < 0:
                    streak += 1
                    continue
            except (TypeError, ValueError):
                pass
            break
        return streak

    def check_consecutive_losses(self, mode: str) -> bool:
        """Latch a day-halt after N straight losing closes. True when tripped now."""
        need = self.cfg.consecutive_loss_halt
        if need <= 0 or self._latched(f"consloss:{mode}"):
            return False
        if self.consecutive_losses(mode) >= need:
            self._latch(f"consloss:{mode}")
            log.warning(
                "CONSECUTIVE LOSS HALT: %d straight losses - halting new entries until tomorrow",
                need,
            )
            return True
        return False

    def check_drawdown_halt(self, mode: str, current_total: float) -> bool:
        """Latch a day-halt after peak-equity drawdown reaches the cap."""
        cap = self.cfg.max_drawdown_halt_pct
        if cap <= 0 or self._latched(f"drawdown:{mode}"):
            return False
        rows = self.store.equity_history(mode, limit=1000)
        if not rows:
            return False
        peak = max(float(r["total"]) for r in rows)
        if peak <= 0:
            return False
        dd = (1 - current_total / peak) * 100
        if dd >= cap:
            self._latch(f"drawdown:{mode}")
            log.warning(
                "DRAWDOWN HALT: -%.2f%% vs peak %.2f (cap %.2f%%) - halting until tomorrow",
                dd,
                peak,
                cap,
            )
            return True
        return False

    # ---------------- sizing ----------------
    def position_size(self, price: float, cash_available: float) -> float:
        """Quote-currency amount to risk on the next entry (0 = skip).

        Rule: ``min(budget * per_trade_pct, cash_available)``, floored at the
        Binance practical minimum (~10 USDT notional). Rounded to cents.
        """
        if price <= 0 or cash_available <= 0:
            return 0.0
        budget = min(
            self.cfg.quote_budget * self.cfg.per_trade_pct / 100.0,
            cash_available,
        )
        if budget < MIN_NOTIONAL_QUOTE:
            return 0.0
        return round(budget, 2)

    @staticmethod
    def risk_size(
        cfg: RiskCfg,
        entry: float,
        stop: float,
        equity: float,
        leverage: int = 1,
        max_margin: float | None = None,
    ) -> dict[str, float] | None:
        """Constant-risk sizing: risk ``risk_per_trade_pct`` of equity.

        ``qty = risk_amount / SL_distance``; margin capped by ``max_margin``
        (the per-trade budget) so wide-stop configs can't explode notional.
        Returns ``{qty, margin, notional}`` or None when disabled/degenerate
        (zero distance, unaffordable minimum). Callers pass ``margin`` to
        futures brokers and ``notional`` to spot brokers.
        """
        risk_pct = cfg.risk_per_trade_pct
        if risk_pct <= 0 or equity <= 0 or entry <= 0:
            return None
        dist = entry - stop
        if dist <= 0:
            return None
        qty = equity * risk_pct / 100.0 / dist
        lev = max(1, leverage)
        notional = qty * entry
        margin = notional / lev
        if max_margin is not None and margin > max_margin > 0:
            scale = max_margin / margin
            qty *= scale
            notional *= scale
            margin = max_margin
        if margin < MIN_NOTIONAL_QUOTE:
            return None
        return {"qty": qty, "margin": round(margin, 2), "notional": round(notional, 2)}

    def entry_levels(self, price: float, atr: float | None = None) -> dict[str, float]:
        """Stop-loss / take-profit for a fresh entry.

        ATR mode (when enabled *and* a valid ATR is supplied) adapts to
        volatility: ``SL = entry − sl_mult|ATR``. Otherwise fixed percentages.
        """
        if self.cfg.atr_stops and atr and atr > 0:
            sl = price - self.cfg.atr_sl_mult * atr
            tp = price + self.cfg.atr_tp_mult * atr
            # Guard against pathological ATR (e.g. bad feed -> negative SL).
            if sl <= 0 or sl >= price:
                sl = price * (1 - self.cfg.stop_loss_pct / 100.0)
            if tp <= price:
                tp = price * (1 + self.cfg.take_profit_pct / 100.0)
        else:
            sl = price * (1 - self.cfg.stop_loss_pct / 100.0)
            tp = price * (1 + self.cfg.take_profit_pct / 100.0)
        return {"stop_loss": round(sl, 8), "take_profit": round(tp, 8)}

    def note_entry(self, symbol: str) -> None:
        self._last_entry_ts[symbol] = time.time()

    def note_entry_ts(self, symbol: str, ts: float) -> None:
        """Restore cooldown state from a persisted timestamp (restart recovery)."""
        prev = self._last_entry_ts.get(symbol, 0.0)
        if ts > prev:
            self._last_entry_ts[symbol] = ts

    # ---------------- exits ----------------
    def check_exit(self, pos: dict, price: float) -> ExitPlan:
        cfg = self.cfg
        entry = float(pos["entry_price"])
        sl, tp = pos.get("stop_loss"), pos.get("take_profit")
        sl = float(sl) if sl is not None else None
        tp = float(tp) if tp is not None else None

        # 0. breakeven lock: once comfortably in profit, the worst case
        # becomes entry + buffer (persisted like the trailing ratchet).
        if cfg.breakeven_trigger_pct > 0 and entry > 0:
            be_trigger = entry * (1 + cfg.breakeven_trigger_pct / 100.0)
            if price >= be_trigger:
                be_sl = entry * (1 + cfg.breakeven_buffer_pct / 100.0)
                if sl is None or be_sl > sl:
                    sl = be_sl
                    try:
                        self.store.update_trail(
                            pos["id"], float(pos.get("trail_high") or entry), sl
                        )
                    except Exception as e:
                        log.warning("breakeven persist failed for %s: %s", pos.get("symbol"), e)
                    log.info("%s breakeven locked SL=%.6g", pos.get("symbol"), sl)

        # 1. hard stop-loss
        if sl is not None and price <= sl:
            return ExitPlan("exit", f"stop-loss hit ({price:.6g} <= SL {sl:.6g})", price)

        # 2. take-profit
        if tp is not None and price >= tp:
            return ExitPlan("exit", f"take-profit hit ({price:.6g} >= TP {tp:.6g})", price)

        # 3. trailing stop (ratchets stop up as price climbs)
        if cfg.trailing_stop_pct > 0:
            high = float(pos.get("trail_high") or entry)
            if price > high:
                high = price
                new_sl = high * (1 - cfg.trailing_stop_pct / 100.0)
                if sl is None or new_sl > sl:  # only ratchet upward
                    sl = new_sl
                try:
                    self.store.update_trail(pos["id"], high, sl)
                except Exception as e:
                    log.warning("trail persist failed for %s: %s", pos.get("symbol"), e)
                log.info("%s trail high=%.6g new SL=%.6g", pos.get("symbol"), high, sl)
            trail_sl = high * (1 - cfg.trailing_stop_pct / 100.0)
            if trail_sl > entry and price <= trail_sl:
                return ExitPlan(
                    "exit", f"trailing stop ({price:.6g} <= trail {trail_sl:.6g})", price
                )

        # 4. time stop: exit dead-money positions based on opened_ts age.
        if cfg.max_hold_min > 0:
            age_min = self._position_age_min(pos)
            if age_min is not None and age_min >= cfg.max_hold_min:
                return ExitPlan(
                    "exit",
                    f"time stop ({age_min:.0f}m held, cap {cfg.max_hold_min}m)",
                    price,
                )
        return ExitPlan("hold", "")

    @staticmethod
    def _position_age_min(pos: dict) -> float | None:
        """Minutes since opened_ts (UTC '...Z'), or None if unparseable."""
        opened = pos.get("opened_ts")
        if not opened:
            return None
        try:
            dt = datetime.strptime(opened, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


def round_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    steps = (value / step).to_integral_value(rounding="ROUND_DOWN")
    return steps * step
