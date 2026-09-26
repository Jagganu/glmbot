"""Market-regime detection: bull / bear / chop / volatile (pure Python).

Used three ways:
  1. Live entry filter (optional ``trading.regime_filter``): skip BUY signals
     when the higher-timeframe regime is hostile (bear / chop).
  2. ``bot.py regime`` CLI: one-line regime diagnosis per watchlist symbol.
  3. Backtest fold labels already carry bull/bear/chop from buy-and-hold;
     this module gives a richer ADX + trend + volatility explanation.

Rule set (transparent, no ML black box):
  - Trend: EMA20 vs EMA50 (+DI vs -DI confirms when ADX is valid).
  - Strength: ADX(14) >= 20 = trending, < 20 = range.
  - Energy: ATR14 / close * 100 >= 3% = volatile (entry filters may skip).
"""

from __future__ import annotations

from .indicators import adx as adx_fn
from .indicators import atr as atr_fn
from .indicators import ema
from .klines import Klines


def detect_regime(
    k: Klines,
    adx_period: int = 14,
    fast: int = 20,
    slow: int = 50,
    volatile_atr_pct: float = 3.0,
) -> dict:
    """Classify the current regime of closed candles ``k``.

    Returns a JSON-serializable dict with ``regime`` in
    {bull, bear, chop, volatile, unknown} plus diagnostics.
    Never raises on short history (returns ``unknown`` with a reason).
    """
    price = float(k.close[-1]) if k.close else 0.0
    if len(k) < max(slow + 2, 2 * adx_period + 2):
        return {
            "regime": "unknown",
            "adx": None,
            "trend": 0,
            "atr_pct": None,
            "reason": f"insufficient history ({len(k)} bars)",
            "price": price,
        }
    f = ema(k.close, fast)
    s = ema(k.close, slow)
    a, pdi, mdi = adx_fn(k.high, k.low, k.close, adx_period)
    try:
        atr_v = atr_fn(k.high, k.low, k.close, 14)[-1]
    except Exception:
        atr_v = None
    if f[-1] is None or s[-1] is None:
        return {
            "regime": "unknown",
            "adx": None,
            "trend": 0,
            "atr_pct": None,
            "reason": "indicator warmup",
            "price": price,
        }
    adx_v = float(a[-1]) if a[-1] is not None else 0.0
    atr_pct = (float(atr_v) / price * 100.0) if atr_v and price else 0.0
    up = f[-1] > s[-1]  # type: ignore[operator]
    trend = 1 if up else -1
    # DI confirmation when available (avoids EMA whipsaw counter-trend longs).
    if pdi[-1] is not None and mdi[-1] is not None:
        plus, minus = pdi[-1], mdi[-1]
        if (up and plus < minus) or (not up and minus < plus):  # type: ignore[operator]
            trend = 0
    if atr_pct >= volatile_atr_pct:
        regime = "volatile"
        reason = f"ATR {atr_pct:.2f}% >= {volatile_atr_pct:g}% (explosive moves, size down)"
    elif adx_v >= 20 and trend == 1:
        regime = "bull"
        reason = f"uptrend + ADX {adx_v:.1f} (trend-following works)"
    elif adx_v >= 20 and trend == -1:
        regime = "bear"
        reason = f"downtrend + ADX {adx_v:.1f} (longs hostile, mean-reversion only)"
    else:
        regime = "chop"
        reason = f"ADX {adx_v:.1f} < 20 (range, breakouts fail)"
    return {
        "regime": regime,
        "adx": round(adx_v, 1),
        "trend": trend,
        "atr_pct": round(atr_pct, 2),
        "reason": reason,
        "price": price,
    }


def regime_allows_entries(regime: str, allow_chop: bool = False) -> tuple[bool, str]:
    """Entry gate for a regime label (used by the trader's HTF filter).

    Bulls always allow longs. Chop is blocked unless ``allow_chop`` is set
    (breakout strategies love chop-entry at the operator's peril). Bear and
    volatile block longs; unknown is allowed (fail-open, never halt on bad data).
    """
    if regime == "bull":
        return True, ""
    if regime == "chop":
        return (True, "") if allow_chop else (False, "regime chop: range market, longs paused")
    if regime == "bear":
        return False, "regime bear: downtrend, longs paused"
    if regime == "volatile":
        return False, "regime volatile: explosive moves, longs paused"
    return True, ""
