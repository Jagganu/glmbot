"""Portfolio / backtest performance metrics — pure Python, no heavy deps.

All functions accept plain float lists. Returns 0.0 (never NaN) on
degenerate input so reports and tables always render.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _clean(curve: Sequence[float]) -> list[float]:
    return [float(x) for x in curve if x is not None and math.isfinite(float(x))]


def returns_from_equity(curve: Sequence[float]) -> list[float]:
    """Simple per-step returns from an equity curve."""
    c = _clean(curve)
    out: list[float] = []
    for i in range(1, len(c)):
        prev = c[i - 1]
        out.append((c[i] / prev - 1.0) if prev else 0.0)
    return out


def sharpe_ratio(curve: Sequence[float], periods_per_year: float = 35_040.0) -> float:
    """Annualized Sharpe of per-step returns (risk-free = 0).

    Default annualization assumes 15-minute bars (4*24*365 = 35,040).
    """
    rets = returns_from_equity(curve)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var) * math.sqrt(periods_per_year)


def sortino_ratio(curve: Sequence[float], periods_per_year: float = 35_040.0) -> float:
    rets = returns_from_equity(curve)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    downside = [r for r in rets if r < 0]
    if len(downside) < 2:
        return 0.0
    var = sum(r * r for r in downside) / len(downside)
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var) * math.sqrt(periods_per_year)


def max_drawdown(curve: Sequence[float]) -> float:
    """Max peak-to-trough drawdown in percent."""
    c = _clean(curve)
    if not c:
        return 0.0
    peak = c[0]
    worst = 0.0
    for x in c:
        peak = max(peak, x)
        if peak > 0:
            worst = max(worst, (peak - x) / peak * 100.0)
    return worst


def profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy(win_rate_pct: float, avg_win: float, avg_loss: float) -> float:
    """Expected value per trade in quote currency."""
    p = win_rate_pct / 100.0
    return p * avg_win - (1 - p) * abs(avg_loss)


def annualized_return(curve: Sequence[float], periods_per_year: float = 35_040.0) -> float:
    """CAGR from an equity curve (15m bars by default). Returns a ratio, not %."""
    c = _clean(curve)
    if len(c) < 2 or c[0] <= 0:
        return 0.0
    try:
        total = c[-1] / c[0]
        if total <= 0:
            return -1.0
        years = (len(c) - 1) / periods_per_year
        if years <= 0:
            return total - 1.0
        return total ** (1.0 / years) - 1.0
    except (OverflowError, ValueError, ZeroDivisionError):
        return 0.0


def calmar_ratio(curve: Sequence[float], periods_per_year: float = 35_040.0) -> float:
    """Annualized return / max-drawdown (0 when flat or no drawdown info)."""
    dd = max_drawdown(curve)
    if dd <= 0:
        return 0.0
    ann = annualized_return(curve, periods_per_year)
    return ann / (dd / 100.0)


def ulcer_index(curve: Sequence[float]) -> float:
    """Ulcer index: RMS of drawdown % (pain index, lower = smoother)."""
    c = _clean(curve)
    if len(c) < 2:
        return 0.0
    peak = c[0]
    sq = 0.0
    for x in c:
        peak = max(peak, x)
        dd = (peak - x) / peak * 100.0 if peak > 0 else 0.0
        sq += dd * dd
    return (sq / len(c)) ** 0.5


def tail_risk(curve: Sequence[float]) -> dict[str, float]:
    """Historical VaR-95 / CVaR-95 of per-step returns (as positive % losses)."""
    rets = returns_from_equity(curve)
    if len(rets) < 10:
        return {"var_95_pct": 0.0, "cvar_95_pct": 0.0}
    ordered = sorted(rets)
    idx = max(0, int(len(ordered) * 0.05) - 1)
    var = -ordered[idx] * 100.0
    tail = ordered[: idx + 1]
    cvar = -(sum(tail) / len(tail)) * 100.0 if tail else 0.0
    return {"var_95_pct": max(0.0, var), "cvar_95_pct": max(0.0, cvar)}


def recovery_factor(curve: Sequence[float]) -> float:
    """Net profit / max-drawdown amount (0 when flat or no drawdown)."""
    c = _clean(curve)
    if len(c) < 2:
        return 0.0
    profit = c[-1] - c[0]
    if profit <= 0:
        return 0.0
    dd_pct = max_drawdown(c)
    if dd_pct <= 0:
        return 0.0
    peak = max(c)
    dd_amt = peak * dd_pct / 100.0
    return profit / dd_amt if dd_amt > 0 else 0.0
