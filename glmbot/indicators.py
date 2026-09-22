"""Technical indicators — pure Python, zero heavy deps (Termux-friendly).

Conventions
  - Inputs are plain ``list[float]`` (or anything list()-able); outputs are
    ``list[float | None]`` aligned with the input (``None`` = warmup).
  - All functions validate ``period >= 1`` and return ``[None]*n`` when there
    is insufficient history instead of raising.
  - ``atr`` aligns to Wilder's smoothing: first valid value at index ``period``.

Available: sma, ema, rsi, macd, bollinger, atr, stoch_rsi, vwap,
supertrend, donchian.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def _seq(values: Sequence[float]) -> List[float]:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"indicator input must be numeric: {e}") from e


def _warm(n: int) -> List[None]:
    return [None] * n  # type: ignore[return-value]


def sma(values: Sequence[float], period: int) -> List[Optional[float]]:
    v = _seq(values)
    n = len(v)
    out: List[Optional[float]] = [None] * n
    if period < 1 or n < period:
        return out
    s = sum(v[:period])
    out[period - 1] = s / period
    for i in range(period, n):
        s += v[i] - v[i - period]
        out[i] = s / period
    return out


def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    v = _seq(values)
    n = len(v)
    out: List[Optional[float]] = [None] * n
    if period < 1 or n < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = sum(v[:period]) / period
    prev = out[period - 1]
    assert prev is not None
    for i in range(period, n):
        prev = alpha * v[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> List[Optional[float]]:
    v = _seq(values)
    n = len(v)
    out: List[Optional[float]] = [None] * n
    if period < 1 or n < period + 1:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = v[i] - v[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    avg_g = gains / period
    avg_l = losses / period
    out[period] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period + 1, n):
        d = v[i] - v[i - 1]
        avg_g = (avg_g * (period - 1) + (d if d > 0 else 0.0)) / period
        avg_l = (avg_l * (period - 1) + (-d if d < 0 else 0.0)) / period
        out[i] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return out


def macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    v = _seq(values)
    n = len(v)
    f = ema(v, fast)
    s = ema(v, slow)
    line: List[Optional[float]] = [None] * n
    for i in range(n):
        if f[i] is not None and s[i] is not None:
            line[i] = f[i] - s[i]  # type: ignore[operator]
    sig: List[Optional[float]] = [None] * n
    first = slow - 1
    if first >= 0 and n - first >= signal and signal > 0:
        seed_vals = [x for x in line[first:first + signal] if x is not None]
        if len(seed_vals) == signal:
            seed = sum(seed_vals) / signal
            sig[first + signal - 1] = seed
            alpha = 2.0 / (signal + 1)
            prev = seed
            for i in range(first + signal, n):
                if line[i] is None:
                    continue
                prev = alpha * line[i] + (1 - alpha) * prev  # type: ignore[operator]
                sig[i] = prev
    hist: List[Optional[float]] = [None] * n
    for i in range(n):
        if line[i] is not None and sig[i] is not None:
            hist[i] = line[i] - sig[i]  # type: ignore[operator]
    return line, sig, hist


def bollinger(values: Sequence[float], period: int = 20, std_dev: float = 2.0
              ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    v = _seq(values)
    n = len(v)
    mid = sma(v, period)
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    if period >= 1 and n >= period:
        for i in range(period - 1, n):
            m = mid[i]
            assert m is not None
            window = v[i - period + 1:i + 1]
            var = sum((x - m) ** 2 for x in window) / period
            sd = var ** 0.5
            upper[i] = m + std_dev * sd
            lower[i] = m - std_dev * sd
    return upper, mid, lower


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float],
        period: int = 14) -> List[Optional[float]]:
    h, l, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    out: List[Optional[float]] = [None] * n
    if period < 1 or n < period + 1 or len(h) != n or len(l) != n:
        return out
    tr = [h[0] - l[0]]
    for i in range(1, n):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    prev = sum(tr[1:period + 1]) / period
    out[period] = prev  # aligned: ATR known from index `period`
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def stoch_rsi(values: Sequence[float], rsi_period: int = 14,
              stoch_period: int = 14) -> List[Optional[float]]:
    """Stochastic-RSI oscillator in [0, 1] (None during warmup)."""
    v = _seq(values)
    n = len(v)
    out: List[Optional[float]] = [None] * n
    if n < rsi_period + stoch_period:
        return out
    r = rsi(v, rsi_period)
    for i in range(rsi_period + stoch_period - 1, n):
        window = [x for x in r[i - stoch_period + 1:i + 1] if x is not None]
        if len(window) < stoch_period or r[i] is None:
            continue
        lo, hi = min(window), max(window)
        out[i] = 0.5 if hi == lo else (r[i] - lo) / (hi - lo)  # type: ignore[operator]
    return out


def vwap(high: Sequence[float], low: Sequence[float], close: Sequence[float],
         volume: Sequence[float], period: int = 20) -> List[Optional[float]]:
    """Rolling VWAP (typical price × volume) over ``period`` bars."""
    h, l, c, vol = _seq(high), _seq(low), _seq(close), _seq(volume)
    n = len(c)
    out: List[Optional[float]] = [None] * n
    if period < 1 or n < period:
        return out
    tp = [(h[i] + l[i] + c[i]) / 3.0 for i in range(n)]
    for i in range(period - 1, n):
        pv = sum(tp[j] * vol[j] for j in range(i - period + 1, i + 1))
        vv = sum(vol[j] for j in range(i - period + 1, i + 1))
        out[i] = (pv / vv) if vv > 0 else tp[i]
    return out


def supertrend(high: Sequence[float], low: Sequence[float], close: Sequence[float],
               period: int = 10, multiplier: float = 3.0
               ) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    """Supertrend line + direction ('up' | 'down' | None per bar)."""
    h, l, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    line: List[Optional[float]] = [None] * n
    direction: List[Optional[str]] = [None] * n
    if n < period + 1 or period < 1:
        return line, direction
    a = atr(h, l, c, period)
    prev_upper = prev_lower = None
    prev_close = None
    prev_dir = "down"
    for i in range(n):
        if a[i] is None:
            continue
        basic_upper = (h[i] + l[i]) / 2 + multiplier * a[i]  # type: ignore[operator]
        basic_lower = (h[i] + l[i]) / 2 - multiplier * a[i]  # type: ignore[operator]
        if prev_upper is None:
            prev_upper, prev_lower, prev_close = basic_upper, basic_lower, c[i]
            direction[i] = prev_dir
            line[i] = basic_upper
            continue
        upper = basic_upper if (basic_upper < prev_upper or (prev_close is not None and prev_close > prev_upper)) else prev_upper
        lower = basic_lower if (basic_lower > prev_lower or (prev_close is not None and prev_close < prev_lower)) else prev_lower
        if c[i] > upper:
            cur = "up"
        elif c[i] < lower:
            cur = "down"
        else:
            cur = prev_dir
        direction[i] = cur
        line[i] = lower if cur == "up" else upper
        prev_upper, prev_lower, prev_close, prev_dir = upper, lower, c[i], cur
    return line, direction


def donchian(high: Sequence[float], low: Sequence[float],
             period: int = 20) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Donchian channel: (upper, lower) = rolling max(high) / min(low)."""
    h, l = _seq(high), _seq(low)
    n = len(h)
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    if period < 1 or n < period:
        return upper, lower
    for i in range(period - 1, n):
        upper[i] = max(h[i - period + 1:i + 1])
        lower[i] = min(l[i - period + 1:i + 1])
    return upper, lower
