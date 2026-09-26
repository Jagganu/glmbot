"""Technical indicators — pure Python, zero heavy deps (Termux-friendly).

Conventions
  - Inputs are plain ``list[float]`` (or anything list()-able); outputs are
    ``list[float | None]`` aligned with the input (``None`` = warmup).
  - All functions validate ``period >= 1`` and return ``[None]*n`` when there
    is insufficient history instead of raising.
  - ``atr`` aligns to Wilder's smoothing: first valid value at index ``period``.

Available: sma, ema, wma, tema, rsi, macd, bollinger, atr, stoch_rsi,
stoch_osc, vwap, supertrend, donchian, keltner, psar, adx, cci, willr,
obv, mfi, roc, ichimoku.
"""

from __future__ import annotations

from collections.abc import Sequence


def _seq(values: Sequence[float]) -> list[float]:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"indicator input must be numeric: {e}") from e


def _warm(n: int) -> list[None]:
    return [None] * n  # type: ignore[return-value]


def sma(values: Sequence[float], period: int) -> list[float | None]:
    v = _seq(values)
    n = len(v)
    out: list[float | None] = [None] * n
    if period < 1 or n < period:
        return out
    s = sum(v[:period])
    out[period - 1] = s / period
    for i in range(period, n):
        s += v[i] - v[i - period]
        out[i] = s / period
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    v = _seq(values)
    n = len(v)
    out: list[float | None] = [None] * n
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


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    v = _seq(values)
    n = len(v)
    out: list[float | None] = [None] * n
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


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    v = _seq(values)
    n = len(v)
    f = ema(v, fast)
    s = ema(v, slow)
    line: list[float | None] = [None] * n
    for i in range(n):
        if f[i] is not None and s[i] is not None:
            line[i] = f[i] - s[i]  # type: ignore[operator]
    sig: list[float | None] = [None] * n
    first = slow - 1
    if first >= 0 and n - first >= signal and signal > 0:
        seed_vals = [x for x in line[first : first + signal] if x is not None]
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
    hist: list[float | None] = [None] * n
    for i in range(n):
        if line[i] is not None and sig[i] is not None:
            hist[i] = line[i] - sig[i]  # type: ignore[operator]
    return line, sig, hist


def bollinger(
    values: Sequence[float], period: int = 20, std_dev: float = 2.0
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    v = _seq(values)
    n = len(v)
    mid = sma(v, period)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    if period >= 1 and n >= period:
        for i in range(period - 1, n):
            m = mid[i]
            assert m is not None
            window = v[i - period + 1 : i + 1]
            var = sum((x - m) ** 2 for x in window) / period
            sd = var**0.5
            upper[i] = m + std_dev * sd
            lower[i] = m - std_dev * sd
    return upper, mid, lower


def atr(
    high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14
) -> list[float | None]:
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    out: list[float | None] = [None] * n
    if period < 1 or n < period + 1 or len(h) != n or len(lo) != n:
        return out
    tr = [h[0] - lo[0]]
    for i in range(1, n):
        tr.append(max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])))
    prev = sum(tr[1 : period + 1]) / period
    out[period] = prev  # aligned: ATR known from index `period`
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def stoch_rsi(
    values: Sequence[float], rsi_period: int = 14, stoch_period: int = 14
) -> list[float | None]:
    """Stochastic-RSI oscillator in [0, 1] (None during warmup)."""
    v = _seq(values)
    n = len(v)
    out: list[float | None] = [None] * n
    if n < rsi_period + stoch_period:
        return out
    r = rsi(v, rsi_period)
    for i in range(rsi_period + stoch_period - 1, n):
        window = [x for x in r[i - stoch_period + 1 : i + 1] if x is not None]
        if len(window) < stoch_period or r[i] is None:
            continue
        lo, hi = min(window), max(window)
        out[i] = 0.5 if hi == lo else (r[i] - lo) / (hi - lo)  # type: ignore[operator]
    return out


def vwap(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    period: int = 20,
) -> list[float | None]:
    """Rolling VWAP (typical price × volume) over ``period`` bars."""
    h, lo, c, vol = _seq(high), _seq(low), _seq(close), _seq(volume)
    n = len(c)
    out: list[float | None] = [None] * n
    if period < 1 or n < period:
        return out
    tp = [(h[i] + lo[i] + c[i]) / 3.0 for i in range(n)]
    for i in range(period - 1, n):
        pv = sum(tp[j] * vol[j] for j in range(i - period + 1, i + 1))
        vv = sum(vol[j] for j in range(i - period + 1, i + 1))
        out[i] = (pv / vv) if vv > 0 else tp[i]
    return out


def supertrend(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[list[float | None], list[str | None]]:
    """Supertrend line + direction ('up' | 'down' | None per bar)."""
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    line: list[float | None] = [None] * n
    direction: list[str | None] = [None] * n
    if n < period + 1 or period < 1:
        return line, direction
    a = atr(h, lo, c, period)
    prev_upper = prev_lower = None
    prev_close = None
    prev_dir = "down"
    for i in range(n):
        if a[i] is None:
            continue
        basic_upper = (h[i] + lo[i]) / 2 + multiplier * a[i]  # type: ignore[operator]
        basic_lower = (h[i] + lo[i]) / 2 - multiplier * a[i]  # type: ignore[operator]
        if prev_upper is None:
            prev_upper, prev_lower, prev_close = basic_upper, basic_lower, c[i]
            direction[i] = prev_dir
            line[i] = basic_upper
            continue
        upper = (
            basic_upper
            if (basic_upper < prev_upper or (prev_close is not None and prev_close > prev_upper))
            else prev_upper
        )
        lower = (
            basic_lower
            if (basic_lower > prev_lower or (prev_close is not None and prev_close < prev_lower))
            else prev_lower
        )
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


def donchian(
    high: Sequence[float], low: Sequence[float], period: int = 20
) -> tuple[list[float | None], list[float | None]]:
    """Donchian channel: (upper, lower) = rolling max(high) / min(low)."""
    h, lo = _seq(high), _seq(low)
    n = len(h)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    if period < 1 or n < period:
        return upper, lower
    for i in range(period - 1, n):
        upper[i] = max(h[i - period + 1 : i + 1])
        lower[i] = min(lo[i - period + 1 : i + 1])
    return upper, lower


def wma(values: Sequence[float], period: int) -> list[float | None]:
    """Weighted moving average (linear weights, most recent heaviest)."""
    v = _seq(values)
    n = len(v)
    out: list[float | None] = [None] * n
    if period < 1 or n < period:
        return out
    denom = period * (period + 1) / 2.0
    for i in range(period - 1, n):
        window = v[i - period + 1 : i + 1]
        out[i] = sum(x * (j + 1) for j, x in enumerate(window)) / denom
    return out


def tema(values: Sequence[float], period: int) -> list[float | None]:
    """Triple EMA: 3*EMA1 - 3*EMA2 + EMA3 (less lag than EMA)."""
    v = _seq(values)
    n = len(v)
    out: list[float | None] = [None] * n
    if period < 1 or n < period * 3:
        return out
    e1 = ema(v, period)
    e1v = [x if x is not None else 0.0 for x in e1]
    e2 = ema(e1v, period)
    e2v = [x if x is not None else 0.0 for x in e2]
    e3 = ema(e2v, period)
    start = period * 3 - 3
    for i in range(start, n):
        if e1[i] is None or e2[i] is None or e3[i] is None:
            continue
        out[i] = 3 * e1[i] - 3 * e2[i] + e3[i]  # type: ignore[operator]
    return out


def roc(values: Sequence[float], period: int = 12) -> list[float | None]:
    """Rate-of-change in percent: (close / close[-period] - 1) * 100."""
    v = _seq(values)
    n = len(v)
    out: list[float | None] = [None] * n
    if period < 1 or n <= period:
        return out
    for i in range(period, n):
        prev = v[i - period]
        out[i] = (v[i] / prev - 1.0) * 100.0 if prev else 0.0
    return out


def cci(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 20,
) -> list[float | None]:
    """Commodity Channel Index (mean-deviation oscillator, ~[-200, +200])."""
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    out: list[float | None] = [None] * n
    if period < 1 or n < period or len(h) != n or len(lo) != n:
        return out
    tp = [(h[i] + lo[i] + c[i]) / 3.0 for i in range(n)]
    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        m = sum(window) / period
        md = sum(abs(x - m) for x in window) / period
        out[i] = (tp[i] - m) / (0.015 * md) if md else 0.0
    return out


def willr(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Williams %R in [-100, 0] (oversold < -80, overbought > -20)."""
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    out: list[float | None] = [None] * n
    if period < 1 or n < period or len(h) != n or len(lo) != n:
        return out
    for i in range(period - 1, n):
        hh = max(h[i - period + 1 : i + 1])
        ll = min(lo[i - period + 1 : i + 1])
        out[i] = (hh - c[i]) / (hh - ll) * -100.0 if hh != ll else 0.0
    return out


def obv(close: Sequence[float], volume: Sequence[float]) -> list[float]:
    """On-Balance Volume (cumulative signed volume, no warmup)."""
    c, vol = _seq(close), _seq(volume)
    n = len(c)
    out: list[float] = [0.0] * n
    if not n:
        return out
    run = 0.0
    for i in range(n):
        if i == 0:
            run = vol[0] if n and len(vol) > 0 else 0.0
        elif len(vol) > i:
            if c[i] > c[i - 1]:
                run += vol[i]
            elif c[i] < c[i - 1]:
                run -= vol[i]
        out[i] = run
    return out


def mfi(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Money Flow Index 0..100 (volume-weighted RSI; <20 oversold, >80 overbought)."""
    h, lo, c, vol = _seq(high), _seq(low), _seq(close), _seq(volume)
    n = len(c)
    out: list[float | None] = [None] * n
    if period < 1 or n <= period or not (len(h) == len(lo) == len(vol) == n):
        return out
    tp = [(h[i] + lo[i] + c[i]) / 3.0 for i in range(n)]
    mf = [tp[i] * vol[i] for i in range(n)]
    for i in range(period, n):
        pos = neg = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                continue
            if tp[j] > tp[j - 1]:
                pos += mf[j]
            elif tp[j] < tp[j - 1]:
                neg += mf[j]
        out[i] = 100.0 if neg == 0 else 100 - 100 / (1 + pos / neg)
    return out


def keltner(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Keltner channels: EMA(close) +/- mult*ATR (trend-following envelope)."""
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    mid = ema(c, ema_period)
    a = atr(h, lo, c, atr_period)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    for i in range(n):
        if mid[i] is None or a[i] is None:
            continue
        upper[i] = mid[i] + multiplier * a[i]  # type: ignore[operator]
        lower[i] = mid[i] - multiplier * a[i]  # type: ignore[operator]
    return upper, mid, lower


def psar(
    high: Sequence[float],
    low: Sequence[float],
    step: float = 0.02,
    max_af: float = 0.2,
) -> list[float | None]:
    """Parabolic SAR (stop-and-reverse dots). Returns SAR value per bar."""
    h, lo = _seq(high), _seq(low)
    n = len(h)
    out: list[float | None] = [None] * n
    if n < 2 or len(lo) != n or step <= 0 or max_af <= 0:
        return out
    up = True
    af = step
    sar = lo[0]
    ep = h[0]
    out[0] = sar
    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if up:
            sar = min(sar, lo[i - 1], lo[i] if i >= 1 else lo[i - 1])
            if lo[i] < sar:
                up = False
                sar = ep
                ep = lo[i]
                af = step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, max_af)
        else:
            sar = max(sar, h[i - 1], h[i] if i >= 1 else h[i - 1])
            if h[i] > sar:
                up = True
                sar = ep
                ep = h[i]
                af = step
            else:
                if lo[i] < ep:
                    ep = lo[i]
                    af = min(af + step, max_af)
        out[i] = sar
    return out


def stoch_osc(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[list[float | None], list[float | None]]:
    """Classic stochastic %K / %D in 0..100 (<20 oversold, >80 overbought)."""
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    k: list[float | None] = [None] * n
    d: list[float | None] = [None] * n
    if k_period < 1 or d_period < 1 or n < k_period or len(h) != n or len(lo) != n:
        return k, d
    for i in range(k_period - 1, n):
        hh = max(h[i - k_period + 1 : i + 1])
        ll = min(lo[i - k_period + 1 : i + 1])
        k[i] = (c[i] - ll) / (hh - ll) * 100.0 if hh != ll else 50.0
    for i in range(k_period - 1 + d_period - 1, n):
        window = [x for x in k[i - d_period + 1 : i + 1] if x is not None]
        if len(window) == d_period:
            d[i] = sum(window) / d_period
    return k, d


def adx(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """ADX trend strength + directional movement (+DI / -DI), Wilder smoothing.

    Returns (adx, plus_di, minus_di). ADX > 25 = trending, < 20 = chop.
    First valid ADX appears at index 2*period-1.
    """
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    out: list[float | None] = [None] * n
    pdi: list[float | None] = [None] * n
    mdi: list[float | None] = [None] * n
    if period < 1 or n < 2 * period or len(h) != n or len(lo) != n:
        return out, pdi, mdi
    tr: list[float] = [0.0] * n
    pdm: list[float] = [0.0] * n
    mdm: list[float] = [0.0] * n
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        dn = lo[i - 1] - lo[i]
        pdm[i] = up if up > 0 and up > dn else 0.0
        mdm[i] = dn if dn > 0 and dn > up else 0.0
    # Wilder smoothing: first value = sum of first `period` readings.
    s_tr = sum(tr[1 : period + 1])
    s_p = sum(pdm[1 : period + 1])
    s_m = sum(mdm[1 : period + 1])
    dx: list[float | None] = [None] * n
    if s_tr > 0:
        pdi[period] = 100 * s_p / s_tr
        mdi[period] = 100 * s_m / s_tr
        tot = pdi[period] + mdi[period]  # type: ignore[operator]
        dx[period] = (abs(pdi[period] - mdi[period]) / tot * 100) if tot else 0.0  # type: ignore[operator]
    for i in range(period + 1, n):
        s_tr = s_tr - s_tr / period + tr[i]
        s_p = s_p - s_p / period + pdm[i]
        s_m = s_m - s_m / period + mdm[i]
        if s_tr > 0:
            pdi[i] = 100 * s_p / s_tr
            mdi[i] = 100 * s_m / s_tr
            tot = pdi[i] + mdi[i]  # type: ignore[operator]
            dx[i] = (abs(pdi[i] - mdi[i]) / tot * 100) if tot else 0.0  # type: ignore[operator]
    valid = [(i, v) for i, v in enumerate(dx) if v is not None]
    if len(valid) >= period:
        first = valid[:period]
        avg = sum(v for _, v in first) / period
        out[first[-1][0]] = avg
        prev = avg
        for i, v in valid[period:]:
            prev = (prev * (period - 1) + v) / period
            out[i] = prev
    return out, pdi, mdi


def ichimoku(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b_period: int = 52,
) -> tuple[list[float | None], list[float | None], list[float | None], list[float | None]]:
    """Ichimoku lines (non-shifted, suitable for closed-bar signals).

    Returns (tenkan, kijun, senkou_a, senkou_b). Cloud = senkou_a vs senkou_b.
    Classic shift (26 bars forward) is intentionally omitted so signals never
    look ahead; trend = price vs cloud + TK cross.
    """
    h, lo, c = _seq(high), _seq(low), _seq(close)
    n = len(c)
    t: list[float | None] = [None] * n
    k: list[float | None] = [None] * n
    sa: list[float | None] = [None] * n
    sb: list[float | None] = [None] * n
    if tenkan < 1 or kijun < 1 or senkou_b_period < 1 or n < senkou_b_period:
        return t, k, sa, sb
    if len(h) != n or len(lo) != n:
        return t, k, sa, sb
    for i in range(n):
        if i >= tenkan - 1:
            t[i] = (max(h[i - tenkan + 1 : i + 1]) + min(lo[i - tenkan + 1 : i + 1])) / 2
        if i >= kijun - 1:
            k[i] = (max(h[i - kijun + 1 : i + 1]) + min(lo[i - kijun + 1 : i + 1])) / 2
        if t[i] is not None and k[i] is not None:
            sa[i] = (t[i] + k[i]) / 2  # type: ignore[operator]
        if i >= senkou_b_period - 1:
            sb[i] = (
                max(h[i - senkou_b_period + 1 : i + 1])
                + min(lo[i - senkou_b_period + 1 : i + 1])
            ) / 2
    return t, k, sa, sb
