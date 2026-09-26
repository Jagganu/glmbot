"""Strategy framework: signal ∈ {HOLD, BUY, SELL}.

A strategy is a pure function of closed candles → :class:`Signal`. The trader
combines votes (see :meth:`Trader._consensus`): BUY entries need ``min_votes``
BUY votes with zero SELL dissent; exits need unanimous SELL (or risk stops).

Conventions
  - ``k`` = :class:`Klines` container; last row = newest *closed* candle.
  - Never look ahead: only use rows ``<= -1`` (we evaluate per closed bar).
  - ``HOLD`` with a reason is always safe (warmup / no-cross / inside-bands).
  - Each strategy exposes ``describe()`` metadata for the ``strategies`` CLI.

Built-ins: ema_cross, rsi_reversion, macd, bollinger, supertrend,
donchian_breakout, vwap_trend, stoch_rsi_cross, bollinger_squeeze,
trend_momentum, adx_trend, ichimoku_trend, keltner_breakout, obv_trend,
mfi_reversion, tema_trend, stoch_cross, cci_reversion. Add new ones by
subclassing :class:`Strategy` and registering in :data:`REGISTRY`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .indicators import (
    adx,
    bollinger,
    cci,
    donchian,
    ema,
    ichimoku,
    keltner,
    macd,
    mfi,
    obv,
    rsi,
    stoch_osc,
    stoch_rsi,
    supertrend,
    tema,
    vwap,
)
from .klines import Klines

log = logging.getLogger("glmbot.strategy")

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


@dataclass
class Signal:
    side: str  # BUY | SELL | HOLD
    reason: str
    symbol: str
    price: float
    strategy: str

    def __post_init__(self) -> None:
        if self.side not in (BUY, SELL, HOLD):
            raise ValueError(f"invalid signal side: {self.side!r}")


@dataclass(frozen=True)
class StrategyMeta:
    name: str
    description: str
    params: dict[str, str]  # param -> "default (meaning)"


class Strategy(ABC):
    name = "base"
    meta: StrategyMeta = StrategyMeta("base", "base class", {})

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})
        self.validate_params()

    def validate_params(self) -> None:
        """Override to reject bad config early (raises ValueError)."""
        return None

    @abstractmethod
    def evaluate(self, symbol: str, k: Klines) -> Signal:
        """Evaluate on closed candles; last row = newest closed candle."""

    def describe(self) -> StrategyMeta:
        return self.meta


class EMACross(Strategy):
    """Golden/death cross of fast vs slow EMA (trend following)."""

    name = "ema_cross"
    meta = StrategyMeta(
        "ema_cross",
        "BUY on fast-EMA cross above slow-EMA; SELL on cross below.",
        {"fast": "9 (fast EMA period)", "slow": "21 (slow EMA period)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.fast = int(self.params.get("fast", 9))
        self.slow = int(self.params.get("slow", 21))

    def validate_params(self) -> None:
        fast = int(self.params.get("fast", 9))
        slow = int(self.params.get("slow", 21))
        if fast < 2 or slow < 3:
            raise ValueError("ema_cross: periods must be >= 2/3")
        if fast >= slow:
            raise ValueError("ema_cross: fast must be < slow")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        closes = k.close
        price = float(closes[-1]) if closes else 0.0
        if len(closes) < self.slow + 2:
            return Signal(
                HOLD,
                f"insufficient history ({len(closes)}/{self.slow + 2})",
                symbol,
                price,
                self.name,
            )
        f = ema(closes, self.fast)
        s = ema(closes, self.slow)
        if f[-2] is None or s[-2] is None or f[-1] is None or s[-1] is None:
            return Signal(HOLD, "ema warmup", symbol, price, self.name)
        prev_diff = f[-2] - s[-2]  # type: ignore[operator]
        curr_diff = f[-1] - s[-1]  # type: ignore[operator]
        if prev_diff <= 0 < curr_diff:
            return Signal(
                BUY, f"EMA{self.fast} crossed above EMA{self.slow}", symbol, price, self.name
            )
        if prev_diff >= 0 > curr_diff:
            return Signal(
                SELL, f"EMA{self.fast} crossed below EMA{self.slow}", symbol, price, self.name
            )
        return Signal(HOLD, "no cross", symbol, price, self.name)


class RSIReversion(Strategy):
    """Mean reversion: BUY when RSI recovers above oversold; SELL below overbought."""

    name = "rsi_reversion"
    meta = StrategyMeta(
        "rsi_reversion",
        "BUY on RSI recovery above oversold; SELL on drop below overbought.",
        {"period": "14", "oversold": "30", "overbought": "70"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 14))
        self.os = float(self.params.get("oversold", 30))
        self.ob = float(self.params.get("overbought", 70))

    def validate_params(self) -> None:
        p = int(self.params.get("period", 14))
        if p < 2:
            raise ValueError("rsi_reversion: period must be >= 2")
        os_ = float(self.params.get("oversold", 30))
        ob = float(self.params.get("overbought", 70))
        if not 0 < os_ < ob < 100:
            raise ValueError("rsi_reversion: need 0 < oversold < overbought < 100")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        closes = k.close
        price = float(closes[-1]) if closes else 0.0
        if len(closes) < self.period + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        r = rsi(closes, self.period)
        prev, cur = r[-2], r[-1]
        if prev is None or cur is None:
            return Signal(HOLD, "rsi warmup", symbol, price, self.name)
        if prev <= self.os < cur:
            return Signal(
                BUY, f"RSI recovered above {self.os:g} ({cur:.1f})", symbol, price, self.name
            )
        if prev >= self.ob > cur:
            return Signal(
                SELL, f"RSI dropped below {self.ob:g} ({cur:.1f})", symbol, price, self.name
            )
        return Signal(HOLD, f"RSI {cur:.1f}", symbol, price, self.name)


class MACDStrategy(Strategy):
    """MACD line vs signal cross (momentum)."""

    name = "macd"
    meta = StrategyMeta(
        "macd",
        "BUY on MACD cross above signal; SELL on cross below.",
        {"fast": "12", "slow": "26", "signal": "9"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.fast = int(self.params.get("fast", 12))
        self.slow = int(self.params.get("slow", 26))
        self.signal = int(self.params.get("signal", 9))

    def validate_params(self) -> None:
        fast, slow = int(self.params.get("fast", 12)), int(self.params.get("slow", 26))
        if fast >= slow:
            raise ValueError("macd: fast must be < slow")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        closes = k.close
        price = float(closes[-1]) if closes else 0.0
        if len(closes) < self.slow + self.signal + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        line, sig, _ = macd(closes, self.fast, self.slow, self.signal)
        if line[-2] is None or sig[-2] is None or line[-1] is None or sig[-1] is None:
            return Signal(HOLD, "macd warmup", symbol, price, self.name)
        prev = line[-2] - sig[-2]  # type: ignore[operator]
        curr = line[-1] - sig[-1]  # type: ignore[operator]
        if prev <= 0 < curr:
            return Signal(BUY, "MACD crossed above signal", symbol, price, self.name)
        if prev >= 0 > curr:
            return Signal(SELL, "MACD crossed below signal", symbol, price, self.name)
        return Signal(HOLD, "no cross", symbol, price, self.name)


class BollingerStrategy(Strategy):
    """Bollinger mean reversion: BUY at/below lower band, SELL at/above upper."""

    name = "bollinger"
    meta = StrategyMeta(
        "bollinger",
        "BUY at lower band touch; SELL at upper band touch.",
        {"period": "20", "std_dev": "2.0 (band width)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 20))
        self.sd = float(self.params.get("std_dev", 2.0))

    def validate_params(self) -> None:
        if int(self.params.get("period", 20)) < 2:
            raise ValueError("bollinger: period must be >= 2")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        closes = k.close
        price = float(closes[-1]) if closes else 0.0
        if len(closes) < self.period + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        upper, _mid, lower = bollinger(closes, self.period, self.sd)
        if upper[-1] is None or lower[-1] is None:
            return Signal(HOLD, "bollinger warmup", symbol, price, self.name)
        u, lo = float(upper[-1]), float(lower[-1])
        if price <= lo:
            return Signal(BUY, f"price at lower band ({lo:.6g})", symbol, price, self.name)
        if price >= u:
            return Signal(SELL, f"price at upper band ({u:.6g})", symbol, price, self.name)
        return Signal(HOLD, "inside bands", symbol, price, self.name)


class SupertrendStrategy(Strategy):
    """Supertrend direction flip (ATR-based trend filter)."""

    name = "supertrend"
    meta = StrategyMeta(
        "supertrend",
        "BUY when supertrend flips up; SELL when it flips down.",
        {"period": "10 (ATR period)", "multiplier": "3.0 (band distance)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 10))
        self.multiplier = float(self.params.get("multiplier", 3.0))

    def validate_params(self) -> None:
        if int(self.params.get("period", 10)) < 2:
            raise ValueError("supertrend: period must be >= 2")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.period + 3:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        _line, direction = supertrend(k.high, k.low, k.close, self.period, self.multiplier)
        prev, cur = direction[-2], direction[-1]
        if prev is None or cur is None:
            return Signal(HOLD, "supertrend warmup", symbol, price, self.name)
        if prev == "down" and cur == "up":
            return Signal(BUY, "supertrend flipped up", symbol, price, self.name)
        if prev == "up" and cur == "down":
            return Signal(SELL, "supertrend flipped down", symbol, price, self.name)
        return Signal(HOLD, f"supertrend {cur}", symbol, price, self.name)


class DonchianBreakout(Strategy):
    """Donchian breakout: BUY on close above highest high; SELL below lowest low."""

    name = "donchian_breakout"
    meta = StrategyMeta(
        "donchian_breakout",
        "BUY on N-bar high breakout; SELL on N-bar low breakdown.",
        {"period": "20 (channel lookback)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 20))

    def validate_params(self) -> None:
        if int(self.params.get("period", 20)) < 2:
            raise ValueError("donchian_breakout: period must be >= 2")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.period + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        upper, lower = donchian(k.high[:-1], k.low[:-1], self.period)
        if upper[-1] is None or lower[-1] is None:
            return Signal(HOLD, "donchian warmup", symbol, price, self.name)
        if price > float(upper[-1]):
            return Signal(BUY, f"breakout above {upper[-1]:.6g}", symbol, price, self.name)
        if price < float(lower[-1]):
            return Signal(SELL, f"breakdown below {lower[-1]:.6g}", symbol, price, self.name)
        return Signal(HOLD, "inside channel", symbol, price, self.name)


class VWAPTrend(Strategy):
    """Institutional trend: BUY on close cross above rolling VWAP, SELL below."""

    name = "vwap_trend"
    meta = StrategyMeta(
        "vwap_trend",
        "BUY on close cross above VWAP; SELL on cross below.",
        {"period": "20 (VWAP lookback)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 20))

    def validate_params(self) -> None:
        if int(self.params.get("period", 20)) < 2:
            raise ValueError("vwap_trend: period must be >= 2")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.period + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        v = vwap(k.high, k.low, k.close, k.volume, self.period)
        if v[-2] is None or v[-1] is None:
            return Signal(HOLD, "vwap warmup", symbol, price, self.name)
        prev_above = k.close[-2] > v[-2]
        curr_above = price > v[-1]
        if not prev_above and curr_above:
            return Signal(BUY, "close crossed above VWAP", symbol, price, self.name)
        if prev_above and not curr_above:
            return Signal(SELL, "close crossed below VWAP", symbol, price, self.name)
        state = "above" if curr_above else "below"
        return Signal(HOLD, f"holding {state} VWAP", symbol, price, self.name)


class StochRSICross(Strategy):
    """Momentum ignition: BUY on StochRSI cross up through oversold,
    SELL on cross down through overbought."""

    name = "stoch_rsi_cross"
    meta = StrategyMeta(
        "stoch_rsi_cross",
        "BUY on StochRSI cross above oversold; SELL on cross below overbought.",
        {
            "rsi_period": "14",
            "stoch_period": "14",
            "oversold": "0.2 (0..1)",
            "overbought": "0.8 (0..1)",
        },
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.stoch_period = int(self.params.get("stoch_period", 14))
        self.os = float(self.params.get("oversold", 0.2))
        self.ob = float(self.params.get("overbought", 0.8))

    def validate_params(self) -> None:
        if int(self.params.get("rsi_period", 14)) < 2:
            raise ValueError("stoch_rsi_cross: rsi_period must be >= 2")
        if int(self.params.get("stoch_period", 14)) < 2:
            raise ValueError("stoch_rsi_cross: stoch_period must be >= 2")
        os_ = float(self.params.get("oversold", 0.2))
        ob = float(self.params.get("overbought", 0.8))
        if not 0 <= os_ < ob <= 1:
            raise ValueError("stoch_rsi_cross: need 0 <= oversold < overbought <= 1")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        need = self.rsi_period + self.stoch_period + 1
        if len(k) < need:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        s = stoch_rsi(k.close, self.rsi_period, self.stoch_period)
        prev, cur = s[-2], s[-1]
        if prev is None or cur is None:
            return Signal(HOLD, "stoch-rsi warmup", symbol, price, self.name)
        if prev <= self.os < cur:
            return Signal(BUY, f"StochRSI crossed above {self.os:g}", symbol, price, self.name)
        if prev >= self.ob > cur:
            return Signal(SELL, f"StochRSI crossed below {self.ob:g}", symbol, price, self.name)
        return Signal(HOLD, f"StochRSI {cur:.2f}", symbol, price, self.name)


class BollingerSqueeze(Strategy):
    """Volatility breakout: bands pinched to a lookback low, then price
    breaks out - BUY above upper, SELL below lower."""

    name = "bollinger_squeeze"
    meta = StrategyMeta(
        "bollinger_squeeze",
        "BUY on upper-band breakout from squeeze; SELL on lower breakdown.",
        {
            "period": "20 (band period)",
            "std_dev": "2.0 (band width)",
            "lookback": "50 (squeeze ranking window)",
        },
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 20))
        self.sd = float(self.params.get("std_dev", 2.0))
        self.lookback = int(self.params.get("lookback", 50))

    def validate_params(self) -> None:
        if int(self.params.get("period", 20)) < 2:
            raise ValueError("bollinger_squeeze: period must be >= 2")
        if int(self.params.get("lookback", 50)) < 2:
            raise ValueError("bollinger_squeeze: lookback must be >= 2")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        closes = k.close
        price = float(closes[-1]) if closes else 0.0
        need = self.period + self.lookback
        if len(closes) < need:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        upper, mid, lower = bollinger(closes, self.period, self.sd)
        if upper[-1] is None or mid[-1] is None or lower[-1] is None:
            return Signal(HOLD, "band warmup", symbol, price, self.name)
        widths = [
            (u - lo) / m
            for u, m, lo in zip(
                upper[-self.lookback :], mid[-self.lookback :], lower[-self.lookback :], strict=True
            )
            if u is not None and m is not None and lo is not None and m != 0
        ]
        if len(widths) < self.lookback:
            return Signal(HOLD, "band warmup", symbol, price, self.name)
        # Squeeze is judged on the CLOSED prior bar (the breakout bar itself
        # always expands the band, so it can never be the minimum).
        was_squeezed = widths[-2] <= min(widths[:-1])
        u, lo = float(upper[-1]), float(lower[-1])
        if was_squeezed and price > u:
            return Signal(BUY, f"squeeze breakout above {u:.6g}", symbol, price, self.name)
        if was_squeezed and price < lo:
            return Signal(SELL, f"squeeze breakdown below {lo:.6g}", symbol, price, self.name)
        state = "squeezed, awaiting break" if was_squeezed else "bands expanded"
        return Signal(HOLD, state, symbol, price, self.name)


class TrendMomentum(Strategy):
    """Regime + ignition: EMA stack defines trend, RSI 50-cross fires.
    BUY in uptrend on RSI cross above 50; SELL in downtrend below 50."""

    name = "trend_momentum"
    meta = StrategyMeta(
        "trend_momentum",
        "BUY: uptrend + RSI cross above 50; SELL: downtrend + RSI cross below 50.",
        {"fast": "20 (trend EMA)", "slow": "50 (regime EMA)", "rsi_period": "14"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.fast = int(self.params.get("fast", 20))
        self.slow = int(self.params.get("slow", 50))
        self.rsi_period = int(self.params.get("rsi_period", 14))

    def validate_params(self) -> None:
        fast = int(self.params.get("fast", 20))
        slow = int(self.params.get("slow", 50))
        if fast < 2 or slow < 3:
            raise ValueError("trend_momentum: periods must be >= 2/3")
        if fast >= slow:
            raise ValueError("trend_momentum: fast must be < slow")
        if int(self.params.get("rsi_period", 14)) < 2:
            raise ValueError("trend_momentum: rsi_period must be >= 2")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        closes = k.close
        price = float(closes[-1]) if closes else 0.0
        if len(closes) < self.slow + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        f = ema(closes, self.fast)
        s = ema(closes, self.slow)
        r = rsi(closes, self.rsi_period)
        if f[-1] is None or s[-1] is None or r[-2] is None or r[-1] is None:
            return Signal(HOLD, "indicator warmup", symbol, price, self.name)
        uptrend = f[-1] > s[-1]
        crossed_up = r[-2] <= 50 < r[-1]
        crossed_down = r[-2] >= 50 > r[-1]
        if uptrend and crossed_up:
            return Signal(BUY, "uptrend + RSI ignition above 50", symbol, price, self.name)
        if not uptrend and crossed_down:
            return Signal(SELL, "downtrend + RSI breakdown below 50", symbol, price, self.name)
        regime = "uptrend" if uptrend else "downtrend"
        return Signal(HOLD, f"{regime}, RSI {r[-1]:.1f}", symbol, price, self.name)


class ADXTrend(Strategy):
    """Trend strength + direction: +DI cross above -DI with ADX confirmation."""

    name = "adx_trend"
    meta = StrategyMeta(
        "adx_trend",
        "BUY on +DI cross above -DI with ADX above threshold; SELL on opposite cross.",
        {"period": "14 (ADX/DI period)", "adx_min": "20.0 (trend strength floor)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 14))
        self.adx_min = float(self.params.get("adx_min", 20.0))

    def validate_params(self) -> None:
        if int(self.params.get("period", 14)) < 2:
            raise ValueError("adx_trend: period must be >= 2")
        if not 0 <= float(self.params.get("adx_min", 20.0)) <= 100:
            raise ValueError("adx_trend: adx_min must be 0..100")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < 2 * self.period + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        a, pdi, mdi = adx(k.high, k.low, k.close, self.period)
        if (
            a[-1] is None
            or pdi[-2] is None
            or mdi[-2] is None
            or pdi[-1] is None
            or mdi[-1] is None
        ):
            return Signal(HOLD, "adx warmup", symbol, price, self.name)
        prev_diff = pdi[-2] - mdi[-2]  # type: ignore[operator]
        curr_diff = pdi[-1] - mdi[-1]  # type: ignore[operator]
        strength = float(a[-1])
        if prev_diff <= 0 < curr_diff and strength >= self.adx_min:
            return Signal(
                BUY, f"+DI crossed above -DI (ADX {strength:.1f})", symbol, price, self.name
            )
        if prev_diff >= 0 > curr_diff:
            return Signal(
                SELL, f"-DI crossed above +DI (ADX {strength:.1f})", symbol, price, self.name
            )
        return Signal(HOLD, f"ADX {strength:.1f}", symbol, price, self.name)


class IchimokuTrend(Strategy):
    """Ichimoku regime: price vs cloud + Tenkan/Kijun cross (no forward shift)."""

    name = "ichimoku_trend"
    meta = StrategyMeta(
        "ichimoku_trend",
        "BUY: TK cross up + price above cloud; SELL: TK cross down or price below cloud.",
        {"tenkan": "9", "kijun": "26", "senkou_b": "52 (cloud lookback)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.tenkan = int(self.params.get("tenkan", 9))
        self.kijun = int(self.params.get("kijun", 26))
        self.senkou_b = int(self.params.get("senkou_b", 52))

    def validate_params(self) -> None:
        t, kj = int(self.params.get("tenkan", 9)), int(self.params.get("kijun", 26))
        sb = int(self.params.get("senkou_b", 52))
        if t < 2 or kj < 2 or sb < 2:
            raise ValueError("ichimoku_trend: periods must be >= 2")
        if t >= kj:
            raise ValueError("ichimoku_trend: tenkan must be < kijun")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.senkou_b + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        t, kj, sa, sb = ichimoku(k.high, k.low, k.close, self.tenkan, self.kijun, self.senkou_b)
        if t[-2] is None or kj[-2] is None or t[-1] is None or kj[-1] is None:
            return Signal(HOLD, "ichimoku warmup", symbol, price, self.name)
        if sa[-1] is None or sb[-1] is None:
            return Signal(HOLD, "cloud warmup", symbol, price, self.name)
        cloud_top = max(float(sa[-1]), float(sb[-1]))
        cloud_bot = min(float(sa[-1]), float(sb[-1]))
        cross_up = (t[-2] - kj[-2]) <= 0 < (t[-1] - kj[-1])  # type: ignore[operator]
        cross_dn = (t[-2] - kj[-2]) >= 0 > (t[-1] - kj[-1])  # type: ignore[operator]
        if cross_up and price > cloud_top:
            return Signal(BUY, "TK cross up above cloud", symbol, price, self.name)
        if cross_dn or price < cloud_bot:
            reason = "TK cross down" if cross_dn else f"price below cloud ({cloud_bot:.6g})"
            return Signal(SELL, reason, symbol, price, self.name)
        above = price > cloud_top
        return Signal(
            HOLD, f"price {'above' if above else 'inside/below'} cloud", symbol, price, self.name
        )


class KeltnerBreakout(Strategy):
    """Volatility breakout: close cross outside Keltner channels (EMA +/- ATR)."""

    name = "keltner_breakout"
    meta = StrategyMeta(
        "keltner_breakout",
        "BUY on close cross above upper band; SELL on cross below lower band.",
        {"ema_period": "20", "atr_period": "10", "multiplier": "2.0 (channel width)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.ema_period = int(self.params.get("ema_period", 20))
        self.atr_period = int(self.params.get("atr_period", 10))
        self.mult = float(self.params.get("multiplier", 2.0))

    def validate_params(self) -> None:
        if int(self.params.get("ema_period", 20)) < 2:
            raise ValueError("keltner_breakout: ema_period must be >= 2")
        if int(self.params.get("atr_period", 10)) < 2:
            raise ValueError("keltner_breakout: atr_period must be >= 2")
        if float(self.params.get("multiplier", 2.0)) <= 0:
            raise ValueError("keltner_breakout: multiplier must be > 0")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < max(self.ema_period, self.atr_period) + 3:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        upper, _mid, lower = keltner(
            k.high, k.low, k.close, self.ema_period, self.atr_period, self.mult
        )
        if upper[-2] is None or lower[-2] is None or upper[-1] is None or lower[-1] is None:
            return Signal(HOLD, "keltner warmup", symbol, price, self.name)
        prev_c = float(k.close[-2])
        if prev_c <= float(upper[-2]) and price > float(upper[-1]):
            return Signal(BUY, f"breakout above Keltner {upper[-1]:.6g}", symbol, price, self.name)
        if prev_c >= float(lower[-2]) and price < float(lower[-1]):
            return Signal(
                SELL, f"breakdown below Keltner {lower[-1]:.6g}", symbol, price, self.name
            )
        return Signal(HOLD, "inside Keltner channel", symbol, price, self.name)


class OBVTrend(Strategy):
    """Smart-money flow: BUY on OBV fast-EMA cross above slow-EMA; SELL below."""

    name = "obv_trend"
    meta = StrategyMeta(
        "obv_trend",
        "BUY on OBV fast-EMA cross above slow-EMA; SELL on cross below.",
        {"fast": "9 (OBV fast EMA)", "slow": "21 (OBV slow EMA)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.fast = int(self.params.get("fast", 9))
        self.slow = int(self.params.get("slow", 21))

    def validate_params(self) -> None:
        fast = int(self.params.get("fast", 9))
        slow = int(self.params.get("slow", 21))
        if fast < 2 or slow < 3:
            raise ValueError("obv_trend: periods must be >= 2/3")
        if fast >= slow:
            raise ValueError("obv_trend: fast must be < slow")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.slow + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        flow = obv(k.close, k.volume)
        f = ema(flow, self.fast)
        s = ema(flow, self.slow)
        if f[-2] is None or s[-2] is None or f[-1] is None or s[-1] is None:
            return Signal(HOLD, "obv warmup", symbol, price, self.name)
        prev_diff = f[-2] - s[-2]  # type: ignore[operator]
        curr_diff = f[-1] - s[-1]  # type: ignore[operator]
        if prev_diff <= 0 < curr_diff:
            return Signal(BUY, "OBV accumulation cross up", symbol, price, self.name)
        if prev_diff >= 0 > curr_diff:
            return Signal(SELL, "OBV distribution cross down", symbol, price, self.name)
        return Signal(HOLD, "no OBV cross", symbol, price, self.name)


class MFIReversion(Strategy):
    """Volume-weighted mean reversion: MFI recovery above oversold = BUY."""

    name = "mfi_reversion"
    meta = StrategyMeta(
        "mfi_reversion",
        "BUY on MFI recovery above oversold; SELL on drop below overbought.",
        {"period": "14", "oversold": "20", "overbought": "80"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 14))
        self.os = float(self.params.get("oversold", 20))
        self.ob = float(self.params.get("overbought", 80))

    def validate_params(self) -> None:
        if int(self.params.get("period", 14)) < 2:
            raise ValueError("mfi_reversion: period must be >= 2")
        os_ = float(self.params.get("oversold", 20))
        ob = float(self.params.get("overbought", 80))
        if not 0 < os_ < ob < 100:
            raise ValueError("mfi_reversion: need 0 < oversold < overbought < 100")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.period + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        m = mfi(k.high, k.low, k.close, k.volume, self.period)
        prev, cur = m[-2], m[-1]
        if prev is None or cur is None:
            return Signal(HOLD, "mfi warmup", symbol, price, self.name)
        if prev <= self.os < cur:
            return Signal(
                BUY, f"MFI recovered above {self.os:g} ({cur:.1f})", symbol, price, self.name
            )
        if prev >= self.ob > cur:
            return Signal(
                SELL, f"MFI dropped below {self.ob:g} ({cur:.1f})", symbol, price, self.name
            )
        return Signal(HOLD, f"MFI {cur:.1f}", symbol, price, self.name)


class TEMATrend(Strategy):
    """Low-lag trend: fast TEMA cross above slow TEMA = BUY (faster than EMA)."""

    name = "tema_trend"
    meta = StrategyMeta(
        "tema_trend",
        "BUY on fast-TEMA cross above slow-TEMA; SELL on cross below.",
        {"fast": "9 (fast TEMA)", "slow": "21 (slow TEMA)"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.fast = int(self.params.get("fast", 9))
        self.slow = int(self.params.get("slow", 21))

    def validate_params(self) -> None:
        fast = int(self.params.get("fast", 9))
        slow = int(self.params.get("slow", 21))
        if fast < 2 or slow < 3:
            raise ValueError("tema_trend: periods must be >= 2/3")
        if fast >= slow:
            raise ValueError("tema_trend: fast must be < slow")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.slow * 3 + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        f = tema(k.close, self.fast)
        s = tema(k.close, self.slow)
        if f[-2] is None or s[-2] is None or f[-1] is None or s[-1] is None:
            return Signal(HOLD, "tema warmup", symbol, price, self.name)
        prev_diff = f[-2] - s[-2]  # type: ignore[operator]
        curr_diff = f[-1] - s[-1]  # type: ignore[operator]
        if prev_diff <= 0 < curr_diff:
            return Signal(
                BUY, f"TEMA{self.fast} crossed above TEMA{self.slow}", symbol, price, self.name
            )
        if prev_diff >= 0 > curr_diff:
            return Signal(
                SELL, f"TEMA{self.fast} crossed below TEMA{self.slow}", symbol, price, self.name
            )
        return Signal(HOLD, "no TEMA cross", symbol, price, self.name)


class StochCross(Strategy):
    """Classic stochastic ignition: %K cross %D inside extreme zones."""

    name = "stoch_cross"
    meta = StrategyMeta(
        "stoch_cross",
        "BUY on %K cross above %D while oversold; SELL on cross below while overbought.",
        {"k_period": "14", "d_period": "3", "oversold": "20", "overbought": "80"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.kp = int(self.params.get("k_period", 14))
        self.dp = int(self.params.get("d_period", 3))
        self.os = float(self.params.get("oversold", 20))
        self.ob = float(self.params.get("overbought", 80))

    def validate_params(self) -> None:
        if int(self.params.get("k_period", 14)) < 2:
            raise ValueError("stoch_cross: k_period must be >= 2")
        if int(self.params.get("d_period", 3)) < 1:
            raise ValueError("stoch_cross: d_period must be >= 1")
        os_ = float(self.params.get("oversold", 20))
        ob = float(self.params.get("overbought", 80))
        if not 0 <= os_ < ob <= 100:
            raise ValueError("stoch_cross: need 0 <= oversold < overbought <= 100")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.kp + self.dp + 1:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        kk, dd = stoch_osc(k.high, k.low, k.close, self.kp, self.dp)
        if kk[-2] is None or dd[-2] is None or kk[-1] is None or dd[-1] is None:
            return Signal(HOLD, "stoch warmup", symbol, price, self.name)
        prev_diff = kk[-2] - dd[-2]  # type: ignore[operator]
        curr_diff = kk[-1] - dd[-1]  # type: ignore[operator]
        if prev_diff <= 0 < curr_diff and kk[-1] <= self.os + 15:  # type: ignore[operator]
            return Signal(BUY, f"%K crossed above %D ({kk[-1]:.1f})", symbol, price, self.name)
        if prev_diff >= 0 > curr_diff and kk[-1] >= self.ob - 15:  # type: ignore[operator]
            return Signal(SELL, f"%K crossed below %D ({kk[-1]:.1f})", symbol, price, self.name)
        return Signal(HOLD, f"Stoch {kk[-1]:.1f}/{dd[-1]:.1f}", symbol, price, self.name)


class CCIReversion(Strategy):
    """CCI extremes: snap-back from +/-100 triggers mean-reversion entries."""

    name = "cci_reversion"
    meta = StrategyMeta(
        "cci_reversion",
        "BUY on CCI cross above oversold; SELL on cross below overbought.",
        {"period": "20", "oversold": "-100", "overbought": "100"},
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.period = int(self.params.get("period", 20))
        self.os = float(self.params.get("oversold", -100))
        self.ob = float(self.params.get("overbought", 100))

    def validate_params(self) -> None:
        if int(self.params.get("period", 20)) < 2:
            raise ValueError("cci_reversion: period must be >= 2")
        if not float(self.params.get("oversold", -100)) < float(self.params.get("overbought", 100)):
            raise ValueError("cci_reversion: need oversold < overbought")

    def evaluate(self, symbol: str, k: Klines) -> Signal:
        price = float(k.close[-1]) if k.close else 0.0
        if len(k) < self.period + 2:
            return Signal(HOLD, "insufficient history", symbol, price, self.name)
        c = cci(k.high, k.low, k.close, self.period)
        prev, cur = c[-2], c[-1]
        if prev is None or cur is None:
            return Signal(HOLD, "cci warmup", symbol, price, self.name)
        if prev <= self.os < cur:
            return Signal(
                BUY, f"CCI recovered above {self.os:g} ({cur:.1f})", symbol, price, self.name
            )
        if prev >= self.ob > cur:
            return Signal(
                SELL, f"CCI dropped below {self.ob:g} ({cur:.1f})", symbol, price, self.name
            )
        return Signal(HOLD, f"CCI {cur:.1f}", symbol, price, self.name)


REGISTRY: dict[str, type] = {
    EMACross.name: EMACross,
    RSIReversion.name: RSIReversion,
    MACDStrategy.name: MACDStrategy,
    BollingerStrategy.name: BollingerStrategy,
    SupertrendStrategy.name: SupertrendStrategy,
    DonchianBreakout.name: DonchianBreakout,
    VWAPTrend.name: VWAPTrend,
    StochRSICross.name: StochRSICross,
    BollingerSqueeze.name: BollingerSqueeze,
    TrendMomentum.name: TrendMomentum,
    ADXTrend.name: ADXTrend,
    IchimokuTrend.name: IchimokuTrend,
    KeltnerBreakout.name: KeltnerBreakout,
    OBVTrend.name: OBVTrend,
    MFIReversion.name: MFIReversion,
    TEMATrend.name: TEMATrend,
    StochCross.name: StochCross,
    CCIReversion.name: CCIReversion,
}

STRATEGY_CATALOG: list[StrategyMeta] = [cls.meta for cls in REGISTRY.values()]


def build_strategies(names: list[str], params: dict[str, dict]) -> list[Strategy]:
    """Instantiate strategies by name; raises ValueError on unknown/invalid."""
    out: list[Strategy] = []
    for n in names:
        cls = REGISTRY.get(n)
        if cls is None:
            raise ValueError(f"unknown strategy '{n}' (available: {sorted(REGISTRY)})")
        try:
            out.append(cls(params.get(n, {})))
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"strategy '{n}': invalid params: {e}") from e
    if not out:
        raise ValueError("no strategies enabled (strategies.active is empty)")
    return out
