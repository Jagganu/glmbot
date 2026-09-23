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
trend_momentum. Add new ones by subclassing :class:`Strategy` and
registering in :data:`REGISTRY`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .indicators import bollinger, donchian, ema, macd, rsi, stoch_rsi, supertrend, vwap
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
