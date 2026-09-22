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
donchian_breakout. Add new ones by subclassing :class:`Strategy` and
registering in :data:`REGISTRY`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .indicators import bollinger, donchian, ema, macd, rsi, supertrend
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


REGISTRY: dict[str, type] = {
    EMACross.name: EMACross,
    RSIReversion.name: RSIReversion,
    MACDStrategy.name: MACDStrategy,
    BollingerStrategy.name: BollingerStrategy,
    SupertrendStrategy.name: SupertrendStrategy,
    DonchianBreakout.name: DonchianBreakout,
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
