"""Configuration loading, interpolation and validation.

Precedence (highest wins):
  1. Explicit CLI ``--config`` path, else ``GLMBOT_CONFIG`` env, else
     ``config.yml`` / ``config.yaml`` in the working directory.
  2. ``GLMBOT_API_KEY`` / ``GLMBOT_API_SECRET`` env vars override file keys.
  3. ``${VAR}`` / ``${VAR:-default}`` placeholders inside YAML values are
     expanded from the environment (useful for Docker / CI secrets).

Validation raises :class:`ConfigError` with *actionable* messages - every
message tells the operator exactly which key to fix and what valid looks
like. Public dataclass fields are backward compatible with v1.0 configs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import __version__

CONFIG_SEARCH = ["config.yml", "config.yaml"]

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised when configuration is missing, unreadable or invalid."""


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in strings."""
    if isinstance(value, str):

        def _sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else m.group(0))

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Risk config
# ---------------------------------------------------------------------------


@dataclass
class RiskCfg:
    """Per-trade and portfolio risk limits.

    New opt-in guards default to *disabled* so existing configs behave
    exactly as before:
      - ``max_daily_trades``: 0 = unlimited.
      - ``max_symbol_positions``: reserved for multi-strategy-per-symbol
        portfolios; 1 = current single-position behaviour.
      - ``slippage_bps``: backtest / paper slippage in basis points (0 = none).
      - ``max_hold_min``: 0 = disabled; else exit a stale position after N minutes.
      - ``consecutive_loss_halt``: halt new entries for the UTC day after N
        straight losing closes (0 = disabled).
      - ``max_drawdown_halt_pct``: halt new entries for the UTC day after peak
        equity drawdown reaches N% (0 = disabled).
      - ``risk_per_trade_pct``: size positions to risk N% of equity (SL-distance
        based), capped by ``per_trade_pct`` budget (0 = legacy budget sizing).
    Breakeven is ON by default (2% move locks the stop to entry + 0.1%).
    """

    quote_budget: float
    per_trade_pct: float
    max_open_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    cooldown_min: int
    atr_stops: bool = False
    atr_period: int = 14
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 3.0
    daily_loss_cap_pct: float = 0.0  # 0 = disabled; e.g. 5 = halt after -5% day
    min_votes: int = 0  # 0 = unanimous consensus
    max_daily_trades: int = 0  # 0 = unlimited
    max_symbol_positions: int = 1
    slippage_bps: float = 0.0
    breakeven_trigger_pct: float = 2.0  # 0 = disabled; else lock SL at entry+buffer
    breakeven_buffer_pct: float = 0.1
    max_hold_min: int = 0  # 0 = disabled; else exit positions older than N minutes
    consecutive_loss_halt: int = 3  # 0 = disabled; else halt day after N straight losses
    max_drawdown_halt_pct: float = 10.0  # 0 = disabled; else halt day after N% peak DD
    risk_per_trade_pct: float = 1.0  # 0 = legacy; else risk N% equity, capped by budget
    stop_slippage_bps: float = 0.0  # extra adverse slippage on stop fills (backtest)
    # --- v2 advanced guards (all opt-in, 0/False = disabled -> backward compatible) ---
    daily_profit_lock_pct: float = 0.0  # halt entries after +N% day (lock in wins)
    chandelier_enabled: bool = False  # ATR chandelier exit (highest high - mult*ATR)
    chandelier_period: int = 22
    chandelier_mult: float = 3.0
    max_atr_pct: float = 0.0  # skip entries when ATR/price*100 > cap (explosive vol)
    min_atr_pct: float = 0.0  # skip entries when ATR/price*100 < floor (dead market)
    volume_filter_mult: float = 0.0  # require last vol >= SMA20 * mult (0 = off)
    cooldown_after_loss_min: int = 0  # extra global cooldown after a losing close

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RiskCfg:
        try:
            return cls(
                quote_budget=float(d["quote_budget"]),
                per_trade_pct=float(d["per_trade_pct"]),
                max_open_positions=int(d["max_open_positions"]),
                stop_loss_pct=float(d["stop_loss_pct"]),
                take_profit_pct=float(d["take_profit_pct"]),
                trailing_stop_pct=float(d.get("trailing_stop_pct", 0.0)),
                cooldown_min=int(d.get("cooldown_min", 0)),
                atr_stops=bool(d.get("atr_stops", False)),
                atr_period=int(d.get("atr_period", 14)),
                atr_sl_mult=float(d.get("atr_sl_mult", 2.0)),
                atr_tp_mult=float(d.get("atr_tp_mult", 3.0)),
                daily_loss_cap_pct=float(d.get("daily_loss_cap_pct", 0.0)),
                min_votes=int(d.get("min_votes", 0)),
                max_daily_trades=int(d.get("max_daily_trades", 0)),
                max_symbol_positions=int(d.get("max_symbol_positions", 1)),
                slippage_bps=float(d.get("slippage_bps", 0.0)),
                breakeven_trigger_pct=float(d.get("breakeven_trigger_pct", 2.0)),
                breakeven_buffer_pct=float(d.get("breakeven_buffer_pct", 0.1)),
                max_hold_min=int(d.get("max_hold_min", 0)),
                consecutive_loss_halt=int(d.get("consecutive_loss_halt", 3)),
                max_drawdown_halt_pct=float(d.get("max_drawdown_halt_pct", 10.0)),
                risk_per_trade_pct=float(d.get("risk_per_trade_pct", 1.0)),
                stop_slippage_bps=float(d.get("stop_slippage_bps", 0.0)),
                daily_profit_lock_pct=float(d.get("daily_profit_lock_pct", 0.0)),
                chandelier_enabled=bool(d.get("chandelier_enabled", False)),
                chandelier_period=int(d.get("chandelier_period", 22)),
                chandelier_mult=float(d.get("chandelier_mult", 3.0)),
                max_atr_pct=float(d.get("max_atr_pct", 0.0)),
                min_atr_pct=float(d.get("min_atr_pct", 0.0)),
                volume_filter_mult=float(d.get("volume_filter_mult", 0.0)),
                cooldown_after_loss_min=int(d.get("cooldown_after_loss_min", 0)),
            )
        except KeyError as e:
            raise ConfigError(f"risk section missing required key: {e}") from e
        except (TypeError, ValueError) as e:
            raise ConfigError(f"risk section has invalid value: {e}") from e

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.quote_budget > 0:
            errs.append("risk.quote_budget must be > 0")
        if not 0 < self.per_trade_pct <= 100:
            errs.append("risk.per_trade_pct must be in (0, 100]")
        if self.max_open_positions < 1:
            errs.append("risk.max_open_positions must be >= 1")
        if self.stop_loss_pct <= 0:
            errs.append("risk.stop_loss_pct must be > 0 (e.g. 2.0)")
        if self.take_profit_pct <= 0:
            errs.append("risk.take_profit_pct must be > 0 (e.g. 4.0)")
        if self.trailing_stop_pct < 0:
            errs.append("risk.trailing_stop_pct must be >= 0 (0 disables)")
        if self.cooldown_min < 0:
            errs.append("risk.cooldown_min must be >= 0")
        if self.atr_stops and self.atr_period < 2:
            errs.append("risk.atr_period must be >= 2 when atr_stops is true")
        if self.daily_loss_cap_pct < 0:
            errs.append("risk.daily_loss_cap_pct must be >= 0 (0 disables)")
        if self.min_votes < 0:
            errs.append("risk.min_votes must be >= 0 (0 = unanimous)")
        if self.max_daily_trades < 0:
            errs.append("risk.max_daily_trades must be >= 0 (0 = unlimited)")
        if self.slippage_bps < 0:
            errs.append("risk.slippage_bps must be >= 0")
        if self.breakeven_trigger_pct < 0:
            errs.append("risk.breakeven_trigger_pct must be >= 0 (0 disables)")
        if self.breakeven_buffer_pct < 0:
            errs.append("risk.breakeven_buffer_pct must be >= 0")
        if self.max_hold_min < 0:
            errs.append("risk.max_hold_min must be >= 0 (0 disables)")
        if self.consecutive_loss_halt < 0:
            errs.append("risk.consecutive_loss_halt must be >= 0 (0 disables)")
        if self.max_drawdown_halt_pct < 0:
            errs.append("risk.max_drawdown_halt_pct must be >= 0 (0 disables)")
        if not 0 <= self.risk_per_trade_pct <= 100:
            errs.append("risk.risk_per_trade_pct must be in [0, 100] (0 disables)")
        if self.stop_slippage_bps < 0:
            errs.append("risk.stop_slippage_bps must be >= 0")
        if self.daily_profit_lock_pct < 0:
            errs.append("risk.daily_profit_lock_pct must be >= 0 (0 disables)")
        if self.chandelier_period < 2:
            errs.append("risk.chandelier_period must be >= 2")
        if self.chandelier_mult <= 0:
            errs.append("risk.chandelier_mult must be > 0")
        if self.max_atr_pct < 0:
            errs.append("risk.max_atr_pct must be >= 0 (0 disables)")
        if self.min_atr_pct < 0:
            errs.append("risk.min_atr_pct must be >= 0 (0 disables)")
        if self.max_atr_pct > 0 and self.min_atr_pct > 0 and self.min_atr_pct > self.max_atr_pct:
            errs.append("risk.min_atr_pct must be <= risk.max_atr_pct")
        if self.volume_filter_mult < 0:
            errs.append("risk.volume_filter_mult must be >= 0 (0 disables)")
        if self.cooldown_after_loss_min < 0:
            errs.append("risk.cooldown_after_loss_min must be >= 0 (0 disables)")
        return errs


# ---------------------------------------------------------------------------
# Bot config
# ---------------------------------------------------------------------------


@dataclass
class BotConfig:
    """Top-level bot configuration (one YAML document)."""

    api_key: str
    api_secret: str
    testnet: bool
    mode: str  # paper | live
    market: str  # spot | futures
    leverage: int
    quote_asset: str
    update_interval_sec: int
    strategies: list[str]
    strategy_params: dict[str, dict[str, Any]]
    risk: RiskCfg
    symbols: list[str]
    sqlite_path: str
    klines_cache_dir: str
    telegram: dict[str, Any] = field(default_factory=dict)
    webhook: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None
    # Exchange-native STOP_MARKET + TAKE_PROFIT_MARKET on every live futures
    # entry (safety net that fires even while the bot is down). Default on.
    exchange_stops: bool = True
    # Regime filter (opt-in, default off): skip longs in bear/chop/volatile
    # regimes as classified by glmbot.regime.detect_regime on closed candles.
    regime_filter: bool = False
    regime_allow_chop: bool = False

    @property
    def base_url(self) -> str:
        if self.market == "futures":
            return (
                "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
            )
        if self.testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"

    @property
    def env_label(self) -> str:
        net = "TESTNET" if self.testnet else "MAINNET"
        return f"{self.mode.upper()} | {self.market.upper()} | {net}"

    def validate(self) -> None:
        errs: list[str] = []
        if self.mode not in ("paper", "live"):
            errs.append("trading.mode must be 'paper' or 'live'")
        if self.mode == "live" and (not self.api_key or self.api_key.startswith("YOUR_")):
            errs.append(
                "api.key not set (required for live mode) - paste a key or switch trading.mode to 'paper'"
            )
        if self.market not in ("spot", "futures"):
            errs.append("trading.trade_type must be 'spot' or 'futures'")
        if self.market == "futures" and not 1 <= self.leverage <= 20:
            errs.append("trading.leverage must be 1..20 (start at 1-2)")
        if self.market == "spot" and self.leverage != 1:
            errs.append("trading.leverage must be 1 for spot (leverage is futures-only)")
        if not self.symbols:
            errs.append("watchlist.symbols is empty - add at least one symbol, e.g. BTCUSDT")
        for s in self.symbols:
            if not s.endswith(self.quote_asset):
                errs.append(
                    f"watchlist symbol {s!r} does not end with quote_asset {self.quote_asset!r} - "
                    "all symbols must share one quote asset"
                )
        if not self.strategies:
            errs.append("strategies.active is empty - enable at least one strategy")
        if self.update_interval_sec < 5:
            errs.append("trading.update_interval_sec must be >= 5 (avoid rate limits)")
        errs.extend(self.risk.validate())
        if self.risk.min_votes > len(self.strategies):
            errs.append(
                f"risk.min_votes={self.risk.min_votes} exceeds active strategies "
                f"({len(self.strategies)}) - entries would never trigger"
            )
        if errs:
            raise ConfigError("; ".join(errs))

    def __repr__(self) -> str:  # never leak secrets in logs / tracebacks
        key = (self.api_key[:4] + "****") if self.api_key else "(none)"
        return (
            f"BotConfig(v{__version__} {self.env_label} symbols={self.symbols} "
            f"strategies={self.strategies} key={key} config={self.config_path})"
        )

    def summary(self) -> dict[str, Any]:
        """Safe (secret-free) dict for `doctor` / `--json` output."""
        return {
            "version": __version__,
            "mode": self.mode,
            "market": self.market,
            "testnet": self.testnet,
            "leverage": self.leverage,
            "quote_asset": self.quote_asset,
            "symbols": self.symbols,
            "strategies": self.strategies,
            "update_interval_sec": self.update_interval_sec,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key and not self.api_key.startswith("YOUR_")),
            "exchange_stops": self.exchange_stops,
            "regime_filter": self.regime_filter,
            "regime_allow_chop": self.regime_allow_chop,
            "risk": {
                "quote_budget": self.risk.quote_budget,
                "per_trade_pct": self.risk.per_trade_pct,
                "max_open_positions": self.risk.max_open_positions,
                "stop_loss_pct": self.risk.stop_loss_pct,
                "take_profit_pct": self.risk.take_profit_pct,
                "trailing_stop_pct": self.risk.trailing_stop_pct,
                "min_votes": self.risk.min_votes,
                "daily_loss_cap_pct": self.risk.daily_loss_cap_pct,
                "atr_stops": self.risk.atr_stops,
                "breakeven_trigger_pct": self.risk.breakeven_trigger_pct,
                "breakeven_buffer_pct": self.risk.breakeven_buffer_pct,
                "max_hold_min": self.risk.max_hold_min,
                "consecutive_loss_halt": self.risk.consecutive_loss_halt,
                "max_drawdown_halt_pct": self.risk.max_drawdown_halt_pct,
                "risk_per_trade_pct": self.risk.risk_per_trade_pct,
                "daily_profit_lock_pct": self.risk.daily_profit_lock_pct,
                "chandelier_enabled": self.risk.chandelier_enabled,
                "chandelier_period": self.risk.chandelier_period,
                "chandelier_mult": self.risk.chandelier_mult,
                "max_atr_pct": self.risk.max_atr_pct,
                "min_atr_pct": self.risk.min_atr_pct,
                "volume_filter_mult": self.risk.volume_filter_mult,
                "cooldown_after_loss_min": self.risk.cooldown_after_loss_min,
            },
            "config_path": self.config_path,
        }


def resolve_config_path(path: str | None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("GLMBOT_CONFIG")
    if env:
        return Path(env)
    for cand in CONFIG_SEARCH:
        if Path(cand).exists():
            return Path(cand)
    return Path(CONFIG_SEARCH[0])


def load_config(path: str | None = None) -> BotConfig:
    p = resolve_config_path(path)
    if not p.exists():
        raise ConfigError(
            f"{p} not found. Run `bot.py init` (copies config.example.yml), "
            "or set GLMBOT_CONFIG to your config path."
        )
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{p} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level")
    raw = _expand_env(raw)
    try:
        api = raw["api"]
        trading = raw["trading"]
        strategies = raw["strategies"]
        risk = raw["risk"]
        watchlist = raw["watchlist"]
        storage = raw.get("storage", {})
    except KeyError as e:
        raise ConfigError(f"config missing section: {e} (see config.example.yml)") from e

    # Env overrides for secrets (CI / Docker friendly).
    api_key = os.environ.get("GLMBOT_API_KEY", api.get("key", ""))
    api_secret = os.environ.get("GLMBOT_API_SECRET", api.get("secret", ""))

    try:
        cfg = BotConfig(
            api_key=api_key or "",
            api_secret=api_secret or "",
            testnet=bool(api.get("testnet", True)),
            mode=str(trading.get("mode", "paper")),
            market=str(trading.get("trade_type", "spot")),
            leverage=int(trading.get("leverage", 1)),
            quote_asset=str(trading.get("quote_asset", "USDT")),
            update_interval_sec=int(trading.get("update_interval_sec", 60)),
            strategies=list(strategies.get("active", ["ema_cross"])),
            strategy_params={
                k: v for k, v in strategies.items() if k != "active" and isinstance(v, dict)
            },
            risk=RiskCfg.from_dict(risk),
            symbols=[str(s).upper() for s in watchlist.get("symbols", [])],
            sqlite_path=str(storage.get("sqlite_path", "data/glm.db")),
            klines_cache_dir=str(storage.get("klines_cache_dir", "data/cache")),
            telegram=raw.get("notifier", {}).get("telegram", {}) or {},
            webhook=raw.get("notifier", {}).get("webhook", {}) or {},
            config_path=str(p),
            exchange_stops=bool(trading.get("exchange_stops", True)),
            regime_filter=bool(trading.get("regime_filter", False)),
            regime_allow_chop=bool(trading.get("regime_allow_chop", False)),
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"config has invalid value types: {e}") from e
    cfg.validate()
    return cfg
