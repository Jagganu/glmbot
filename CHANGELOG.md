# Changelog — glmbot
# Follows Keep a Changelog (https://keepachangelog.com). Versions are SemVer.

## [1.2.0] — 2026-09-23
### Added
- Exchange-native safety net (live futures): every entry attaches
  `STOP_MARKET` + `TAKE_PROFIT_MARKET` (`closePosition`) orders that fire on
  Binance even while the bot is down. Toggle via `trading.exchange_stops`
  (default true). Order ids journaled (auto-migrated `positions` table);
  cancelled best-effort after our own fill closes the position; pre-feature
  positions get armed on startup. `status` marks protected stops with `[EX]`.
- Exchange reconciliation: if a close fails but `positionRisk` shows flat
  (stop fired while away / manual close), the journal closes with a
  `reconciled:` trade instead of leaving a ghost position.
- 10 new offline tests (46 total) covering order payloads, tick rounding,
  best-effort cancel, journal migration and reconcile paths.

## [1.1.0] — 2026-09-22
### Added
- Professional CLI: banner, `version`, `doctor`, `validate`, `export-trades`,
  `strategies` list, `--json` machine output and `--no-color` on every command.
- Structured logging (`glmbot/logging_setup.py`): Rich console + rotating file
  handler under `data/logs/`, secret redaction, `-v/--verbose` flag.
- Configuration hardening: env-var interpolation (`${VAR}`), `GLMBOT_*`
  overrides, strict validation with actionable messages, masked `__repr__`,
  `config validate` command.
- Binance client hardening: User-Agent, jittered exponential backoff,
  HTTP 451 geo-restriction message, request timeouts, batch `ticker_prices()`,
  on-disk `exchangeInfo` cache (24 h TTL), clearer signed-request errors.
- Storage: WAL mode, indexes, `backup()`, `export_trades_csv()`,
  per-symbol trade queries, `counts()` health summary.
- Risk: risk-based sizing notes, `max_daily_trades` + `max_symbol_positions`
  guards (opt-in, default off → backward compatible), structured `ExitPlan`.
- Indicators: `stoch_rsi`, `vwap`, `supertrend`, `donchian` + input validation.
- Strategies: `supertrend` + `donchian_breakout` strategies, `describe()` /
  `STRATEGY_CATALOG` metadata, parameter validation, `strategies` CLI.
- Backtester: honors `min_votes` + SELL veto (was unanimous-only), ATR stops,
  futures leverage + funding-aware fee path, slippage modeling, full metrics
  (Sharpe, Sortino, profit factor, expectancy, avg win/loss, exposure,
  buy-and-hold delta), CSV trade export.
- Trader: startup preflight (keys, balance, filters, leverage), graceful
  SIGINT/SIGTERM shutdown, health heartbeat file, cycle timing stats,
  stale-quote detection, restart cooldown restore.
- Notifier: HTML-escaped Telegram messages, rate limiting, `format_trade()`
  helpers, webhook JSON envelope with event type.
- Reports: portfolio dashboard (`status` shows equity + open risk + kill
  switch), `show_backtest` with extended metrics, CSV export.
- DevOps: `pyproject.toml`, `Makefile`, `Dockerfile`, CI workflow,
  `.env.example`, pinned requirements, universal `setup.sh`.
- Docs: rewritten README with architecture, config reference, operations
  runbook, troubleshooting matrix, disclaimer.

### Fixed
- Backtest entry consensus now matches live trader (`min_votes` + SELL veto).
- Backtest trailing-stop ratchet no longer requires `trail_high > entry`
  before updating the high (matches `RiskManager`).
- Futures paper PnL path documented; backtest supports leverage-scaled
  notional when `market=futures`.
- `Klines.drop_duplicates_by_time` keeps newest bar on duplicates.

### Changed
- Version bumped to 1.1.0. All `tests.py` (34 tests) still pass unmodified
  in spirit — public APIs preserved.

## [1.0.0] — baseline
- Spot + futures trading loop, 4 strategies, vote consensus, risk manager
  (SL/TP/trailing/ATR/daily kill switch), SQLite journal, Telegram/webhook
  alerts, backtester, Termux support.
