# Changelog — glmbot
# Follows Keep a Changelog (https://keepachangelog.com). Versions are SemVer.

## [1.5.0] — 2026-09-24
### Added (Tier 2: trustworthy backtests)
- Intrabar stops (#18): SL on bar LOW, TP on bar HIGH, SL-first on double
  print, stop fills at min(close, stop) + extra stop slippage. Fixed a real
  look-ahead bug found along the way: the engine evaluated signals on the
  newest bars (`rows()`) instead of the expanding prefix (`head()`); warmup
  now counts active strategies only.
- Funding fees (#19): historical funding rates fetched and deducted while
  futures positions are held; reported per symbol.
- Stop slippage (#21): `risk.stop_slippage_bps` (default 0) models worse
  fills on stop exits.
- Walk-forward folds (#20): `--folds N` splits history chronologically with
  bull/bear/chop labels and a profitable-folds consistency score.
- Ablation (#17): `--ablate` runs full vs minus-one vs single sets with
  keep/drop verdicts per strategy.
- 9 new tests (84 total).

## [1.4.0] — 2026-09-24
### Added (Tier 1: survival, honest testing, risk control)
- Trailing-sync: ratcheted journal SL is pushed to the exchange stop
  (place-new-then-cancel, never unprotected); armed trigger journaled.
- Closed-candle signals: live strategy input drops the forming bar, so live
  matches the backtest (loader drops it too).
- Per-cycle reconciliation: journal-open/exchange-flat ghosts auto-close
  (10 min grace); unknown manual positions alert once/day, never auto-traded.
- Kill-switch trio: consecutive-loss halt (default 3) + peak-drawdown halt
  (default 10%), UTC-day latched, reason shown in `status`.
- Constant-risk sizing: `risk_per_trade_pct` (default 1%) sizes by SL
  distance, capped by the per-trade budget; mirrored in the backtester.
- 18 new tests (74 total) covering sync, reconcile cycle, kill switches,
  risk sizing, closed candles.

## [1.3.0] — 2026-09-23
### Added
- 4 new strategies (10 total): `vwap_trend` (VWAP cross), `stoch_rsi_cross`
  (momentum ignition), `bollinger_squeeze` (volatility breakout after pinch),
  `trend_momentum` (EMA-stack regime + RSI 50-cross). Off by default — enabling
  changes vote math, backtest first.
- Breakeven stop: once up `breakeven_trigger_pct` (default 2%), SL locks to
  entry + `breakeven_buffer_pct` (default 0.1%). Persisted like the trailing
  ratchet; mirrored in the backtester.
- Time stop: `max_hold_min` (default 0 = off) flat-exits stale positions by
  `opened_ts` age; mirrored in the backtester via bar timestamps.
- 10 new tests (56 total): signal ignition per strategy, warmup HOLDs, bad
  params, breakeven lock/persist/disabled, time-stop trip/disabled.

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
