# glmbot — Professional CLI Trading Bot for Binance

Terminal-native spot + USDⓈ-M futures bot. Runs on Windows, Linux, macOS and
**Termux on Android**. Pure-Python core (`requests` + `PyYAML`), optional `rich`
for color.

```
┌────────────────────────────────────────────────────────────────┐
│  bot.py validate         →  config check (no network)           │
│  bot.py doctor            →  env + connectivity diagnostics      │
│  bot.py backtest -d 7     →  strategy validation on real klines  │
│  bot.py run               →  trading loop (paper / live)         │
│  bot.py status            →  portfolio dashboard                 │
└────────────────────────────────────────────────────────────────┘
```

## Highlights

- **Spot + futures** — `spot` via api.binance.com, `futures` via fapi
  (leverage 1–20x, one-way mode, reduceOnly closes). Futures *testnet* has a
  broken price feed, so the bot reads decisions from **real mainnet data**
  while orders execute on testnet (hybrid mode, automatic).
- **10 strategies, vote consensus** — `ema_cross`, `rsi_reversion`, `macd`,
  `bollinger`, `supertrend`, `donchian_breakout`, `vwap_trend`,
  `stoch_rsi_cross`, `bollinger_squeeze`, `trend_momentum`. `risk.min_votes: 0` =
  unanimous, `N` = at-least-N-of-M with SELL veto. See `bot.py strategies`.
- **Risk-first engine** — % budget sizing, max positions, per-symbol cooldown,
  fixed-% or ATR stops, breakeven lock, ratcheting trailing stop, time stop,
  daily-loss kill switch, opt-in daily trade budget. Exits checked *before*
  entries every cycle.
- **Faithful backtester** — same strategy + risk code as live, 1-bar execution
  delay, fees + configurable slippage, ATR/min-votes support, Sharpe/Sortino/
  profit-factor/expectancy/exposure/buy-and-hold metrics, CSV export.
- **SQLite journal** — trades, positions, equity curve, signals, kill-switch
  state. Survives restarts (positions + cooldowns restored). CSV export.
- **Alerts** — Telegram (HTML-escaped, throttled) + generic webhook
  (`{event, text, ts}` envelope) on fills, errors, kill-switch, startup.
- **Hardened API client** — HMAC GET+POST, clock sync, jittered backoff,
  429/418 handling, HTTP-451 geo guidance, exchange filters
  (LOT_SIZE/MARKET_LOT_SIZE/PRICE_FILTER/MIN_NOTIONAL), async-fill polling,
  on-disk exchangeInfo cache, batch prices.
- **Ops-ready** — `doctor` diagnostics, startup preflight, heartbeat file,
  structured rotating logs (`data/logs/`), graceful SIGTERM shutdown,
  Dockerfile, CI, `Makefile`.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# pip install -r requirements.txt               # Linux / macOS / Termux-full
# pip install requests PyYAML                   # minimal (ASCII UI)

python bot.py init            # create config.yml (chmod 600)
# edit config.yml → keys, trade_type, leverage, watchlist, risk
python bot.py validate
python bot.py doctor
python bot.py price btc eth
python bot.py backtest -d 7
python bot.py run
```

Or `bash setup.sh` (universal) / `bash setup-termux.sh` (Android), or
`make install && make doctor`.

## CLI reference

| Command | Description |
|---|---|
| `init [--force]` | copy `config.example.yml` → `config.yml` |
| `validate` | strict config + strategy check (offline) |
| `doctor` | deps, config, ping, clock, keys, wallet, filters |
| `version` | version + dependency health |
| `strategies` | strategy catalog with parameters |
| `price SYM…` | live prices (auto `USDT` suffix) |
| `candles SYM [-n N] [-r ROWS] [-i 15m]` | candle table |
| `test-connection` | ping + signed-request + wallet |
| `backtest [-d DAYS] [--csv FILE]` | walk-forward backtest (+ CSV export) |
| `run [-y]` | trading loop (preflight + heartbeat) |
| `status` | dashboard: equity, day PnL, kill switch, positions |
| `positions` / `trades [-N]` / `equity` / `signals [-N]` | journal views |
| `export-trades FILE [--all-modes]` | journal → CSV |
| `set-mode paper\|live` | switch mode in config.yml |

Global flags: `-c PATH` config (or `GLMBOT_CONFIG`), `-v` debug,
`--json` machine output (most commands), `--no-color`, `--log-file PATH`.

## API keys

| Environment | Get keys | Config |
|---|---|---|
| Spot testnet | testnet.binance.vision (GitHub login) | `testnet: true` + `trade_type: spot` |
| Futures testnet | testnet.binancefuture.com → API Key tab → System generated (HMAC) | `testnet: true` + `trade_type: futures` |
| Mainnet | binance.com → API Management | `testnet: false` |

Keys can also come from env: `GLMBOT_API_KEY` / `GLMBOT_API_SECRET`
(Docker/CI friendly), or `${VAR}` interpolation inside `config.yml`.

**Security**: `config.yml` is gitignored + `chmod 600`. Restrict keys
(spot: *Enable Reading + Spot Trading*; futures: *Futures*; **no
withdrawals**; IP whitelist). Futures demo keys reset periodically —
regenerate if auth fails. `bot.py doctor` tells you exactly which
environment your key works in.

## How it trades

Each cycle (`trading.update_interval_sec`, default 60 s):

1. **Data** — latest 15m close per symbol (batch `ticker/prices` with
   per-symbol fallback; stale-quote warning beyond ~17 min). **Signals use
   closed candles only** (forming bar dropped) so live matches the backtest;
   exits use the latest price.
2. **Exits first** — breakeven lock → hard SL → TP → trailing ratchet →
   time stop → unanimous-SELL signal exit. Every open position, every cycle.
   Ratchets also sync up to the exchange stop (place-new-then-cancel).
3. **Entries** — strategies vote; `min_votes` threshold + SELL veto →
   gates (max positions, kill-switch trio, daily budget, cooldown) →
   size = `risk_per_trade_pct` of equity by SL distance, capped by the
   `per_trade_pct` budget (≥ 10 USDT), leverage-scaled notional on futures,
   exchange-filter-clamped → market fill → exchange stops armed → journal.
4. **Bookkeeping** — per-cycle journal-vs-exchange reconciliation (flat
   ghosts closed, unknown manual positions alerted once/day), equity
   snapshot, kill-switch evaluation, heartbeat, cycle timing log.

```
      ┌─────────┐   15m closes   ┌────────────┐  BUY/SELL/HOLD  ┌───────────┐
      │ Binance │ ─────────────► │ Strategies │ ──────────────► │ Consensus │
      │  REST   │                │  (6 built- │   min_votes +   │  (entry?  │
      │ spot +  │ ◄───────────── │   ins)     │   SELL veto     │   exit?)  │
      │ futures │  market orders └────────────┘                 └─────┬─────┘
      └─────────┘                                                    │
           ▲                                                         ▼
      fills│                                                  ┌────────────┐
           │                                                  │    Risk    │
           │                                                  │ SL/TP/     │
           │                                                  │ trail/kill │
           │                                                  └─────┬──────┘
           │                                                        │
      ┌────┴────────────────────────────────────────────────────────┘
      ▼
 ┌─────────┐   journal + alerts
 │ Brokers │ ─► SQLite (trades/positions/equity/signals) → Telegram/webhook
 │paper/   │
 │live/fut │
 └─────────┘
```

## Risk controls (`config.yml` → `risk:`)

```yaml
risk:
  per_trade_pct: 2.0        # % of budget as margin/notional per position
  max_open_positions: 3
  stop_loss_pct: 2.0        # price move (futures: × leverage on margin!)
  take_profit_pct: 4.0
  trailing_stop_pct: 1.5    # 0 = disabled
  breakeven_trigger_pct: 2.0 # lock SL to entry+buffer once up 2% (0 = off)
  breakeven_buffer_pct: 0.1
  max_hold_min: 0          # time-stop: flat-exit stale positions (0 = off)
  consecutive_loss_halt: 3 # halt day after 3 straight losing closes (0 = off)
  max_drawdown_halt_pct: 10.0 # halt day after -10% vs peak equity (0 = off)
  risk_per_trade_pct: 1.0  # risk 1% equity/trade by SL distance, capped by budget
  cooldown_min: 30
  atr_stops: false          # true = SL/TP from ATR (adapts to volatility)
  atr_sl_mult: 2.0
  atr_tp_mult: 3.0
  daily_loss_cap_pct: 5.0   # kill switch: halt entries after -5% day (0 = off)
  min_votes: 0              # 0 = unanimous; N = at-least-N-of-M
  max_daily_trades: 0       # 0 = unlimited
  slippage_bps: 0.0         # backtest slippage (e.g. 5 = 0.05%)
```

```yaml
trading:
  exchange_stops: true    # live futures: attach STOP_MARKET + TAKE_PROFIT_MARKET
                          # (closePosition) on every entry - they fire on Binance
                          # even while the bot is down; bot-side exits act first
```

**Leverage warning**: at 10x, a 2% adverse move ≈ 20% of margin. Start 1–2x,
prove the system on paper/testnet, size small live.

## Configuration reference

- `api.key/secret` — or `GLMBOT_API_KEY`/`GLMBOT_API_SECRET` env.
- `trading.quote_asset` — all symbols must end with it (e.g. `USDT`).
- `trading.mode` — `paper` (simulated, persists `paper_cash` in DB) | `live`.
- `trading.trade_type` — `spot` | `futures`; `leverage` must be 1 for spot.
- `strategies.active` + per-strategy blocks — unknown names fail `validate`.
- `watchlist.symbols` — e.g. `["BTCUSDT","ETHUSDT"]`.
- `storage.sqlite_path` — default `data/glm.db` (single file, portable).
- `notifier.telegram/webhook` — `enabled` + credentials/URL.

Full documented template: `config.example.yml`. Machine-readable current
config: `bot.py validate --json`. Secrets are never printed (masked repr).

## Latency model (read before asking for "faster stops")

- **Stop/TP reaction is exchange-side, single-digit ms.** Every live futures
  entry attaches `STOP_MARKET` + `TAKE_PROFIT_MARKET` (`closePosition`,
  `MARK_PRICE` trigger). Binance's matching engine evaluates the trigger and
  fills at market with zero network hops — no bot loop involved. Verify with
  `status` (`(EX)` marker) or the `openAlgoOrders` endpoint.
- **Bot-side reaction cannot beat physics.** Measured RTT from a home
  connection is ~160–350 ms per request; detect-then-order needs two trips,
  so ~350–700 ms is the floor. Sub-200 ms bot-loop reaction requires
  colocation next to the matching engine — no code change fixes distance.
- **What the 60 s loop is for:** entries, signal exits, trailing/breakeven
  ratchets, reconciliation. Stops do not need the loop; the loop needs the
  stops (as backstop). Keep `exchange_stops: true`.

## Backtesting

```bash
python bot.py backtest -d 30 --csv backtest.csv
```

- Real 15m klines paginated from the data market (same source as live).
- Signals evaluated per closed bar; fills at the **next** bar's close.
- Metrics: trades, win rate, PnL%, MaxDD, profit factor, expectancy,
  avg win/loss, Sharpe/Sortino (15m-annualized), exposure%, buy-and-hold
  delta, fees. Portfolio panel rolls up independent per-symbol runs.
- `--csv` writes summary + `.trades.csv` (per-fill audit trail).

Limitations (be honest with yourself): no funding payments, no liquidation
modeling, stops assumed to fill, no order-book depth. Backtests ≠ future
results — they validate *logic*, not profitability.

## Operations runbook

- **Logs**: console (Rich) + `data/logs/glmbot.log` (rotating, secrets
  redacted). `-v` for DEBUG, `--log-file` to override.
- **Heartbeat**: `data/glm.db` journal + `data/glmbot.heartbeat` (UTC ISO
  timestamp, updated every cycle — point your supervisor at it).
- **Supervision**: `tmux` / `systemd` / Docker restart-policy. The loop
  sleeps in 1 s slices so SIGTERM stops promptly; paper cash + positions
  persist across restarts.
- **Backup**: `sqlite3 data/glm.db ".backup data/glm.$(date +%F).db"` or
  `Store.backup()`; `export-trades` for CSV.
- **Docker**: `docker build -t glmbot . && docker run --env-file .env
  -v ./data:/app/data glmbot run`.

## Termux (Android)

```bash
pkg install python git
git clone <repo> glmbot && cd glmbot
bash setup-termux.sh            # light (~2 MB); --full for rich colors
nano config.yml
python bot.py doctor
python bot.py run
```

Keep alive: `termux-wake-lock`, `tmux new -s bot`, disable battery
optimization for Termux. `data/glm.db` is portable PC ↔ phone.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `451 restricted location` | geo-blocked IP | testnet, allowed region, or compliant proxy |
| `-2015 invalid API key` | wrong env / IP whitelist / expired demo key | `doctor`; match spot↔futures testnet; regenerate demo keys |
| `-1021 timestamp` | clock skew | `test-connection` (auto-syncs); NTP |
| `order below minimums` | tiny size / dust | raise `per_trade_pct` / `quote_budget` (≥ 10 USDT) |
| `no price returned` | delisted/typo symbol | fix `watchlist.symbols`; check `get_filter` via `doctor` |
| Kill switch active | `-daily_loss_cap_pct` hit | intentional halt; resumes next UTC day |
| No signals ever | `min_votes` > agreeing strategies | lower `min_votes` or reduce `strategies.active` |
| Futures testnet weird PnL | fake testnet price feed | automatic: decisions use mainnet data (by design) |

## Development

```bash
pip install -r requirements-dev.txt
python tests.py          # 74 unit tests, no network
make lint | make typecheck
```

Architecture: `api.py` (REST) → `klines.py`/`indicators.py` →
`strategies.py` → `trader.py` (+ `risk.py`) → `broker.py` →
`storage.py`/`notifier.py` → `report.py`/`ui.py`. `bot.py` is CLI only.
`backtest.py` reuses strategy + risk semantics with its own fill model.
CI runs lint + tests on 3.10–3.12.

## Disclaimer

Educational software, **no warranty**. Crypto is risky; leverage multiplies
losses. Always paper/testnet first, start small live. You are responsible
for your configuration, keys and funds. See `LICENSE` (MIT) and
`CHANGELOG.md`.
