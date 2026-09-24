#!/usr/bin/env python3
"""glmbot CLI - professional Binance spot + USD-M futures trading bot.

Usage:
  py bot.py <command> [options]

Core workflow:
  init -> test-connection -> backtest -> run -> status

Commands:
  init                  create config.yml from template
  validate              check config without touching the network
  doctor                environment + connectivity diagnostics
  version               show version + dependency health
  strategies            list available strategies + parameters
  price SYM [SYM...]    current prices (public, no key needed)
  candles SYM           recent candles table
  test-connection       ping Binance + verify API keys
  backtest [-d DAYS]    walk-forward backtest on real klines
  run                   start trading loop (paper or live)
  status                portfolio dashboard (positions + equity)
  positions             open positions
  trades [-N]           recent trades
  equity                equity snapshots
  signals [-N]          recent strategy signals
  export-trades FILE    export journal to CSV
  set-mode MODE         paper|live (edits config.yml)

Global flags: -c PATH (config), -v (debug logs), --json (machine output),
  --no-color (plain output), --log-file PATH (file logging).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

from glmbot import __version__
from glmbot.ui import HAS_RICH, Table, banner, status_line
from glmbot.ui import console as _get_console

console = _get_console()


def _print_json(data: object, **_ignored: object) -> None:
    """Machine output via builtin print: Rich would word-wrap and corrupt JSON."""
    print(json.dumps(data, indent=2, default=str))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def setup_logging(verbose: bool, log_file: str | None = None) -> None:
    from glmbot.logging_setup import setup_logging as _setup

    _setup(verbose=verbose, log_file=log_file)


def _emit(data: dict, as_json: bool) -> None:
    if as_json:
        _print_json(data)
    # callers print human tables when not json


def _load(args):
    from glmbot.config import ConfigError, load_config

    try:
        return load_config(args.config)
    except ConfigError as e:
        if getattr(args, "json", False):
            _print_json({"ok": False, "error": str(e)})
        else:
            console.print(f"[red]config error:[/red] {e}")
            console.print("[dim]hint: run `bot.py validate` or `bot.py init --force`[/dim]")
        sys.exit(2)


def _client(cfg):
    from glmbot.api import BinanceClient

    return BinanceClient(cfg.api_key, cfg.api_secret, testnet=cfg.testnet, market=cfg.market)


def _need_rich_note() -> None:
    if not HAS_RICH:
        console.print("[dim]tip: `pip install rich` for colored tables[/dim]")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_init(args) -> int:
    dest = Path(args.config or "config.yml")
    if dest.exists() and not args.force:
        console.print(f"[yellow]{dest} already exists (use --force to overwrite)[/yellow]")
        return 1
    shutil.copy("config.example.yml", dest)
    try:
        import os

        os.chmod(dest, 0o600)
    except Exception:
        pass
    console.print(
        f"[green]OK created {dest}[/green] - edit api.key/api.secret, then run `bot.py doctor`"
    )
    return 0


def cmd_validate(args) -> int:
    from glmbot.config import ConfigError, load_config
    from glmbot.strategies import build_strategies

    path = args.config or "config.yml"
    try:
        cfg = load_config(path)
    except ConfigError as e:
        msg = f"invalid: {e}"
        if args.json:
            _print_json({"ok": False, "error": str(e)})
        else:
            console.print(f"[red]FAIL {path}: {msg}[/red]")
        return 1
    try:
        strats = build_strategies(cfg.strategies, cfg.strategy_params)
    except ValueError as e:
        if args.json:
            _print_json({"ok": False, "error": str(e)})
        else:
            console.print(f"[red]FAIL strategies: {e}[/red]")
        return 1
    data = {"ok": True, "config": cfg.summary(), "strategies": [s.name for s in strats]}
    if args.json:
        _print_json(data)
    else:
        console.print(f"[green]OK {path} valid[/green] - {cfg.env_label}")
        console.print(f"  symbols: {', '.join(cfg.symbols)}")
        console.print(f"  strategies: {', '.join(s.name for s in strats)}")
        console.print(
            f"  risk: {cfg.risk.per_trade_pct}%/trade | "
            f"SL {cfg.risk.stop_loss_pct}% | TP {cfg.risk.take_profit_pct}% | "
            f"max {cfg.risk.max_open_positions} positions"
        )
    return 0


def cmd_version(args) -> int:
    import importlib.metadata
    import importlib.util

    deps = {}
    for mod, label in (("requests", "requests"), ("yaml", "PyYAML"), ("rich", "rich")):
        spec = importlib.util.find_spec(mod)
        if spec is None:
            deps[label] = "missing"
            continue
        try:
            deps[label] = importlib.metadata.version(label)
        except Exception:
            try:
                deps[label] = getattr(__import__(mod), "__version__", "installed")
            except Exception:
                deps[label] = "installed"
    data = {
        "glmbot": __version__,
        "python": sys.version.split()[0],
        "deps": deps,
        "rich_ui": HAS_RICH,
    }
    if args.json:
        _print_json(data)
    else:
        console.print(banner(__version__))
        console.print(f"Python {data['python']}")
        for k, v in deps.items():
            console.print(status_line(v not in ("", "missing"), k, str(v)))
    return 0


def cmd_strategies(args) -> int:
    from glmbot.strategies import STRATEGY_CATALOG

    if args.json:
        _print_json(
            [
                {"name": m.name, "description": m.description, "params": m.params}
                for m in STRATEGY_CATALOG
            ]
        )
        return 0
    table = Table(title="Available strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Signal logic")
    table.add_column("Parameters", style="dim")
    for m in STRATEGY_CATALOG:
        table.add_row(m.name, m.description, ", ".join(f"{k}={v}" for k, v in m.params.items()))
    console.print(table)
    console.print(
        "[dim]enable in config.yml -> strategies.active; tune voting via risk.min_votes[/dim]"
    )
    return 0


def cmd_doctor(args) -> int:
    import importlib.util

    checks: list[tuple[bool, str, str]] = []
    ok = True

    # python + deps
    checks.append((sys.version_info >= (3, 10), "Python >= 3.10", sys.version.split()[0]))
    for mod, label in (("requests", "requests"), ("yaml", "PyYAML")):
        found = importlib.util.find_spec(mod) is not None
        checks.append(
            (
                found,
                f"dependency: {label}",
                "installed" if found else "MISSING - pip install -r requirements.txt",
            )
        )
    checks.append(
        (
            HAS_RICH,
            "rich UI (optional)",
            "installed" if HAS_RICH else "not installed - ASCII fallback",
        )
    )

    # config
    from glmbot.config import ConfigError, load_config

    cfg = None
    try:
        cfg = load_config(args.config)
        checks.append((True, "config", f"{cfg.config_path} ({cfg.env_label})"))
    except ConfigError as e:
        checks.append((False, "config", str(e)))

    # network + keys
    if cfg is not None:
        client = _client(cfg)
        ping = client.ping()
        checks.append(
            (
                ping,
                f"Binance REST ({client.base})",
                "reachable" if ping else "UNREACHABLE - check network/VPN",
            )
        )
        try:
            off = client.sync_time()
            checks.append((abs(off) < 5000, "clock sync", f"offset {off}ms"))
        except Exception as e:
            checks.append((False, "clock sync", str(e)[:100]))
        if cfg.api_key and not cfg.api_key.startswith("YOUR_"):
            try:
                if cfg.market == "futures":
                    bal = client.futures_balance()
                    checks.append(
                        (
                            True,
                            "signed request (futures)",
                            f"wallet {bal['total']:.2f} USDT (free {bal['free']:.2f})",
                        )
                    )
                else:
                    acct = client.account()
                    checks.append(
                        (True, "signed request (spot)", f"{len(acct.get('balances', []))} balances")
                    )
            except Exception as e:
                checks.append((False, "signed request", str(e)[:160]))
        else:
            checks.append((True, "API key", "not set - public endpoints only"))
        # filters
        try:
            missing = []
            for s in cfg.symbols:
                try:
                    client.get_filter(s)
                except Exception:
                    missing.append(s)
            checks.append(
                (
                    not missing,
                    "watchlist filters",
                    "all OK" if not missing else f"missing: {', '.join(missing)}",
                )
            )
        except Exception as e:
            checks.append((False, "watchlist filters", str(e)[:120]))

    for passed, label, detail in checks:
        ok = ok and passed
        if not args.json:
            console.print(status_line(passed, label, detail))
    if args.json:
        _print_json(
            {
                "ok": ok,
                "checks": [{"ok": p, "label": lbl, "detail": d} for p, lbl, d in checks],
            }
        )
    else:
        console.print(
            "[green]OK healthy[/green]" if ok else "[red]FAIL issues found - see above[/red]"
        )
    return 0 if ok else 1


def cmd_price(args) -> int:
    from glmbot.api import BinanceError

    cfg = _load(args)
    client = _client(cfg)
    syms = []
    for s in args.symbols:
        s = s.upper()
        syms.append(s if s.endswith(cfg.quote_asset) else s + cfg.quote_asset)
    try:
        prices = client.ticker_prices(syms) if hasattr(client, "ticker_prices") else None
    except BinanceError:
        prices = None
    rows = []
    for sym in syms:
        try:
            p = prices[sym] if prices and sym in prices else client.ticker_price(sym)
            rows.append((sym, f"{p:,.6g}", True))
        except BinanceError as e:
            rows.append((sym, str(e)[:80], False))
    if args.json:
        _print_json(
            {"prices": [{"symbol": s, "ok": ok_, "price_or_error": v} for s, v, ok_ in rows]}
        )
        return 0
    table = Table(
        title=f"Prices | {client.data_base if client.using_mainnet_data else client.base}"
    )
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", justify="right")
    for s, v, _ in rows:
        table.add_row(s, v)
    console.print(table)
    return 0


def cmd_candles(args) -> int:
    cfg = _load(args)
    client = _client(cfg)
    sym = args.symbol.upper()
    if not sym.endswith(cfg.quote_asset):
        sym += cfg.quote_asset
    try:
        kl = client.klines(sym, args.interval, args.limit)
    except Exception as e:
        console.print(f"[red]FAIL klines failed:[/red] {e}")
        return 1
    if args.json:
        _print_json({"symbol": sym, "interval": args.interval, "candles": kl[-args.rows :]})
        return 0
    table = Table(title=f"{sym} {args.interval} (last {args.rows})")
    for col in ("time", "open", "high", "low", "close", "volume"):
        table.add_column(col, justify="right")
    for k in kl[-args.rows :]:
        ts = time.strftime("%m-%d %H:%M", time.localtime(k["open_time"] / 1000))
        table.add_row(
            ts,
            f"{k['open']:,.6g}",
            f"{k['high']:,.6g}",
            f"{k['low']:,.6g}",
            f"{k['close']:,.6g}",
            f"{k['volume']:,.4g}",
        )
    console.print(table)
    return 0


def cmd_test_connection(args) -> int:
    cfg = _load(args)
    client = _client(cfg)
    ok = client.ping()
    result: dict = {"ping": ok, "base": client.base}
    if args.json:
        pass  # filled below
    else:
        console.print(status_line(ok, "Binance REST reachable", client.base))
    if not ok:
        if args.json:
            _print_json({"ok": False, **result})
        else:
            console.print("[dim]hint: VPN/firewall/geo-block? try `bot.py doctor`[/dim]")
        return 1
    try:
        offset = client.sync_time()
    except Exception as e:
        offset = 0
        console.print(f"[yellow]time sync failed:[/yellow] {e}") if not args.json else None
    result["time_offset_ms"] = offset
    if not args.json:
        console.print(f"server time offset: {offset} ms")
    if cfg.api_key and not cfg.api_key.startswith("YOUR_"):
        try:
            if cfg.market == "futures":
                bal = client.futures_balance()
                result.update(
                    {"signed": True, "wallet_total": bal["total"], "wallet_free": bal["free"]}
                )
                if not args.json:
                    console.print(
                        f"[green]OK[/] signed request (FUTURES {'testnet' if cfg.testnet else 'MAINNET'}) "
                        f"wallet: {bal['total']:.2f} USDT (free {bal['free']:.2f})"
                    )
            else:
                acct = client.account()
                result.update({"signed": True, "balances": len(acct.get("balances", []))})
                if not args.json:
                    console.print(
                        f"[green]OK[/] signed request ({'testnet' if cfg.testnet else 'MAINNET'}) "
                        f"account has {len(acct.get('balances', []))} balances"
                    )
        except Exception as e:
            result["signed"] = False
            result["error"] = str(e)[:300]
            if args.json:
                _print_json({"ok": False, **result})
            else:
                console.print(f"[red]FAIL signed request:[/red] {e}")
                console.print(
                    "[dim]hint: wrong env (spot vs futures testnet), IP whitelist, or expired demo keys[/dim]"
                )
            return 1
    else:
        result["signed"] = None
        if not args.json:
            console.print("[yellow]no API key set - public endpoints only[/yellow]")
    if args.json:
        _print_json({"ok": True, **result})
    return 0


def cmd_backtest(args) -> int:
    cfg = _load(args)
    client = _client(cfg)

    from glmbot.backtest import Backtester
    from glmbot.klines import Klines
    from glmbot.report import export_backtest_csv, show_backtest

    if not args.json:
        console.print(f"Loading {args.days}d of 15m candles for {len(cfg.symbols)} symbols...")
    klines = {}
    target = args.days * 24 * 4
    for sym in cfg.symbols:
        all_rows: list = []
        end_time = None
        first_page = True
        while len(all_rows) < target:
            kl = client.klines(
                sym, "15m", limit=min(1000, target - len(all_rows)), end_time=end_time
            )
            if not kl:
                break
            end_time = kl[0]["open_time"] - 1
            if first_page:
                # Newest page ends at the still-forming candle - drop it so the
                # backtest runs on closed bars only (#11: no look-ahead).
                kl = kl[:-1]
                first_page = False
            before = len(all_rows)
            all_rows = kl + all_rows
            if len(all_rows) == before:
                break  # no progress (degenerate page) - avoid infinite loop
            if not args.json:
                console.print(f"  {sym}: {len(all_rows)}/{target} candles", end="\r")
            if len(kl) < min(1000, target):
                break
        k = Klines(all_rows).drop_duplicates_by_time()
        klines[sym] = k
        if not args.json:
            console.print(f"  OK {sym}: {len(k)} candles" + " " * 20)
    bt = Backtester(cfg, klines)
    results = bt.run()
    if args.json:
        _print_json({s: r.to_dict() for s, r in results.items()})
    else:
        show_backtest(results, cfg.risk.quote_budget)
    if args.csv:
        path = export_backtest_csv(results, args.csv)
        bt.export_trades_csv(results, str(Path(args.csv).with_suffix("")) + ".trades.csv")
        if not args.json:
            console.print(f"[green]OK wrote {path}[/green]")
    return 0


def cmd_run(args) -> int:
    cfg = _load(args)
    from glmbot.broker import FuturesBroker, LiveBroker, PaperBroker
    from glmbot.notifier import notify_factory
    from glmbot.storage import Store
    from glmbot.trader import Trader

    if not args.json:
        console.print(banner(__version__))
    store = Store(cfg.sqlite_path)
    client = _client(cfg)
    with contextlib.suppress(Exception):
        client.sync_time()
    notifier = notify_factory(cfg)
    if cfg.mode == "paper":
        cash = store.get_paper_state(cfg.risk.quote_budget)
        broker = PaperBroker(client, cash, leverage=cfg.leverage if cfg.market == "futures" else 1)
        if not args.json:
            console.print(
                f"[blue]* PAPER[/] {cfg.env_label} | starting cash: {cash:,.2f} {cfg.quote_asset}"
            )
    else:
        broker = (
            FuturesBroker(client, leverage=cfg.leverage)
            if cfg.market == "futures"
            else LiveBroker(client)
        )
        if not args.json:
            console.print(
                f"[red bold]* LIVE {cfg.market.upper()} - real orders will be placed[/red bold]"
            )
        if not args.yes:
            confirm = console.input("type LIVE to continue: ")
            if confirm.strip() != "LIVE":
                console.print("aborted.")
                return 1
    trader = Trader(cfg, store, client, broker, notifier)
    try:
        warnings = trader.preflight()
    except RuntimeError as e:
        console.print(f"[red]FAIL preflight failed:[/red] {e}")
        return 1
    for w in warnings:
        console.print(f"[yellow]![/] {w}")
    try:
        trader.run_forever()
    except KeyboardInterrupt:
        console.print("\n[bold]stopped by user[/bold]")
        notifier.send("glmbot stopped")
    return 0


def cmd_status(args) -> int:
    cfg = _load(args)
    from glmbot.report import show_status, status_dict
    from glmbot.storage import Store

    store = Store(cfg.sqlite_path)
    client = _client(cfg)
    positions = store.open_positions(cfg.mode)
    if args.json:
        _print_json(status_dict(cfg, store, client, positions))
        return 0
    show_status(cfg, store, client, positions)
    hist = store.equity_history(cfg.mode, limit=10)
    if hist:
        console.print(f"last equity: {hist[-1]['total']:,.2f} {cfg.quote_asset}")
    return 0


def cmd_positions(args) -> int:
    cfg = _load(args)
    from glmbot.report import show_positions
    from glmbot.storage import Store

    store = Store(cfg.sqlite_path)
    rows = store.open_positions(cfg.mode)
    if args.json:
        _print_json(rows)
        return 0
    show_positions(rows)
    return 0


def cmd_trades(args) -> int:
    cfg = _load(args)
    from glmbot.report import show_trades, trades_summary
    from glmbot.storage import Store

    store = Store(cfg.sqlite_path)
    rows = store.trades(mode=cfg.mode, limit=args.limit)
    if args.json:
        _print_json({"summary": trades_summary(rows), "trades": rows})
        return 0
    show_trades(rows)
    return 0


def cmd_equity(args) -> int:
    cfg = _load(args)
    from glmbot.report import show_equity
    from glmbot.storage import Store

    store = Store(cfg.sqlite_path)
    hist = store.equity_history(cfg.mode, limit=500)
    if args.json:
        _print_json(hist[-args.limit :] if args.limit else hist)
        return 0
    show_equity(hist)
    return 0


def cmd_signals(args) -> int:
    cfg = _load(args)
    from glmbot.storage import Store

    store = Store(cfg.sqlite_path)
    rows = store.recent_signals(args.limit)
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        console.print("[dim]no signals logged yet - run the bot once to generate signals[/dim]")
        return 0
    table = Table(title="Recent signals")
    for col in ("ts", "symbol", "strategy", "side", "price", "reason"):
        table.add_column(col)
    for r in rows:
        side = r["side"]
        color = "green" if side == "BUY" else "red" if side == "SELL" else "dim"
        table.add_row(
            (r["ts"] or "")[:19],
            r["symbol"],
            r["strategy"],
            f"[{color}]{side}[/{color}]",
            f"{r['price']:,.6g}",
            (r["reason"] or "")[:50],
        )
    console.print(table)
    return 0


def cmd_export_trades(args) -> int:
    cfg = _load(args)
    from glmbot.storage import Store

    store = Store(cfg.sqlite_path)
    n = store.export_trades_csv(args.file, mode=cfg.mode if not args.all_modes else None)
    console.print(f"[green]OK exported {n} trades -> {args.file}[/green]")
    return 0


def cmd_set_mode(args) -> int:
    import re

    p = Path(args.config or "config.yml")
    if not p.exists():
        console.print(f"[red]{p} not found[/red]")
        return 1
    text = p.read_text(encoding="utf-8")
    new, n = re.subn(r"(\s*mode:\s*)(\w+)", rf"\g<1>{args.mode}", text, count=1)
    if n == 0:
        console.print("[red]could not find trading.mode in config[/red]")
        return 1
    p.write_text(new, encoding="utf-8")
    console.print(
        f"[green]OK mode set to {args.mode}[/green]"
        + (" - [red bold]LIVE trading enabled, be careful[/]" if args.mode == "live" else "")
    )
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glmbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", default=None, help="config file path (or GLMBOT_CONFIG)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("--no-color", action="store_true", help="disable colored output")
    parser.add_argument("--log-file", default=None, help="also log to this file")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="create config.yml from template")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser("validate", help="validate config (no network)")
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("doctor", help="diagnose environment + connectivity")
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("version", help="show version + deps")
    sp.set_defaults(fn=cmd_version)

    sp = sub.add_parser("strategies", help="list strategies + parameters")
    sp.set_defaults(fn=cmd_strategies)

    sp = sub.add_parser("price", help="show current prices")
    sp.add_argument("symbols", nargs="+")
    sp.set_defaults(fn=cmd_price)

    sp = sub.add_parser("candles", help="show recent candles")
    sp.add_argument("symbol")
    sp.add_argument("-n", "--limit", type=int, default=100)
    sp.add_argument("-r", "--rows", type=int, default=15)
    sp.add_argument("-i", "--interval", default="15m")
    sp.set_defaults(fn=cmd_candles)

    sp = sub.add_parser("test-connection", help="ping Binance + verify keys")
    sp.set_defaults(fn=cmd_test_connection)

    sp = sub.add_parser("backtest", help="backtest strategies on watchlist")
    sp.add_argument("-d", "--days", type=int, default=7)
    sp.add_argument("--csv", default=None, help="write summary CSV to PATH (+ .trades.csv)")
    sp.set_defaults(fn=cmd_backtest)

    sp = sub.add_parser("run", help="start trading loop")
    sp.add_argument("-y", "--yes", action="store_true", help="skip live confirmation")
    sp.set_defaults(fn=cmd_run)

    for name, fn, help_ in (
        ("status", cmd_status, "portfolio dashboard"),
        ("positions", cmd_positions, "list open positions"),
        ("equity", cmd_equity, "equity snapshots"),
    ):
        sp = sub.add_parser(name, help=help_)
        if name == "equity":
            sp.add_argument("-N", "--limit", type=int, default=20)
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("signals", help="recent strategy signals")
    sp.add_argument("-N", "--limit", type=int, default=20)
    sp.set_defaults(fn=cmd_signals)

    sp = sub.add_parser("trades", help="recent trades")
    sp.add_argument("-N", "--limit", type=int, default=20, dest="limit")
    sp.set_defaults(fn=cmd_trades)

    sp = sub.add_parser("export-trades", help="export journal to CSV")
    sp.add_argument("file", help="output CSV path")
    sp.add_argument("--all-modes", action="store_true", help="include paper+live")
    sp.set_defaults(fn=cmd_export_trades)

    sp = sub.add_parser("set-mode", help="set paper|live in config")
    sp.add_argument("mode", choices=["paper", "live"])
    sp.set_defaults(fn=cmd_set_mode)

    return parser


def _hoist_global_flags(argv: list[str]) -> list[str]:
    """Allow global flags anywhere: `bot.py trades -N 5 --json` works.

    Moves -c/--config, -v/--verbose, --json, --no-color, --log-file ahead of
    the subcommand. Command-specific flags (-N, -d, --csv, ...) stay put.
    """
    BOOL_FLAGS = {"--json", "--no-color", "-v", "--verbose"}
    VAL_FLAGS = {"-c", "--config", "--log-file"}
    hoisted: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in BOOL_FLAGS:
            hoisted.append(tok)
        elif tok in VAL_FLAGS:
            hoisted.append(tok)
            if i + 1 < len(argv):
                i += 1
                hoisted.append(argv[i])
        else:
            rest.append(tok)
        i += 1
    return hoisted + rest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_hoist_global_flags(sys.argv[1:] if argv is None else argv))
    if args.no_color:
        from glmbot.ui import set_no_color

        set_no_color()
    setup_logging(args.verbose, log_file=args.log_file)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        console.print("\n[bold]interrupted[/bold]")
        return 130
    except SystemExit:
        raise
    except Exception as e:
        import logging

        logging.getLogger("glmbot").exception("command failed: %s", e)
        if args.json:
            _print_json({"ok": False, "error": str(e)[:500]})
        else:
            console.print(f"[red]FAIL failed:[/red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
