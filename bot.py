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
  regime [SYM...]       market-regime diagnosis (bull/bear/chop/volatile)
  screener              ranked watchlist scan (momentum + regime + volume)
  funding [SYM...]      futures funding rates + 7d average
  test-connection       ping Binance + verify API keys
  backtest [-d DAYS]    walk-forward backtest on real klines
  optimize [-d DAYS]    grid-search strategy params (ranks by Sharpe/PnL/Calmar)
  analyze               journal analytics by symbol + strategy
  run                   start trading loop (paper or live)
  status                portfolio dashboard (positions + equity)
  positions             open positions
  trades [-N]           recent trades
  equity                equity snapshots
  signals [-N]          recent strategy signals
  export-trades FILE    export journal to CSV
  set-mode MODE         paper|live (edits config.yml)
  set-env ENV           demo|real = testnet|mainnet API (edits config.yml)

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
    if args.ablate and args.folds > 1:
        console.print("[red]use either --ablate or --folds, not both[/red]")
        return 1

    from glmbot.backtest import Backtester, fold_splits, summarize_run
    from glmbot.klines import Klines
    from glmbot.report import export_backtest_csv, show_ablation, show_backtest, show_folds

    if not args.json:
        console.print(f"Loading {args.days}d of 15m candles for {len(cfg.symbols)} symbols...")
    klines = {}
    target = args.days * 24 * 4
    for sym in cfg.symbols:
        all_rows: list = []
        end_time = None
        first_page = True
        oldest_seen: int | None = None
        while len(all_rows) < target:
            kl = client.klines(
                sym, "15m", limit=min(1000, target - len(all_rows)), end_time=end_time
            )
            if not kl:
                break
            if oldest_seen is not None and kl[0]["open_time"] >= oldest_seen:
                break  # no progress (exchange clamped) - avoid infinite loop
            oldest_seen = kl[0]["open_time"]
            end_time = oldest_seen - 1
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
            # NOTE: no short-page break - the API sometimes returns 999/1000
            # with more history behind it; empty + no-progress breaks suffice.
        k = Klines(all_rows).drop_duplicates_by_time()
        klines[sym] = k
        if not args.json:
            console.print(f"  OK {sym}: {len(k)} candles" + " " * 20)

    # ---- funding history, futures only (#19) ----
    funding: dict[str, list] = {}
    if cfg.market == "futures":
        for sym, k in klines.items():
            if not len(k):
                continue
            try:
                funding[sym] = client.funding_history(sym, k.open_time[0], k.open_time[-1])
                if not args.json:
                    console.print(f"  OK {sym}: {len(funding[sym])} funding events")
            except Exception as e:
                console.print(f"  [yellow]! {sym} funding unavailable: {e}[/yellow]")
                funding[sym] = []

    # ---- walk-forward folds (#20) ----
    if args.folds > 1:
        ref_sym = cfg.symbols[0]
        ref_folds = fold_splits(klines[ref_sym], args.folds)
        fold_rows = []
        for fi, (label, _refk) in enumerate(ref_folds):
            per_fold = {}
            for sym, k in klines.items():
                folds = fold_splits(k, args.folds)
                if fi < len(folds):
                    per_fold[sym] = folds[fi][1]
            if not per_fold:
                continue
            results = Backtester(cfg, per_fold, funding=funding).run()
            s = summarize_run(results, cfg.risk.quote_budget)
            bh = sum(r.buy_hold_pct for r in results.values()) / max(len(results), 1)
            fold_rows.append(
                {
                    "label": label,
                    "trades": s["trades"],
                    "win_rate": s["win_rate"],
                    "pnl_pct": s["pnl_pct"],
                    "worst_max_dd": s["worst_max_dd"],
                    "buy_hold_pct": bh,
                }
            )
            if not args.json:
                console.print(f"  OK {label}: {s['pnl_pct']:+.2f}%")
        if args.json:
            _print_json(fold_rows)
        else:
            show_folds(fold_rows)
        return 0

    # ---- ablation (#17) ----
    if args.ablate:
        import dataclasses

        full = cfg.strategies
        sets = [("full", "full", full)]
        sets += [(f"no_{s}", "drop", [x for x in full if x != s]) for s in full]
        sets += [(f"only_{s}", "single", [s]) for s in full]
        rows = []
        base_pnl = 0.0
        for label, kind, active in sets:
            if not active:
                continue
            sub = dataclasses.replace(cfg, strategies=list(active))
            results = Backtester(sub, klines, funding=funding).run()
            s = summarize_run(results, cfg.risk.quote_budget)
            if kind == "full":
                base_pnl = s["pnl_pct"]
            rows.append(
                {
                    "label": f"{label} ({'+'.join(active)})",
                    "kind": kind,
                    "trades": s["trades"],
                    "win_rate": s["win_rate"],
                    "pnl_pct": s["pnl_pct"],
                }
            )
            if not args.json:
                console.print(f"  OK {label}: {s['pnl_pct']:+.2f}% ({int(s['trades'])} trades)")
        if args.json:
            _print_json({"base_pnl_pct": base_pnl, "rows": rows})
        else:
            show_ablation(rows, base_pnl)
        return 0

    bt = Backtester(cfg, klines, funding=funding)
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


def _load_history(cfg, client, days: int, quiet: bool = False):
    """Shared kline + funding loader (closed bars only, deduped)."""
    from glmbot.klines import Klines

    klines = {}
    target = max(60, days * 24 * 4)
    for sym in cfg.symbols:
        all_rows: list = []
        end_time = None
        first_page = True
        oldest_seen: int | None = None
        while len(all_rows) < target:
            kl = client.klines(
                sym, "15m", limit=min(1000, target - len(all_rows)), end_time=end_time
            )
            if not kl:
                break
            if oldest_seen is not None and kl[0]["open_time"] >= oldest_seen:
                break
            oldest_seen = kl[0]["open_time"]
            end_time = oldest_seen - 1
            if first_page:
                kl = kl[:-1]
                first_page = False
            before = len(all_rows)
            all_rows = kl + all_rows
            if len(all_rows) == before:
                break
        k = Klines(all_rows).drop_duplicates_by_time()
        klines[sym] = k
        if not quiet:
            console.print(f"  OK {sym}: {len(k)} candles" + " " * 20)
    funding: dict[str, list] = {}
    if cfg.market == "futures":
        for sym, k in klines.items():
            if not len(k):
                continue
            try:
                funding[sym] = client.funding_history(sym, k.open_time[0], k.open_time[-1])
            except Exception:
                funding[sym] = []
    return klines, funding


def _parse_grid_sets(sets: list[str] | None) -> dict[str, list]:
    """Parse --set KEY=v1,v2 into a param grid (ints/floats auto-typed)."""
    grid: dict[str, list] = {}
    for item in sets or []:
        if "=" not in item or "." not in item.split("=")[0]:
            raise ValueError(f"bad --set {item!r} (want 'strategy.param=v1,v2')")
        key, raw_vals = item.split("=", 1)
        vals = []
        for v in raw_vals.split(","):
            v = v.strip()
            if not v:
                continue
            try:
                vals.append(int(v))
                continue
            except ValueError:
                pass
            try:
                vals.append(float(v))
                continue
            except ValueError:
                pass
            vals.append(v)
        if not vals:
            raise ValueError(f"bad --set {item!r} (no values)")
        grid[key.strip()] = vals
    return grid


def _default_grid(cfg) -> dict[str, list]:
    """Small sensible grid when the user passes no --set (capped, safe)."""
    active = list(cfg.strategies)
    grid: dict[str, list] = {}
    if "ema_cross" in active:
        grid["ema_cross.fast"] = [5, 9, 12]
        grid["ema_cross.slow"] = [21, 26]
    elif "rsi_reversion" in active:
        grid["rsi_reversion.period"] = [10, 14, 21]
        grid["rsi_reversion.oversold"] = [25, 30]
    elif "macd" in active:
        grid["macd.fast"] = [8, 12]
        grid["macd.slow"] = [21, 26]
    elif active:
        first = active[0]
        grid[f"{first}.__note__"] = [0]  # placeholder replaced below
        grid.pop(f"{first}.__note__")
        # Generic fallback: vary min_votes-equivalent via risk is not gridable;
        # instead vary nothing and tell the user. Empty grid errors cleanly.
    return grid


def cmd_optimize(args) -> int:
    cfg = _load(args)
    client = _client(cfg)
    from glmbot.optimize import run_optimization

    try:
        grid = _parse_grid_sets(args.set)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    if not grid:
        grid = _default_grid(cfg)
    if not grid:
        console.print(
            "[red]no grid: pass --set strategy.param=v1,v2 (e.g. --set ema_cross.fast=5,9,12)[/red]"
        )
        return 1
    if not args.json:
        console.print(f"Loading {args.days}d of 15m candles for {len(cfg.symbols)} symbols...")
    klines, funding = _load_history(cfg, client, args.days, quiet=bool(args.json))
    rows = run_optimization(
        cfg, klines, grid, funding=funding, metric=args.metric, max_combos=args.max_combos
    )
    if args.json:
        _print_json({"metric": args.metric, "grid": grid, "rows": rows})
        return 0
    table = Table(title=f"Optimize (ranked by {args.metric}, {len(rows)} combos)")
    table.add_column("Rank", justify="right")
    table.add_column("Params", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("PnL%", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD%", justify="right")
    for i, r in enumerate(rows[:20], 1):
        table.add_row(
            str(i),
            r["label"][:60],
            str(int(r["trades"])),
            f"{r['win_rate']:.0f}%",
            f"{r['pnl_pct']:+.2f}%",
            f"{r['sharpe']:.2f}",
            f"{r['worst_max_dd']:.2f}",
        )
    console.print(table)
    if rows:
        console.print(f"[green]best: {rows[0]['label']}[/] (score {rows[0]['score']:.3f})")
        console.print(
            "[dim]tip: apply the winner to config.yml, then re-run backtest --folds to confirm[/dim]"
        )
    return 0


def cmd_screener(args) -> int:
    cfg = _load(args)
    client = _client(cfg)
    from glmbot.indicators import adx as adx_fn
    from glmbot.indicators import atr as atr_fn
    from glmbot.indicators import rsi as rsi_fn
    from glmbot.klines import Klines
    from glmbot.regime import detect_regime
    from glmbot.strategies import build_strategies

    strats = build_strategies(cfg.strategies, cfg.strategy_params)
    rows = []
    syms = args.symbols or cfg.symbols
    for sym in syms:
        s = sym.upper()
        if not s.endswith(cfg.quote_asset):
            s += cfg.quote_asset
        try:
            raw = client.klines(s, "15m", limit=120)
        except Exception as e:
            rows.append({"symbol": s, "error": str(e)[:80]})
            continue
        k = Klines(raw).closed()
        if len(k) < 60:
            rows.append({"symbol": s, "error": f"only {len(k)} closed bars"})
            continue
        votes = []
        for st in strats:
            try:
                votes.append(st.evaluate(s, k).side)
            except Exception:
                votes.append("HOLD")
        buys = sum(1 for v in votes if v == "BUY")
        sells = sum(1 for v in votes if v == "SELL")
        try:
            rsi_v = rsi_fn(k.close, 14)[-1]
        except Exception:
            rsi_v = None
        try:
            adx_v = adx_fn(k.high, k.low, k.close, 14)[0][-1]
        except Exception:
            adx_v = None
        try:
            atr_v = atr_fn(k.high, k.low, k.close, 14)[-1]
            atr_pct = (atr_v / k.close[-1] * 100) if atr_v and k.close[-1] else None
        except Exception:
            atr_pct = None
        vol_ratio = None
        try:
            if len(k.volume) >= 20 and sum(k.volume[-20:]) > 0:
                sma20 = sum(k.volume[-20:]) / 20
                vol_ratio = k.volume[-1] / sma20 if sma20 else None
        except Exception:
            pass
        try:
            reg = detect_regime(k)
        except Exception:
            reg = {"regime": "unknown", "reason": ""}
        score = (buys - sells) + (
            1
            if reg.get("regime") == "bull"
            else (-2 if reg.get("regime") in ("bear", "volatile") else 0)
        )
        chg = (k.close[-1] / k.close[0] - 1) * 100 if k.close[0] else 0.0
        rows.append(
            {
                "symbol": s,
                "price": k.close[-1],
                "chg_pct": chg,
                "buys": buys,
                "sells": sells,
                "rsi": rsi_v,
                "adx": adx_v,
                "atr_pct": atr_pct,
                "vol_x": vol_ratio,
                "regime": reg.get("regime"),
                "score": score,
                "reason": reg.get("reason", ""),
            }
        )
    rows.sort(key=lambda r: r.get("score", -99), reverse=True)
    if args.json:
        _print_json(rows)
        return 0
    table = Table(title="Screener (15m closed bars, sorted by score)")
    for col, just in (
        ("Symbol", "left"),
        ("Price", "right"),
        ("Trend%", "right"),
        ("B/S", "right"),
        ("RSI", "right"),
        ("ADX", "right"),
        ("ATR%", "right"),
        ("Volx", "right"),
        ("Regime", "left"),
        ("Score", "right"),
    ):
        table.add_column(col, justify=just)
    for r in rows:
        if "error" in r:
            table.add_row(r["symbol"], "-", "-", "-", "-", "-", "-", "-", r["error"], "-")
            continue
        table.add_row(
            r["symbol"],
            f"{r['price']:,.6g}",
            f"{r['chg_pct']:+.1f}%",
            f"{r['buys']}/{r['sells']}",
            f"{r['rsi']:.0f}" if r["rsi"] is not None else "-",
            f"{r['adx']:.0f}" if r["adx"] is not None else "-",
            f"{r['atr_pct']:.2f}" if r["atr_pct"] is not None else "-",
            f"{r['vol_x']:.1f}x" if r["vol_x"] is not None else "-",
            str(r["regime"]),
            str(r["score"]),
        )
    console.print(table)
    console.print(
        "[dim]score = (BUY votes - SELL votes) + regime bonus (bull +1, bear/volatile -2)[/dim]"
    )
    return 0


def cmd_regime(args) -> int:
    cfg = _load(args)
    client = _client(cfg)
    from glmbot.klines import Klines
    from glmbot.regime import detect_regime

    syms = args.symbols or cfg.symbols
    rows = []
    for sym in syms:
        s = sym.upper()
        if not s.endswith(cfg.quote_asset):
            s += cfg.quote_asset
        try:
            raw = client.klines(s, "15m", limit=120)
        except Exception as e:
            rows.append({"symbol": s, "regime": "unknown", "reason": str(e)[:100]})
            continue
        k = Klines(raw).closed()
        try:
            info = detect_regime(k)
        except Exception as e:
            info = {"regime": "unknown", "reason": str(e)[:100]}
        rows.append({"symbol": s, **info})
    if args.json:
        _print_json(rows)
        return 0
    table = Table(title="Market regime (15m closed bars)")
    for col in ("Symbol", "Regime", "ADX", "ATR%", "Diagnosis"):
        table.add_column(col)
    for r in rows:
        adx_s = f"{r.get('adx')}" if r.get("adx") is not None else "-"
        atr_s = f"{r.get('atr_pct')}%" if r.get("atr_pct") is not None else "-"
        table.add_row(
            r["symbol"], str(r.get("regime", "?")), adx_s, atr_s, str(r.get("reason", ""))[:60]
        )
    console.print(table)
    return 0


def cmd_funding(args) -> int:
    cfg = _load(args)
    client = _client(cfg)
    if cfg.market != "futures":
        msg = "funding is futures-only (config trade_type=spot)"
        if args.json:
            _print_json({"ok": False, "error": msg})
        else:
            console.print(f"[yellow]{msg}[/yellow]")
        return 1
    import time as _time

    syms = args.symbols or cfg.symbols
    rows = []
    for sym in syms:
        s = sym.upper()
        if not s.endswith(cfg.quote_asset):
            s += cfg.quote_asset
        try:
            now_ms = int(_time.time() * 1000)
            evts = client.funding_history(s, now_ms - 7 * 86400 * 1000, now_ms)
        except Exception as e:
            rows.append({"symbol": s, "error": str(e)[:100]})
            continue
        rates = [r for _, r in evts]
        last = rates[-1] if rates else None
        avg = (sum(rates) / len(rates)) if rates else None
        rows.append(
            {
                "symbol": s,
                "events_7d": len(rates),
                "last_pct": (last * 100 if last is not None else None),
                "avg_7d_pct": (avg * 100 if avg is not None else None),
                "max_7d_pct": (max(rates) * 100 if rates else None),
            }
        )
    if args.json:
        _print_json(rows)
        return 0
    table = Table(title="Funding rates (positive = longs pay shorts)")
    for col, just in (
        ("Symbol", "left"),
        ("Events7d", "right"),
        ("Last%", "right"),
        ("Avg7d%", "right"),
        ("Max7d%", "right"),
    ):
        table.add_column(col, justify=just)

    def _fmt(x):
        return f"{x:.4f}" if x is not None else "-"

    for r in rows:
        if "error" in r:
            table.add_row(r["symbol"], "-", "-", "-", r["error"][:40])
            continue
        table.add_row(
            r["symbol"],
            str(r["events_7d"]),
            _fmt(r["last_pct"]),
            _fmt(r["avg_7d_pct"]),
            _fmt(r["max_7d_pct"]),
        )
    console.print(table)
    return 0


def cmd_analyze(args) -> int:
    cfg = _load(args)
    from glmbot.storage import Store

    store = Store(cfg.sqlite_path)
    closed = store.closed_positions(cfg.mode, limit=10000)
    if args.json:
        from glmbot.report import trades_summary

        by_sym: dict[str, list] = {}
        by_strat: dict[str, list] = {}
        for p in closed:
            by_sym.setdefault(p["symbol"], []).append(p)
            by_strat.setdefault(p.get("strategy", "?"), []).append(p)

        def _agg(ps):
            pnls = [float(x.get("pnl_quote") or 0) for x in ps]
            wins = sum(1 for v in pnls if v > 0)
            gp = sum(v for v in pnls if v > 0)
            gl = sum(-v for v in pnls if v < 0)
            return {
                "trades": len(ps),
                "wins": wins,
                "win_rate": round(wins / len(ps) * 100, 1) if ps else 0.0,
                "pnl": round(sum(pnls), 2),
                "profit_factor": round(gp / gl, 2) if gl > 0 else None,
                "avg_win": round(gp / wins, 2) if wins else 0.0,
                "avg_loss": round(-gl / (len(ps) - wins), 2) if len(ps) - wins else 0.0,
            }

        _print_json(
            {
                "mode": cfg.mode,
                "closed": len(closed),
                "by_symbol": {k: _agg(v) for k, v in by_sym.items()},
                "by_strategy": {k: _agg(v) for k, v in by_strat.items()},
                "journal": trades_summary(store.trades(mode=cfg.mode, limit=100000)),
            }
        )
        return 0
    if not closed:
        console.print("[dim]no closed positions yet - run the bot or backtest first[/dim]")
        return 0
    pnls = [float(p.get("pnl_quote") or 0) for p in closed]
    wins = sum(1 for v in pnls if v > 0)
    gp = sum(v for v in pnls if v > 0)
    gl = sum(-v for v in pnls if v < 0)
    pf = (gp / gl) if gl > 0 else float("inf")
    console.print(f"[bold]Trade analytics ({cfg.mode}, {len(closed)} closed)[/]")
    console.print(
        f"Win rate {(wins / len(closed) * 100):.0f}% ({wins}/{len(closed)}) | "
        f"PnL {sum(pnls):+.2f} {cfg.quote_asset} | "
        f"PF {'inf' if pf == float('inf') else f'{pf:.2f}'}"
    )
    # by-symbol table
    table = Table(title="By symbol")
    for col, just in (
        ("Symbol", "left"),
        ("N", "right"),
        ("Win%", "right"),
        ("PnL", "right"),
        ("PF", "right"),
    ):
        table.add_column(col, justify=just)
    by_sym: dict[str, list] = {}
    for p in closed:
        by_sym.setdefault(p["symbol"], []).append(p)
    for sym in sorted(
        by_sym, key=lambda s: sum(float(x.get("pnl_quote") or 0) for x in by_sym[s]), reverse=True
    )[:15]:
        ps = by_sym[sym]
        pp = [float(x.get("pnl_quote") or 0) for x in ps]
        w = sum(1 for v in pp if v > 0)
        _gp = sum(v for v in pp if v > 0)
        _gl = sum(-v for v in pp if v < 0)
        _pf = (_gp / _gl) if _gl > 0 else float("inf")
        table.add_row(
            sym,
            str(len(ps)),
            f"{w / len(ps) * 100:.0f}%",
            f"{sum(pp):+.2f}",
            "inf" if _pf == float("inf") else f"{_pf:.2f}",
        )
    console.print(table)
    # by-strategy table
    table2 = Table(title="By strategy")
    for col, just in (("Strategy", "left"), ("N", "right"), ("Win%", "right"), ("PnL", "right")):
        table2.add_column(col, justify=just)
    by_st: dict[str, list] = {}
    for p in closed:
        by_st.setdefault(p.get("strategy", "?"), []).append(p)
    for st in sorted(
        by_st, key=lambda s: sum(float(x.get("pnl_quote") or 0) for x in by_st[s]), reverse=True
    )[:15]:
        ps = by_st[st]
        pp = [float(x.get("pnl_quote") or 0) for x in ps]
        w = sum(1 for v in pp if v > 0)
        table2.add_row(st, str(len(ps)), f"{w / len(ps) * 100:.0f}%", f"{sum(pp):+.2f}")
    console.print(table2)
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


def cmd_set_env(args) -> int:
    """Switch between demo (testnet) and real (mainnet) Binance API."""
    import re

    p = Path(args.config or "config.yml")
    if not p.exists():
        console.print(f"[red]{p} not found[/red]")
        return 1
    text = p.read_text(encoding="utf-8")
    want = "true" if args.env == "demo" else "false"
    new, n = re.subn(r"(testnet:\s*)(\w+)", rf"\g<1>{want}", text, count=1)
    if n == 0:
        console.print("[red]could not find api.testnet in config[/red]")
        return 1
    p.write_text(new, encoding="utf-8")
    if args.env == "real":
        console.print(
            "[green]OK env set to REAL (mainnet)[/green]"
            " - [red bold]real money. Paste mainnet keys, then run `doctor`[/]"
        )
    else:
        console.print(
            "[green]OK env set to DEMO (testnet)[/green] - safe to test, then run `doctor`"
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
    sp.add_argument(
        "--folds",
        type=int,
        default=1,
        help="walk-forward folds across the period (regime-labelled bull/bear/chop)",
    )
    sp.add_argument(
        "--ablate",
        action="store_true",
        help="strategy ablation: full set vs minus-one vs single runs",
    )
    sp.set_defaults(fn=cmd_backtest)

    sp = sub.add_parser("optimize", help="grid-search strategy params on real klines")
    sp.add_argument("-d", "--days", type=int, default=14)
    sp.add_argument(
        "--metric",
        default="sharpe",
        choices=["sharpe", "pnl", "calmar", "profit_factor", "win_rate"],
    )
    sp.add_argument("--max-combos", type=int, default=64)
    sp.add_argument(
        "--set", action="append", default=[], help="param grid: strategy.param=v1,v2 (repeatable)"
    )
    sp.set_defaults(fn=cmd_optimize)

    sp = sub.add_parser("screener", help="ranked watchlist scan (momentum + regime)")
    sp.add_argument("symbols", nargs="*", default=[])
    sp.set_defaults(fn=cmd_screener)

    sp = sub.add_parser("regime", help="market-regime diagnosis per symbol")
    sp.add_argument("symbols", nargs="*", default=[])
    sp.set_defaults(fn=cmd_regime)

    sp = sub.add_parser("funding", help="futures funding rates + 7d average")
    sp.add_argument("symbols", nargs="*", default=[])
    sp.set_defaults(fn=cmd_funding)

    sp = sub.add_parser("analyze", help="journal analytics by symbol + strategy")
    sp.set_defaults(fn=cmd_analyze)

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

    sp = sub.add_parser("set-env", help="set demo (testnet) | real (mainnet) API")
    sp.add_argument("env", choices=["demo", "real"])
    sp.set_defaults(fn=cmd_set_env)

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
