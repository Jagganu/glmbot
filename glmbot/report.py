"""Terminal reports: status dashboard, positions, trades, equity, backtest.

Rich when available, plain ASCII otherwise (Termux-light installs).
All ``show_*`` functions render to the shared console; ``*_dict`` helpers
return JSON-serializable structures for ``--json`` machine output.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .backtest import BTResult
from .storage import Store
from .ui import Panel, Table, console

_console = console()


def fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


def colored_pct(x: float) -> str:
    color = "green" if x >= 0 else "red"
    return f"[{color}]{fmt_pct(x)}[/{color}]"


# ---------------- status dashboard ----------------


def status_dict(cfg, store: Store, client, positions: list[dict]) -> dict[str, Any]:
    quotes: dict[str, float] = {}
    for p in positions:
        try:
            quotes[p["symbol"]] = client.ticker_price(p["symbol"])
        except Exception:
            quotes[p["symbol"]] = float(p["entry_price"])
    rows = []
    unreal = 0.0
    for p in positions:
        cur = quotes[p["symbol"]]
        pnl = (cur - p["entry_price"]) * p["qty"]
        unreal += pnl
        rows.append(
            {
                "symbol": p["symbol"],
                "qty": p["qty"],
                "entry": p["entry_price"],
                "current": cur,
                "stop_loss": p.get("stop_loss"),
                "take_profit": p.get("take_profit"),
                "protected": bool(p.get("stop_order_id") or p.get("take_order_id")),
                "pnl_quote": round(pnl, 2),
                "pnl_pct": round((cur / p["entry_price"] - 1) * 100, 2)
                if p["entry_price"]
                else 0.0,
            }
        )
    hist = store.equity_history(cfg.mode, limit=500)
    equity = hist[-1]["total"] if hist else None
    day_open = None
    for r in reversed(hist):
        if str(r.get("ts", ""))[:10] != (hist[-1].get("ts", "")[:10] if hist else ""):
            break
        day_open = r["total"]
    day_pnl = (equity - day_open) if (equity is not None and day_open) else 0.0
    halted, halt_reason = False, ""
    try:
        from .risk import RiskManager

        halted, halt_reason = RiskManager(cfg.risk, store).halted(cfg.mode)
    except Exception:
        pass
    return {
        "mode": cfg.mode,
        "market": cfg.market,
        "env": cfg.env_label,
        "positions": rows,
        "n_positions": len(rows),
        "unrealized_pnl": round(unreal, 2),
        "equity": equity,
        "day_pnl": round(day_pnl, 2),
        "kill_switch": halted,
        "kill_reason": halt_reason,
    }


def show_status(cfg, store: Store, client, positions: list[dict]) -> None:
    d = status_dict(cfg, store, client, positions)
    _console.print(
        f"[bold]glmbot[/] | {cfg.env_label} | "
        f"{d['n_positions']}/{cfg.risk.max_open_positions} positions"
    )
    if d["equity"] is not None:
        _console.print(
            f"Equity [bold]{fmt_money(d['equity'])} {cfg.quote_asset}[/] | "
            f"day {colored_pct(d['day_pnl'] / d['equity'] * 100) if d['equity'] else '-'} "
            f"({d['day_pnl']:+.2f}) | unrealized {d['unrealized_pnl']:+.2f}"
        )
    if d["kill_switch"]:
        _console.print(f"[red bold]!! halted - no new entries today ({d['kill_reason']})[/]")
    if not positions:
        _console.print("[dim]no open positions[/dim]")
        return
    table = Table(title=f"Open positions - {cfg.mode.upper()}", show_lines=False)
    for col, just in (
        ("Symbol", "left"),
        ("Qty", "right"),
        ("Entry", "right"),
        ("Current", "right"),
        ("SL", "right"),
        ("TP", "right"),
        ("PnL%", "right"),
        ("PnL", "right"),
    ):
        table.add_column(col, justify=just)
    for r in d["positions"]:
        sl = f"{r['stop_loss']:,.6g}" if r["stop_loss"] else "-"
        if r.get("protected"):
            sl += " (EX)"
        table.add_row(
            r["symbol"],
            f"{r['qty']:.6f}",
            f"{r['entry']:,.6g}",
            f"{r['current']:,.6g}",
            sl,
            f"{r['take_profit']:,.6g}" if r["take_profit"] else "-",
            colored_pct(r["pnl_pct"]),
            f"{r['pnl_quote']:+.2f}",
        )
    _console.print(table)
    if any(r.get("protected") for r in d["positions"]):
        _console.print(
            "[dim](EX) = exchange-native stop/TP armed (fires even if bot is down)[/dim]"
        )


def show_positions(positions: list[dict]) -> None:
    if not positions:
        _console.print("[dim]no open positions[/dim]")
        return
    table = Table(title="Open positions")
    for col in (
        "symbol",
        "strategy",
        "qty",
        "entry_price",
        "stop_loss",
        "take_profit",
        "opened_ts",
    ):
        table.add_column(
            col, justify="right" if col not in ("symbol", "strategy", "opened_ts") else "left"
        )
    for p in positions:
        table.add_row(
            p["symbol"],
            p["strategy"],
            f"{p['qty']:.6f}",
            f"{p['entry_price']:,.6g}",
            f"{p['stop_loss']:,.6g}" if p["stop_loss"] else "-",
            f"{p['take_profit']:,.6g}" if p["take_profit"] else "-",
            (p["opened_ts"] or "-")[:19],
        )
    _console.print(table)


def show_trades(trades: list[dict]) -> None:
    if not trades:
        _console.print("[dim]no trades yet[/dim]")
        return
    table = Table(title=f"Recent trades ({len(trades)})")
    table.add_column("ts", style="dim")
    table.add_column("mode")
    table.add_column("symbol", style="cyan")
    table.add_column("side")
    table.add_column("qty", justify="right")
    table.add_column("price", justify="right")
    table.add_column("quote", justify="right")
    table.add_column("fee", justify="right")
    table.add_column("reason", style="dim")
    for t in trades:
        side_color = "green" if t["side"] == "BUY" else "red"
        table.add_row(
            (t["ts"] or "")[:19],
            t["mode"],
            t["symbol"],
            f"[{side_color}]{t['side']}[/{side_color}]",
            f"{t['qty']:.6f}",
            f"{t['price']:,.6g}",
            fmt_money(t["quote_amt"]),
            f"{float(t.get('fee') or 0):.4f}",
            (t["reason"] or "")[:40],
        )
    _console.print(table)


def trades_summary(trades: list[dict]) -> dict[str, Any]:
    buys = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]
    fees = sum(float(t.get("fee") or 0) for t in trades)
    return {
        "n_trades": len(trades),
        "n_buys": len(buys),
        "n_sells": len(sells),
        "total_fees": round(fees, 4),
    }


def show_equity(history: list[dict]) -> None:
    if not history:
        _console.print("[dim]no equity snapshots yet - run the bot once to seed the curve[/dim]")
        return
    first, last = history[0]["total"], history[-1]["total"]
    chg = (last / first - 1) * 100 if first else 0.0
    _console.print(
        f"Equity [bold]{fmt_money(last)}[/] ({colored_pct(chg)} since {str(history[0]['ts'])[:10]}) | "
        f"{len(history)} snapshots"
    )
    table = Table(title="Equity snapshots (last 20)")
    table.add_column("ts", style="dim")
    table.add_column("cash", justify="right")
    table.add_column("positions", justify="right")
    table.add_column("total", justify="right")
    for e in history[-20:]:
        table.add_row(
            (e["ts"] or "")[:19],
            fmt_money(e["cash"]),
            fmt_money(e["positions_value"]),
            fmt_money(e["total"]),
        )
    _console.print(table)


def show_backtest(results: dict[str, BTResult], starting: float) -> None:
    from .backtest import summarize_run

    table = Table(title="Backtest (15m, 1-bar delay, intrabar stops, fees+slip+funding)")
    for col, just in (
        ("Symbol", "left"),
        ("Trades", "right"),
        ("Win%", "right"),
        ("PnL%", "right"),
        ("MaxDD%", "right"),
        ("PF", "right"),
        ("Sharpe", "right"),
        ("Fees", "right"),
        ("Fund", "right"),
        ("Final", "right"),
    ):
        table.add_column(col, justify=just)
    for sym, r in sorted(results.items()):
        pf = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
        table.add_row(
            sym,
            str(r.n_trades),
            f"{r.win_rate:.0f}%",
            colored_pct(r.pnl_pct),
            f"{r.max_drawdown_pct:.2f}",
            pf,
            f"{r.sharpe:.2f}",
            f"{r.total_fees:.2f}",
            f"{r.total_funding:+.2f}",
            fmt_money(r.final_equity),
        )
    _console.print(table)
    s = summarize_run(results, starting)
    _console.print(
        Panel(
            f"Symbols: {int(s['symbols'])} | trades: {int(s['trades'])} | "
            f"win rate: {s['win_rate']:.0f}% | worst MaxDD: {s['worst_max_dd']:.2f}%\n"
            f"Fees: {s['fees']:.2f} | funding: {s['funding']:+.2f}\n"
            f"Sum of independent per-symbol runs: [bold]{fmt_money(s['final'])}[/] "
            f"({colored_pct(s['pnl_pct'])} vs {fmt_money(s['allocated'])} allocated)\n"
            "[dim]Backtests != future results. Conservative by design: intrabar "
            "stops (SL-first), stop slippage, funding deducted.[/]",
            title="Portfolio",
        )
    )


def export_backtest_csv(results: dict[str, BTResult], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "symbol",
                "n_trades",
                "win_rate",
                "pnl_pct",
                "max_dd_pct",
                "profit_factor",
                "sharpe",
                "sortino",
                "expectancy",
                "fees",
                "funding",
                "final_equity",
            ]
        )
        for sym, r in sorted(results.items()):
            w.writerow(
                [
                    sym,
                    r.n_trades,
                    round(r.win_rate, 2),
                    round(r.pnl_pct, 2),
                    round(r.max_drawdown_pct, 2),
                    ("" if r.profit_factor == float("inf") else round(r.profit_factor, 2)),
                    round(r.sharpe, 2),
                    round(r.sortino, 2),
                    round(r.expectancy, 2),
                    round(r.total_fees, 2),
                    round(r.total_funding, 2),
                    round(r.final_equity, 2),
                ]
            )
    return path


def show_folds(folds: list[dict]) -> None:
    """Walk-forward table (#20): one row per fold with regime + consistency."""
    table = Table(title="Walk-forward folds (chronological, regime-labelled)")
    for col, just in (
        ("Fold", "left"),
        ("Trades", "right"),
        ("Win%", "right"),
        ("PnL%", "right"),
        ("MaxDD%", "right"),
        ("B&H%", "right"),
    ):
        table.add_column(col, justify=just)
    pos = 0
    for f in folds:
        if f["pnl_pct"] > 0:
            pos += 1
        table.add_row(
            f["label"],
            str(int(f["trades"])),
            f"{f['win_rate']:.0f}%",
            colored_pct(f["pnl_pct"]),
            f"{f['worst_max_dd']:.2f}",
            colored_pct(f["buy_hold_pct"]),
        )
    _console.print(table)
    _console.print(
        Panel(
            f"Profitable folds: [bold]{pos}/{len(folds)}[/] "
            f"({(pos / len(folds) * 100) if folds else 0:.0f}%)\n"
            "[dim]Edge needs green across bull, bear AND chop - one-regime "
            "wonders don't survive.[/]",
            title="Consistency",
        )
    )


def show_ablation(rows: list[dict], base_pnl: float) -> None:
    """Ablation table (#17): each row drops (or isolates) one strategy."""
    table = Table(title="Strategy ablation (delta vs full set)")
    for col, just in (
        ("Set", "left"),
        ("Trades", "right"),
        ("Win%", "right"),
        ("PnL%", "right"),
        ("dPnL%", "right"),
        ("Verdict", "left"),
    ):
        table.add_column(col, justify=just)
    for r in rows:
        delta = r["pnl_pct"] - base_pnl
        if r["kind"] == "full":
            verdict = "baseline"
        elif r["kind"] == "drop":
            verdict = "[green]KEEP[/]" if delta < 0 else "[red]drop candidate[/]"
        else:
            verdict = "standalone edge" if r["pnl_pct"] > 0 else "[dim]no edge alone[/]"
        table.add_row(
            r["label"],
            str(int(r["trades"])),
            f"{r['win_rate']:.0f}%",
            colored_pct(r["pnl_pct"]),
            colored_pct(delta) if r["kind"] != "full" else "-",
            verdict,
        )
    _console.print(table)
    _console.print("[dim]KEEP = removing it hurts. Drop candidates add cost without edge.[/]")
