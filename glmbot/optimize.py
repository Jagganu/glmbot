"""Grid-search parameter optimization over the faithful backtester.

Searches cartesian products of ``strategy.param`` value lists (e.g.
``{"ema_cross.fast": [5, 9, 12]}``), runs :class:`Backtester` per combo on
real klines, and ranks by a chosen metric. Pure Python, no extra deps.

Keep grids small: combos explode multiplicatively. The CLI caps at 64 by
default and truncates with a warning so a phone never melts.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from typing import Any

log = logging.getLogger("glmbot.optimize")


def grid_combos(param_grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand ``{"strat.param": [values]}`` into a list of flat dicts."""
    keys = sorted(param_grid)
    vals = [list(param_grid[k]) or [None] for k in keys]
    combos: list[dict[str, Any]] = []
    for prod in itertools.product(*vals):
        combos.append(dict(zip(keys, prod, strict=True)))
    return combos


def apply_combo(cfg, combo: dict[str, Any]):
    """Return a copy of ``cfg`` with ``strategy.param`` overrides applied."""
    params = {s: dict(p) for s, p in cfg.strategy_params.items()}
    for dotted, value in combo.items():
        if "." not in dotted:
            raise ValueError(f"bad grid key {dotted!r} (want 'strategy.param')")
        strat, param = dotted.split(".", 1)
        params.setdefault(strat, {})[param] = value
    return dataclasses.replace(cfg, strategy_params=params)


def combo_label(combo: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))


def run_optimization(
    cfg,
    klines: dict,
    param_grid: dict[str, list],
    funding: dict | None = None,
    metric: str = "sharpe",
    max_combos: int = 64,
) -> list[dict[str, Any]]:
    """Run the grid and return ranked rows (best first).

    ``metric`` in {sharpe, pnl, calmar, profit_factor, win_rate}. Score is the
    portfolio aggregate across symbols (mean Sharpe / total PnL), so the winner
    generalizes instead of overfitting one pair.
    """
    from .backtest import Backtester, summarize_run
    from .metrics import calmar_ratio

    combos = grid_combos(param_grid)
    truncated = False
    if len(combos) > max_combos:
        combos = combos[:max_combos]
        truncated = True
    rows: list[dict[str, Any]] = []
    for combo in combos:
        try:
            sub = apply_combo(cfg, combo)
        except Exception as e:
            log.warning("skip combo %s: %s", combo, e)
            continue
        try:
            results = Backtester(sub, klines, funding=funding).run()
        except Exception as e:
            log.warning("combo %s failed: %s", combo, e)
            continue
        s = summarize_run(results, cfg.risk.quote_budget)
        sharpes = [r.sharpe for r in results.values()]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
        calmars = []
        for r in results.values():
            try:
                calmars.append(calmar_ratio(r.equity_curve))
            except Exception:
                calmars.append(0.0)
        avg_calmar = sum(calmars) / len(calmars) if calmars else 0.0
        pfs = [r.profit_factor for r in results.values() if r.profit_factor != float("inf")]
        avg_pf = sum(pfs) / len(pfs) if pfs else 0.0
        key = {
            "sharpe": avg_sharpe,
            "pnl": s["pnl_pct"],
            "calmar": avg_calmar,
            "profit_factor": avg_pf,
            "win_rate": s["win_rate"],
        }.get(metric, avg_sharpe)
        rows.append(
            {
                "params": dict(combo),
                "label": combo_label(combo),
                "trades": s["trades"],
                "win_rate": s["win_rate"],
                "pnl_pct": s["pnl_pct"],
                "worst_max_dd": s["worst_max_dd"],
                "sharpe": avg_sharpe,
                "calmar": avg_calmar,
                "profit_factor": avg_pf,
                "score": key,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    if truncated:
        log.warning("grid truncated to first %d combos (keep grids small)", max_combos)
    return rows
