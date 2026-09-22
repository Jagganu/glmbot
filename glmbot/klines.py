"""Lightweight kline container (pandas-free, Termux-friendly).

Behaves like the DataFrame sliver the strategies actually used:
  ``k.close / k.high / k.low / k.open / k.volume`` → float lists (oldest→newest)
  ``k[i]`` → dict row, ``len(k)``, ``k.rows(n)`` → last n rows, ``k.copy()``.

Plus professional helpers: validation, time-range slicing, CSV export and
basic statistics for diagnostics and backtest reporting.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


class Klines:
    __slots__ = ("open_time", "open", "high", "low", "close", "volume", "n")

    def __init__(self, rows: Optional[List[Dict]] = None):
        rows = rows or []
        self.open_time: List[int] = [int(r.get("open_time", i)) for i, r in enumerate(rows)]
        self.open: List[float] = [float(r.get("open", r.get("close", 0))) for r in rows]
        self.high: List[float] = [float(r.get("high", r.get("close", 0))) for r in rows]
        self.low: List[float] = [float(r.get("low", r.get("close", 0))) for r in rows]
        self.close: List[float] = [float(r.get("close", 0)) for r in rows]
        self.volume: List[float] = [float(r.get("volume", 0)) for r in rows]
        self.n = len(rows)
        self._sanitize()

    # ---- construction helpers ----
    @classmethod
    def from_lists(cls, close: List[float], high: Optional[List[float]] = None,
                   low: Optional[List[float]] = None, volume: Optional[List[float]] = None,
                   open_: Optional[List[float]] = None) -> "Klines":
        k = cls()
        k.close = [float(c) for c in close]
        n = len(k.close)
        k.high = [float(x) for x in high] if high else list(k.close)
        k.low = [float(x) for x in low] if low else list(k.close)
        k.open = [float(x) for x in open_] if open_ else list(k.close)
        k.volume = [float(x) for x in volume] if volume else [0.0] * n
        k.open_time = list(range(n))
        k.n = n
        k._sanitize()
        return k

    @classmethod
    def concat(cls, older: "Klines", newer: "Klines") -> "Klines":
        """older rows first, then newer rows."""
        k = cls()
        for f in ("open_time", "open", "high", "low", "close", "volume"):
            setattr(k, f, list(getattr(older, f)) + list(getattr(newer, f)))
        k.n = older.n + newer.n
        return k

    def drop_duplicates_by_time(self) -> "Klines":
        """Deduplicate on open_time, keeping the NEWEST row on conflicts."""
        seen = set()
        out = Klines()
        for f in ("open_time", "open", "high", "low", "close", "volume"):
            setattr(out, f, [])
        for i in range(self.n - 1, -1, -1):  # newest → oldest
            t = self.open_time[i]
            if t in seen:
                continue
            seen.add(t)
            out.open_time.insert(0, t)
            out.open.insert(0, self.open[i])
            out.high.insert(0, self.high[i])
            out.low.insert(0, self.low[i])
            out.close.insert(0, self.close[i])
            out.volume.insert(0, self.volume[i])
        out.n = len(out.open_time)
        return out

    # ---- access ----
    def __len__(self) -> int:
        return self.n

    def __bool__(self) -> bool:
        return self.n > 0

    def __getitem__(self, i: int) -> Dict:
        return {
            "open_time": self.open_time[i],
            "open": self.open[i], "high": self.high[i], "low": self.low[i],
            "close": self.close[i], "volume": self.volume[i],
        }

    def rows(self, count: int) -> List[Dict]:
        """Last ``count`` rows as dicts (chronological)."""
        if count <= 0:
            return []
        return [self[i] for i in range(max(0, self.n - count), self.n)]

    def copy(self) -> "Klines":
        k = Klines()
        for f in ("open_time", "open", "high", "low", "close", "volume"):
            setattr(k, f, list(getattr(self, f)))
        k.n = self.n
        return k

    def slice_time(self, start_ms: Optional[int] = None,
                   end_ms: Optional[int] = None) -> "Klines":
        """Filter rows to [start_ms, end_ms] (inclusive)."""
        idx = [
            i for i in range(self.n)
            if (start_ms is None or self.open_time[i] >= start_ms)
            and (end_ms is None or self.open_time[i] <= end_ms)
        ]
        k = Klines()
        for f in ("open_time", "open", "high", "low", "close", "volume"):
            src = getattr(self, f)
            setattr(k, f, [src[i] for i in idx])
        k.n = len(idx)
        return k

    # ---- diagnostics ----
    def validate(self) -> List[str]:
        """Return a list of data-quality issues (empty = clean)."""
        issues: List[str] = []
        if self.n == 0:
            return ["empty kline series"]
        lens = {len(getattr(self, f)) for f in ("open_time", "open", "high", "low", "close", "volume")}
        if len(lens) != 1:
            issues.append(f"ragged columns: lengths={sorted(lens)}")
        for i in range(self.n):
            if not (self.low[i] <= self.open[i] <= self.high[i] or True):
                pass  # open outside range is possible on some feeds; not fatal
            if not (self.low[i] <= self.close[i] <= self.high[i]):
                issues.append(f"row {i}: close {self.close[i]} outside [low, high]")
                break
            if self.close[i] <= 0:
                issues.append(f"row {i}: non-positive close")
                break
        for i in range(1, self.n):
            if self.open_time[i] < self.open_time[i - 1]:
                issues.append("open_time not sorted ascending")
                break
        dups = self.n - len(set(self.open_time))
        if dups:
            issues.append(f"{dups} duplicate open_time values")
        return issues

    def stats(self) -> Dict[str, float]:
        if not self.n:
            return {"n": 0}
        first, last = self.close[0], self.close[-1]
        ret = (last / first - 1) * 100 if first else 0.0
        return {
            "n": float(self.n),
            "first": first, "last": last,
            "return_pct": ret,
            "high": max(self.high), "low": min(self.low),
            "volume_total": sum(self.volume),
        }

    def to_csv(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["open_time", "open", "high", "low", "close", "volume"])
            for i in range(self.n):
                w.writerow([self.open_time[i], self.open[i], self.high[i],
                            self.low[i], self.close[i], self.volume[i]])
        return path

    # ---- internals ----
    def _sanitize(self) -> None:
        """Fix high/low inversions defensively (bad feed rows)."""
        for i in range(self.n):
            hi, lo = self.high[i], self.low[i]
            if hi < lo:
                self.high[i], self.low[i] = lo, hi

    def __repr__(self) -> str:
        if not self.n:
            return "Klines(n=0)"
        return f"Klines(n={self.n} last={self.close[-1]:.6g})"
