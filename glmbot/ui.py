"""UI abstraction: Rich when available, clean ASCII otherwise.

Design goals
  - Core bot runs on ``requests + PyYAML`` only (Termux). When ``rich`` is
    installed you get color, tables, panels and progress - same call sites.
  - ``console()`` returns a process-wide shared console.
  - ``banner()``, ``rule()``, ``status_line()`` helpers keep CLI output
    consistent and professional across commands.
"""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.progress import Progress as _RichProgress
    from rich.table import Table as _RichTable

    HAS_RICH = True
except ImportError:  # minimal installs (Termux --light)
    HAS_RICH = False


class PlainConsole:
    def print(self, *args: Any, **kw: Any) -> None:
        text = " ".join(str(a) for a in args)
        out = _strip_markup(text)
        if out:
            print(out)

    def input(self, prompt: str = "") -> str:
        return input(_strip_markup(prompt))

    def log(self, *args: Any, **kw: Any) -> None:
        self.print(*args, **kw)

    def rule(self, title: str = "", **kw: Any) -> None:
        line = ("-" * 10 + f" {title} " + "-" * 10) if title else "-" * 40
        print(_strip_markup(line))


def _strip_markup(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "[":
            j = text.find("]", i)
            if j != -1:
                tag = text[i + 1 : j]
                if " " not in tag and "\n" not in tag and len(tag) < 40:
                    i = j + 1
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


class PlainTable:
    def __init__(self, title: str = "", show_lines: bool = False, **kw: Any):
        self.title = title
        self.columns: list[str] = []
        self.rows: list[list[str]] = []
        self._justify: list[str] = []

    def add_column(self, name: str, justify: str = "left", **kw: Any) -> None:
        self.columns.append(name)
        self._justify.append(justify)

    def add_row(self, *cells: Any, **kw: Any) -> None:
        self.rows.append([_strip_markup(str(c)) for c in cells])

    def _widths(self) -> list[int]:
        w = [len(c) for c in self.columns]
        for r in self.rows:
            for i, cell in enumerate(r):
                if i < len(w):
                    w[i] = max(w[i], len(cell))
        return w

    def __str__(self) -> str:
        if not self.columns:
            return ""
        # pad short rows defensively
        rows = [r + [""] * (len(self.columns) - len(r)) for r in self.rows]
        w = self._widths()
        sep = "+" + "+".join("-" * (x + 2) for x in w) + "+"
        head = "|" + "|".join(f" {self.columns[i]:<{w[i]}} " for i in range(len(w))) + "|"

        def fmt(cell: str, width: int, justify: str) -> str:
            return f" {cell:>{width}} " if justify == "right" else f" {cell:<{width}} "

        lines = [sep, head, sep]
        for r in rows:
            lines.append(
                "|" + "|".join(fmt(r[i], w[i], self._justify[i]) for i in range(len(w))) + "|"
            )
        lines.append(sep)
        if self.title:
            pad = max(0, (len(sep) - len(self.title)) // 2)
            lines = [" " * pad + self.title] + lines
        return "\n".join(lines)


class PlainPanel:
    def __init__(self, content: str, title: str = "", **kw: Any):
        self.content = _strip_markup(content)
        self.title = title

    def __str__(self) -> str:
        body = self.content
        if self.title:
            body = f"[{self.title}]\n{body}"
        return body


# ---------------- public API (mirrors rich usage in the codebase) ----------
if HAS_RICH:
    from rich import box as _box

    class _AsciiTable(_RichTable):
        """Rich Table that defaults to ASCII borders (cp1252-safe on Windows)."""

        def __init__(self, *args: Any, **kw: Any):
            kw.setdefault("box", _box.ASCII)
            super().__init__(*args, **kw)

    def _make_console() -> Any:
        # legacy_windows=False avoids the cp1252-only legacy renderer;
        # modern Windows 10+ handles ANSI, older ones fall back gracefully.
        try:
            return _RichConsole(legacy_windows=False)
        except TypeError:
            return _RichConsole()

    Console = _RichConsole
    Table = _AsciiTable
    Panel = _RichPanel
    Progress = _RichProgress
else:
    Console = PlainConsole  # type: ignore[assignment]
    Table = PlainTable  # type: ignore[assignment]
    Panel = PlainPanel  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]

_CONSOLE: Any | None = None


def console() -> Any:
    """Process-wide shared console instance."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = _make_console() if HAS_RICH else Console()
    return _CONSOLE


BANNER = r"""
   ____ _     __  __ ____   ___ _____
  / ___| |   |  \/  | __ ) / _ \_   _|
 | |  _| |   | |\/| |  _ \| | | || |
 | |_| | |___| |  | | |_) | |_| || |
  \____|_____|_|  |_|____/ \___/ |_|
  Binance spot + USD-M futures trading bot
""".rstrip("\n")


def banner(version: str = "") -> str:
    line = BANNER
    if version:
        line += f"\n  v{version}"
    return line


def rule(title: str = "") -> None:
    c = console()
    if hasattr(c, "rule"):
        c.rule(title)
    else:
        c.print(f"--- {title} ---")


def status_line(ok: bool, label: str, detail: str = "") -> str:
    icon = "OK" if ok else "FAIL"
    extra = f" - {detail}" if detail else ""
    return f"[{'green' if ok else 'red'}]{icon}[/] {label}{extra}"
