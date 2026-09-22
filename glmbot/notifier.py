"""Notifications: Telegram + generic webhook (JSON POST).

Reliability: short timeouts, per-destination rate limiting (1 msg / 1.2 s),
HTML-escaped Telegram text (parse_mode=HTML), structured webhook envelope
``{event, text, ts}``. Failures are logged, never raised - a dead webhook
must not crash the trading loop.
"""
from __future__ import annotations

import html
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

log = logging.getLogger("glmbot.notify")

_MIN_INTERVAL = 1.2


class Notifier:
    def __init__(self, telegram: Dict[str, Any] | None, webhook: Dict[str, Any] | None,
                 session: Optional[requests.Session] = None):
        telegram = telegram or {}
        webhook = webhook or {}
        self.tg_enabled = bool(telegram.get("enabled") and telegram.get("bot_token") and telegram.get("chat_id"))
        self.tg = telegram
        self.wh_enabled = bool(webhook.get("enabled") and webhook.get("url"))
        self.wh = webhook
        self._s = session or requests.Session()
        self._last_send = 0.0

    @property
    def enabled(self) -> bool:
        return self.tg_enabled or self.wh_enabled

    def _throttle(self) -> None:
        gap = time.time() - self._last_send
        if gap < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - gap)
        self._last_send = time.time()

    def send(self, text: str, event: str = "info") -> None:
        """Send a plain-text alert to all enabled destinations."""
        if not self.enabled:
            return
        self._throttle()
        if self.tg_enabled:
            try:
                self._s.post(
                    f"https://api.telegram.org/bot{self.tg['bot_token']}/sendMessage",
                    json={"chat_id": self.tg["chat_id"],
                          "text": html.escape(text),
                          "parse_mode": "HTML",
                          "disable_web_page_preview": True},
                    timeout=10,
                )
            except Exception as e:
                log.warning("telegram notify failed: %s", e)
        if self.wh_enabled:
            try:
                self._s.post(
                    self.wh["url"],
                    json={"event": event, "text": text,
                          "ts": datetime.now(timezone.utc).isoformat()},
                    timeout=10,
                )
            except Exception as e:
                log.warning("webhook notify failed: %s", e)

    # -- convenience wrappers (consistent formatting across trader/CLI) --
    def trade(self, text: str) -> None:
        self.send(text, event="trade")

    def error(self, text: str) -> None:
        self.send(f"ERROR: {text}", event="error")

    def kill_switch(self, text: str) -> None:
        self.send(text, event="kill_switch")


def notify_factory(cfg) -> Notifier:
    return Notifier(getattr(cfg, "telegram", {}), getattr(cfg, "webhook", {}))


def format_buy(mode: str, symbol: str, qty: float, price: float, notional: float,
               quote: str, sl: float, tp: float, reason: str,
               leverage: int = 1, market: str = "spot") -> str:
    lev = f" {leverage}x" if market == "futures" else ""
    return (
        f"[{mode.upper()}{lev}] BUY {symbol} qty={qty:.6f} @ {price:.6g} "
        f"(notional {notional:.2f} {quote}) | SL {sl:.6g} TP {tp:.6g} | {reason}"
    )


def format_sell(mode: str, symbol: str, qty: float, price: float, pnl: float,
                quote: str, pct: float, reason: str,
                leverage: int = 1, market: str = "spot") -> str:
    lev = f" {leverage}x" if market == "futures" else ""
    emoji = "[WIN]" if pnl >= 0 else "[LOSS]"
    return (
        f"{emoji} [{mode.upper()}{lev}] SELL {symbol} qty={qty:.6f} @ {price:.6g} | "
        f"PnL {pnl:+.2f} {quote} ({pct:+.2f}%) | {reason}"
    )
