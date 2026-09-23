"""Binance REST client: public + signed endpoints (GET/POST), spot AND futures.

Hardened for production use:
  - HMAC-signed GET/POST with clock-sync offset + ``recvWindow``.
  - Jittered exponential backoff on 429/418/5xx + network errors.
  - Clear, actionable errors (incl. HTTP 451 geo-restriction guidance).
  - Hybrid data routing: futures *testnet* has a broken price feed, so market
    data is read from real mainnet spot while orders go to the testnet.
  - On-disk ``exchangeInfo`` cache (24 h TTL) to cut startup weight.
  - ``User-Agent: glmbot/<version>`` on every request.

Public surface used by the rest of the bot (stable):
  BinanceClient, BinanceError, SymbolFilter, RateLimiter
"""

from __future__ import annotations

import hashlib
import hmac
import json as _json
import logging
import random
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from . import __version__

log = logging.getLogger("glmbot.api")

PUB = "https://api.binance.com"
PUB_TN = "https://testnet.binance.vision"
FUT = "https://fapi.binance.com"
FUT_TN = "https://testnet.binancefuture.com"

USER_AGENT = f"glmbot/{__version__} (+https://github.com/glmbot)"
GEO_MSG = (
    "Binance returned HTTP 451 (restricted location). "
    "Your IP/country is geo-blocked by binance.com. "
    "Options: use the testnet, trade from an allowed region, "
    "or point the bot at a compliant endpoint/proxy."
)


class BinanceError(Exception):
    """Typed Binance / transport error with HTTP status + Binance code."""

    def __init__(self, status: int, code: int, msg: str):
        self.status = status
        self.code = code
        self.msg = msg
        super().__init__(f"[{status}] code={code} {msg}")

    def is_rate_limit(self) -> bool:
        return self.status in (429, 418)

    def is_retryable(self) -> bool:
        return self.status in (429, 418) or self.status >= 500 or self.status == 0


class RateLimiter:
    """Thread-safe sliding-window request limiter (client-side safety net).

    Binance enforces *weight*-based limits server-side; this limiter only
    guards against accidental tight loops.
    """

    def __init__(self, max_per_minute: int = 1000):
        self.max = max_per_minute
        self.lock = threading.Lock()
        self.hits: list[float] = []

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                self.hits = [h for h in self.hits if now - h < 60]
                if len(self.hits) < self.max:
                    self.hits.append(now)
                    return
            time.sleep(0.05)


class SymbolFilter:
    """Per-symbol LOT_SIZE / MARKET_LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL rules.

    Works for both spot and futures ``exchangeInfo`` payloads.
    """

    def __init__(self, info: dict[str, Any]):
        f = {flt.get("filterType"): flt for flt in info.get("filters", [])}
        lot = f.get("LOT_SIZE", {})
        mlot = f.get("MARKET_LOT_SIZE", lot)
        price_f = f.get("PRICE_FILTER", {})
        self.symbol: str = info.get("symbol", "")
        self.base_asset: str = info.get("baseAsset", "")
        self.quote_asset: str = info.get("quoteAsset", "")
        self.status: str = info.get("status", "TRADING")
        try:
            self.min_qty = Decimal(str(lot.get("minQty", "0")))
            self.max_qty = Decimal(str(lot.get("maxQty", "999999999")))
            self.max_market_qty = Decimal(str(mlot.get("maxQty", lot.get("maxQty", "999999999"))))
            self.step_size = Decimal(str(lot.get("stepSize", "0.00000001")))
            self.tick_size = Decimal(str(price_f.get("tickSize", "0.00000001")))
            notional = f.get("NOTIONAL", {}).get("minNotional")
            if notional is None:
                notional = f.get("MIN_NOTIONAL", {}).get("notional", "0")
                if notional is None:
                    notional = f.get("MIN_NOTIONAL", {}).get("minNotional", "0")
            self.min_notional = Decimal(str(notional or "0"))
        except (InvalidOperation, ValueError, TypeError):
            self.min_qty = Decimal("0")
            self.max_qty = Decimal("999999999")
            self.max_market_qty = Decimal("999999999")
            self.step_size = Decimal("0.00000001")
            self.tick_size = Decimal("0.00000001")
            self.min_notional = Decimal("0")

    # -- rounding ------------------------------------------------------
    def round_qty(self, qty: Decimal) -> Decimal:
        if self.step_size <= 0:
            return qty
        steps = (qty / self.step_size).to_integral_value(rounding="ROUND_DOWN")
        return steps * self.step_size

    def round_price(self, price: Decimal) -> Decimal:
        if self.tick_size <= 0:
            return price
        ticks = (price / self.tick_size).to_integral_value(rounding="ROUND_HALF_UP")
        return ticks * self.tick_size

    def clamp_qty(self, qty: Decimal, price: Decimal, market_order: bool = True) -> Decimal | None:
        """Return qty adjusted to exchange filters, or None if untradable."""
        if self.status != "TRADING":
            return None
        q = self.round_qty(qty)
        if q < self.min_qty:
            return None
        cap = self.max_market_qty if market_order else self.max_qty
        if q > cap:
            q = cap
            if q < self.min_qty:
                return None
        if self.min_notional and q * price < self.min_notional:
            return None
        return q

    def describe(self) -> str:
        return (
            f"{self.symbol}: minQty={self.min_qty} step={self.step_size} "
            f"tick={self.tick_size} minNotional={self.min_notional}"
        )


class BinanceClient:
    """Synchronous Binance REST wrapper for SPOT and USD-M FUTURES.

    Args:
        api_key / api_secret: empty string = public-only mode.
        testnet: Binance testnet when True, mainnet otherwise.
        market: ``"spot"`` or ``"futures"``.
        timeout: per-request seconds.
        session: injectable ``requests.Session`` (tests).
        data_market: force market-data source (``"spot"``); defaults to
            automatic hybrid routing for futures testnet.
        cache_dir: on-disk ``exchangeInfo`` cache directory.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        market: str = "spot",
        timeout: int = 15,
        session: requests.Session | None = None,
        data_market: str | None = None,
        cache_dir: str = "data/cache",
    ):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.testnet = bool(testnet)
        self.market = market if market in ("spot", "futures") else "spot"
        if market == "futures":
            self.base = FUT_TN if self.testnet else FUT
            self.info_path = "/fapi/v1/exchangeInfo"
            self.kline_path = "/fapi/v1/klines"
            self.price_path = "/fapi/v1/ticker/price"
            self.order_path = "/fapi/v1/order"
            self.open_orders_path = "/fapi/v1/openOrders"
            # Conditional orders (STOP_MARKET / TAKE_PROFIT_MARKET with
            # closePosition) live behind the Algo Order API, not /order.
            self.algo_order_path = "/fapi/v1/algoOrder"
            self.open_algo_orders_path = "/fapi/v1/openAlgoOrders"
            self.account_path = "/fapi/v2/account"
        else:
            self.base = PUB_TN if self.testnet else PUB
            self.info_path = "/api/v3/exchangeInfo"
            self.kline_path = "/api/v3/klines"
            self.price_path = "/api/v3/ticker/price"
            self.order_path = "/api/v3/order"
            self.open_orders_path = "/api/v3/openOrders"
            self.account_path = "/api/v3/account"
        # --- data source routing ---
        self.data_base = PUB
        self.data_info_path = "/api/v3/exchangeInfo"
        self.data_kline_path = "/api/v3/klines"
        self.data_price_path = "/api/v3/ticker/price"
        if data_market == "spot" or (self.market == "futures" and self.testnet):
            self.using_mainnet_data = True
        else:
            self.using_mainnet_data = self.market != "futures" or not self.testnet
        self._exec_filters: dict[str, SymbolFilter] = {}
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.limiter = RateLimiter()
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self._filters: dict[str, SymbolFilter] = {}
        self._server_time_offset = 0
        if self.api_key:
            self.s.headers["X-MBX-APIKEY"] = self.api_key

    # ---------------- internals ----------------
    def _sleep_backoff(self, attempt: int) -> None:
        wait = min(2**attempt, 30) + random.uniform(0, 0.5)
        log.warning("retry %d/5 in %.1fs", attempt + 1, wait)
        time.sleep(wait)

    def _retry(self, fn, *a, **kw):
        last: Exception | None = None
        for attempt in range(5):
            try:
                return fn(*a, **kw)
            except BinanceError as e:
                last = e
                if e.is_retryable():
                    log.warning("Binance %s", e)
                    if attempt < 4:
                        self._sleep_backoff(attempt)
                        continue
                raise
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
                log.warning("network error %s", e)
                if attempt < 4:
                    self._sleep_backoff(attempt)
                    continue
                break
        raise BinanceError(0, 0, f"retries exhausted: {last}")

    def _request(
        self,
        base: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        method: str = "GET",
    ) -> Any:
        self.limiter.acquire()
        params = dict(params or {})
        url = f"{base}{path}"
        try:
            if signed:
                if not self.api_key:
                    raise BinanceError(0, 0, "no API key configured (public-only mode)")
                params["timestamp"] = int(time.time() * 1000) + self._server_time_offset
                params["recvWindow"] = 10000
                qs = urlencode(params, doseq=True)
                sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
                qs = f"{qs}&signature={sig}"
                if method == "GET":
                    r = self.s.get(f"{url}?{qs}", timeout=self.timeout)
                elif method == "DELETE":
                    r = self.s.delete(f"{url}?{qs}", timeout=self.timeout)
                else:
                    r = self.s.post(
                        url,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        data=qs,
                        timeout=self.timeout,
                    )
            else:
                if method == "GET":
                    r = self.s.get(url, params=params, timeout=self.timeout)
                else:
                    r = self.s.post(url, data=params, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:
            raise BinanceError(0, 0, f"transport error: {e}") from e
        if r.status_code == 451:
            raise BinanceError(451, -1, GEO_MSG)
        if r.status_code not in (200, 201):
            code = -1
            msg = r.text[:500]
            try:
                j = r.json()
                code = int(j.get("code", -1))
                msg = str(j.get("msg", r.text))[:500]
            except Exception:
                pass
            # Enrich common auth errors with remediation hints.
            if code == -2015:
                msg += " - check key permissions / IP whitelist / testnet-vs-mainnet mismatch"
            elif code == -1021:
                msg += " - clock skew; run test-connection (auto syncs time)"
            raise BinanceError(r.status_code, code, msg)
        try:
            return r.json()
        except Exception:
            return r.text

    def _sget(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._retry(self._request, self.base, path, params, True, "GET")

    def _spost(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._retry(self._request, self.base, path, params, True, "POST")

    def _sdelete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._retry(self._request, self.base, path, params, True, "DELETE")

    # ---------------- time ----------------
    def ping(self) -> bool:
        try:
            p = "/fapi/v1/ping" if self.market == "futures" else "/api/v3/ping"
            r = self.s.get(f"{self.base}{p}", timeout=self.timeout)
            return r.status_code == 200
        except Exception:
            return False

    def server_time(self) -> int:
        p = "/fapi/v1/time" if self.market == "futures" else "/api/v3/time"
        j = self.s.get(f"{self.base}{p}", timeout=self.timeout).json()
        return int(j["serverTime"])

    def sync_time(self) -> int:
        try:
            st = self.server_time()
            self._server_time_offset = st - int(time.time() * 1000)
        except Exception as e:
            log.warning("time sync failed (continuing with local clock): %s", e)
        return self._server_time_offset

    # ---------------- exchange info (cached) ----------------
    def _cache_path(self, name: str) -> Path:
        safe = f"{self.market}-{'testnet' if self.testnet else 'main'}-{name}.json"
        return self.cache_dir / safe

    def _read_cache(self, name: str, ttl_sec: int = 86_400) -> Any | None:
        try:
            p = self._cache_path(name)
            if not p.exists():
                return None
            if time.time() - p.stat().st_mtime > ttl_sec:
                return None
            return _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, name: str, payload: Any) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(name).write_text(_json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    # ---------------- public market data ----------------
    def exchange_filters(self) -> dict[str, SymbolFilter]:
        """USDT data-market filters (for decisions + spot sizing). Cached."""
        if self._filters:
            return self._filters
        base = self.data_base if self.using_mainnet_data else self.base
        path = self.data_info_path if self.using_mainnet_data else self.info_path
        cache_key = f"data-filters-{'main' if self.using_mainnet_data else 'exec'}"
        cached = self._read_cache(cache_key)
        j = cached if cached is not None else self._retry(self._request, base, path)
        if cached is None:
            self._write_cache(cache_key, j)
        for s in j.get("symbols", []) if isinstance(j, dict) else []:
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                try:
                    self._filters[s["symbol"]] = SymbolFilter(s)
                except Exception:
                    continue
        return self._filters

    def get_filter(self, symbol: str) -> SymbolFilter:
        """ORDER-SIZING filter from the EXECUTION market."""
        if self.market == self._data_market_name():
            return self._lookup_filter(self.exchange_filters(), symbol)
        if not self._exec_filters:
            cache_key = "exec-filters"
            cached = self._read_cache(cache_key)
            j = (
                cached
                if cached is not None
                else self._retry(self._request, self.base, self.info_path)
            )
            if cached is None:
                self._write_cache(cache_key, j)
            for s in j.get("symbols", []) if isinstance(j, dict) else []:
                if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                    try:
                        self._exec_filters[s["symbol"]] = SymbolFilter(s)
                    except Exception:
                        continue
        return self._lookup_filter(self._exec_filters, symbol)

    def _data_market_name(self) -> str:
        return "spot" if self.using_mainnet_data else self.market

    @staticmethod
    def _lookup_filter(filters: dict[str, SymbolFilter], symbol: str) -> SymbolFilter:
        if symbol not in filters:
            raise BinanceError(
                0, 0, f"no USDT filter for {symbol} (delisted or wrong quote asset?)"
            )
        return filters[symbol]

    def klines(
        self, symbol: str, interval: str = "15m", limit: int = 300, end_time: int | None = None
    ) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("klines limit must be 1..1000")
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        base = self.data_base if self.using_mainnet_data else self.base
        path = self.data_kline_path if self.using_mainnet_data else self.kline_path
        j = self._retry(self._request, base, path, params)
        out = []
        for k in j:
            try:
                out.append(
                    {
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": int(k[6]),
                    }
                )
            except (IndexError, ValueError, TypeError):
                continue
        return out

    def ticker_price(self, symbol: str) -> float:
        prices = self.ticker_prices([symbol])
        return prices[symbol]

    def ticker_prices(self, symbols: list[str]) -> dict[str, float]:
        """Batch price fetch (one HTTP call for N symbols)."""
        base = self.data_base if self.using_mainnet_data else self.base
        path = self.data_price_path if self.using_mainnet_data else self.price_path
        j = self._retry(self._request, base, path, {"symbols": json_dumps(symbols)})
        out: dict[str, float] = {}
        if isinstance(j, dict):  # single-symbol response shape
            j = [j]
        for row in j:
            try:
                out[row["symbol"]] = float(row["price"])
            except (KeyError, ValueError, TypeError):
                continue
        missing = [s for s in symbols if s not in out]
        if missing:
            raise BinanceError(0, 0, f"no price returned for: {', '.join(missing)}")
        return out

    # ---------------- account (signed) ----------------
    def account(self) -> dict[str, Any]:
        return self._sget(self.account_path)

    def spot_balances(self) -> dict[str, float]:
        acct = self.account()
        return {b.get("asset", ""): float(b.get("free", 0) or 0) for b in acct.get("balances", [])}

    # ---------------- orders (signed, POST) ----------------
    def get_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self._sget(self.order_path, {"symbol": symbol, "orderId": order_id})

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: str | None = None,
        quote_order_qty: str | None = None,
        reduce_only: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol, "side": side, "type": order_type}
        if quantity:
            params["quantity"] = quantity
        if quote_order_qty:
            params["quoteOrderQty"] = quote_order_qty
        if reduce_only and self.market == "futures":
            params["reduceOnly"] = "true"
        if extra:
            params.update(extra)
        res = self._spost(self.order_path, params)
        if isinstance(res, dict) and res.get("status") == "NEW" and res.get("type") == "MARKET":
            oid = res.get("orderId")
            for _ in range(10):
                time.sleep(0.4)
                try:
                    chk = self.get_order(symbol, oid)
                    if float(chk.get("executedQty", 0) or 0) > 0 and chk.get("status") in (
                        "FILLED",
                        "PARTIALLY_FILLED",
                    ):
                        return chk
                    if chk.get("status") == "FILLED":
                        return chk
                except BinanceError:
                    continue
        return res

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        """Cancel one open order. Returns Binance response (or raises)."""
        res = self._sdelete(self.order_path, {"symbol": symbol, "orderId": order_id})
        return res if isinstance(res, dict) else {"status": res}

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """All open orders (optionally per symbol). Empty list when none."""
        params = {"symbol": symbol} if symbol else {}
        res = self._sget(self.open_orders_path, params)
        return res if isinstance(res, list) else []

    # ---------------- futures-only helpers ----------------
    def _require_futures(self, what: str) -> None:
        if self.market != "futures":
            raise BinanceError(0, 0, f"{what} is futures-only (client market={self.market})")

    def place_protection_stop(
        self, symbol: str, side: str, quantity: str, stop_price: str, kind: str = "STOP_MARKET"
    ) -> dict[str, Any]:
        """Exchange-native safety net on the regular order endpoint: STOP_MARKET
        or TAKE_PROFIT_MARKET with explicit quantity + reduceOnly (close-only),
        triggered on MARK_PRICE. Fires on Binance even while the bot is down.
        Caller rounds quantity to stepSize and stop_price to tickSize."""
        self._require_futures("place_protection_stop")
        if kind not in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            raise ValueError(f"unknown protection kind: {kind!r}")
        return self._spost(
            self.order_path,
            {
                "symbol": symbol,
                "side": side,
                "type": kind,
                "quantity": quantity,
                "stopPrice": stop_price,
                "reduceOnly": "true",
                "workingType": "MARK_PRICE",
            },
        )

    def cancel_algo_order(self, symbol: str, algo_id: int) -> dict[str, Any]:
        """Cancel one algo (conditional) order. Returns Binance response."""
        self._require_futures("cancel_algo_order")
        res = self._sdelete(self.algo_order_path, {"symbol": symbol, "algoId": algo_id})
        return res if isinstance(res, dict) else {"status": res}

    def open_algo_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """All open algo (conditional) orders. Empty list when none."""
        self._require_futures("open_algo_orders")
        params = {"symbol": symbol} if symbol else {}
        res = self._sget(self.open_algo_orders_path, params)
        return res if isinstance(res, list) else []

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        self._require_futures("set_leverage")
        return self._spost("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def set_margin_type(self, symbol: str, isolated: bool = True) -> Any:
        self._require_futures("set_margin_type")
        try:
            return self._spost(
                "/fapi/v1/marginType",
                {"symbol": symbol, "marginType": "ISOLATED" if isolated else "CROSSED"},
            )
        except BinanceError as e:
            if e.code == -4046:  # already set to requested type
                return {"msg": "unchanged"}
            raise

    def futures_position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        self._require_futures("futures_position_risk")
        params = {"symbol": symbol} if symbol else {}
        return self._sget("/fapi/v2/positionRisk", params)

    def futures_balance(self) -> dict[str, Any]:
        """USDT-M futures wallet as {'free','total'} in USDT."""
        self._require_futures("futures_balance")
        acct = self.account()
        for a in acct.get("assets", []):
            if a.get("asset") == "USDT":
                return {
                    "free": float(a.get("availableBalance", a.get("walletBalance", 0)) or 0),
                    "total": float(a.get("walletBalance", 0) or 0),
                }
        return {"free": 0.0, "total": 0.0}

    def change_position_mode(self, hedge: bool = False) -> Any:
        self._require_futures("change_position_mode")
        try:
            return self._spost(
                "/fapi/v1/positionSide/dual",
                {"dualSidePosition": "true" if hedge else "false"},
            )
        except BinanceError as e:
            if e.code == -4059:  # already in requested mode
                return {"msg": "unchanged"}
            raise


def json_dumps(x: Any) -> str:
    return _json.dumps(x)
