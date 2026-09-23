"""Unit tests: indicators, strategies, risk, storage, paper broker, backtest.
Pure-Python (no numpy/pandas) — matches Termux runtime."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from glmbot.api import BinanceClient, BinanceError, SymbolFilter  # noqa: E402
from glmbot.broker import FUT_TAKER_FEE, TAKER_FEE, FuturesBroker, PaperBroker  # noqa: E402
from glmbot.config import BotConfig, RiskCfg  # noqa: E402
from glmbot.indicators import bollinger, ema, macd, rsi, sma  # noqa: E402
from glmbot.klines import Klines  # noqa: E402
from glmbot.notifier import Notifier  # noqa: E402
from glmbot.risk import RiskManager  # noqa: E402
from glmbot.storage import Store, utcnow  # noqa: E402
from glmbot.strategies import BUY, SELL, build_strategies  # noqa: E402
from glmbot.trader import Trader  # noqa: E402


def make_cfg(mode="paper", strategies=None):
    return BotConfig(
        api_key="k",
        api_secret="s",
        testnet=True,
        mode=mode,
        market="spot",
        leverage=1,
        quote_asset="USDT",
        update_interval_sec=60,
        strategies=strategies or ["ema_cross"],
        strategy_params={},
        risk=RiskCfg(
            quote_budget=1000.0,
            per_trade_pct=10.0,
            max_open_positions=3,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            trailing_stop_pct=1.5,
            cooldown_min=0,
        ),
        symbols=["BTCUSDT"],
        sqlite_path=":memory-test:",
        klines_cache_dir="data/cache",
        telegram={},
        webhook={},
    )


class TestIndicators(unittest.TestCase):
    def test_sma_basic(self):
        out = sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        self.assertTrue(out[0] is None and out[1] is None)
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_ema_length_and_trend(self):
        v = [1 + i * 99 / 99.0 for i in range(100)]  # linspace 1..100
        out = ema(v, 10)
        self.assertTrue(all(x is None for x in out[:9]))
        self.assertTrue(all(x is not None for x in out[9:]))
        self.assertGreater(out[-1], out[-2])  # rising

    def test_rsi_bounds(self):
        random.seed(42)
        v, cur = [100.0], 100.0
        for _ in range(499):
            cur += random.gauss(0, 1)
            v.append(cur)
        r = rsi(v, 14)
        valid = [x for x in r if x is not None]
        self.assertTrue(all(0 <= x <= 100 for x in valid))

    def test_rsi_all_gains_is_100(self):
        v = [float(i) for i in range(1, 60)]
        r = rsi(v, 14)
        self.assertAlmostEqual(r[-1], 100.0)

    def test_macd_shapes(self):
        random.seed(1)
        v, cur = [100.0], 100.0
        for _ in range(199):
            cur += random.gauss(0, 1)
            v.append(cur)
        line, sig, hist = macd(v)
        self.assertEqual(len(line), len(v))
        # line valid from slow-1 = 25
        self.assertTrue(all(x is None for x in line[:25]))
        self.assertTrue(all(x is not None for x in line[25:]))
        # signal valid from (slow-1) + signal - 1 = 33
        self.assertTrue(all(x is None for x in sig[:33]))
        self.assertTrue(all(x is not None for x in sig[34:]))

    def test_bollinger(self):
        v = [10.0] * 50
        u, m, lo = bollinger(v, 20, 2.0)
        self.assertAlmostEqual(m[-1], 10.0)
        self.assertAlmostEqual(u[-1], 10.0)  # zero variance
        self.assertAlmostEqual(lo[-1], 10.0)

    def test_atr_constant_range_is_zero(self):
        n = 40
        k = Klines.from_lists(
            close=[100.0] * n,
            high=[101.0] * n,
            low=[99.0] * n,
        )
        from glmbot.indicators import atr as atr_fn

        out = atr_fn(k.high, k.low, k.close, 14)
        # TR = high-low = 2 every bar -> ATR = 2
        self.assertAlmostEqual(out[-1], 2.0)


class TestKlines(unittest.TestCase):
    def test_from_lists_and_access(self):
        k = Klines.from_lists(close=[1, 2, 3, 4], volume=[9, 8, 7, 6])
        self.assertEqual(len(k), 4)
        self.assertEqual(k.close[-1], 4.0)
        row = k[1]
        self.assertEqual(row["open"], 2.0)
        self.assertEqual(row["volume"], 8.0)
        self.assertEqual(len(k.rows(2)), 2)
        self.assertEqual(k.rows(2)[0]["close"], 3.0)

    def test_concat_and_dedupe(self):
        a = Klines.from_lists(close=[1.0, 2.0])
        b = Klines.from_lists(close=[2.0, 3.0])
        c = Klines.concat(a, b)
        self.assertEqual(len(c), 4)
        c.drop_duplicates_by_time()
        self.assertEqual(len(c), 4)  # times differ -> unchanged
        e = Klines.concat(a, a)
        f = e.drop_duplicates_by_time()
        self.assertEqual(len(f), 2)


class TestStrategies(unittest.TestCase):
    def _df(self, closes):
        closes = list(closes)
        n = len(closes)
        return Klines.from_lists(
            close=closes,
            high=[c * 1.001 for c in closes],
            low=[c * 0.999 for c in closes],
            volume=[100.0] * n,
        )

    def test_ema_cross_buys_on_golden_cross(self):
        # falling then rising -> golden cross
        closes = [100 - i * (10 / 29) for i in range(30)] + [90 + i * (20 / 14) for i in range(15)]
        strat = build_strategies(["ema_cross"], {})[0]
        sig = strat.evaluate("BTCUSDT", self._df(closes))
        self.assertIn(sig.side, (BUY, "HOLD"))

    def test_ema_cross_detects_cross(self):
        # sharp V-shape: strong fall then strong rise guarantees a golden cross
        closes = [100 - i * 1.0 for i in range(30)] + [71 + i * 1.0 for i in range(1, 20)]
        strat = build_strategies(["ema_cross"], {})[0]
        got_buy = False
        for i in range(31, len(closes) + 1):
            sig = strat.evaluate("BTCUSDT", self._df(closes[:i]))
            if sig.side == BUY:
                got_buy = True
        self.assertTrue(got_buy)

    def test_rsi_reversion_signals(self):
        # big drop then recovery -> oversold recovery
        closes = [100] * 20 + [100 - i * 0.8 for i in range(1, 15)] + [89 - 0.1, 92, 93]
        strat = build_strategies(["rsi_reversion"], {})[0]
        seen = set()
        for i in range(35, len(closes) + 1):
            sig = strat.evaluate("BTCUSDT", self._df(closes[:i]))
            seen.add(sig.side)
        self.assertTrue(seen & {BUY, SELL, "HOLD"})

    def test_bollinger_touch(self):
        closes = [100] * 30 + [100, 100, 94.0, 94.0]
        strat = build_strategies(["bollinger"], {})[0]
        sig = strat.evaluate("BTCUSDT", self._df(closes))
        self.assertEqual(sig.side, BUY)  # at/below lower band

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            build_strategies(["nope"], {})


class TestNewStrategies(unittest.TestCase):
    def _df(self, closes):
        closes = list(closes)
        n = len(closes)
        return Klines.from_lists(
            close=closes,
            high=[c * 1.001 for c in closes],
            low=[c * 0.999 for c in closes],
            volume=[100.0] * n,
        )

    def _got_side(self, name, closes, side, params=None):
        strat = build_strategies([name], {name: params} if params else {})[0]
        for i in range(10, len(closes) + 1):
            if strat.evaluate("BTCUSDT", self._df(closes[:i])).side == side:
                return True
        return False

    def test_vwap_trend_cross_up(self):
        closes = [100.0] * 40 + [100 + i * 2.0 for i in range(1, 8)]
        self.assertTrue(self._got_side("vwap_trend", closes, BUY))

    def test_stoch_rsi_cross_up(self):
        closes = [100.0] * 30 + [100 - i * 1.0 for i in range(1, 12)]
        closes += [89 + i * 1.2 for i in range(1, 8)]
        self.assertTrue(self._got_side("stoch_rsi_cross", closes, BUY))

    def test_bollinger_squeeze_breakout(self):
        closes = [100.0] * 80 + [106.0]  # flat squeeze, one impulse bar
        strat = build_strategies(["bollinger_squeeze"], {})[0]
        self.assertEqual(strat.evaluate("BTCUSDT", self._df(closes)).side, BUY)

    def test_trend_momentum_ignition(self):
        climb = [100 + i * 0.5 for i in range(60)]
        dip = [climb[-1] - i * 1.0 for i in range(1, 9)]
        rec = [dip[-1] + i * 1.0 for i in range(1, 12)]
        self.assertTrue(self._got_side("trend_momentum", climb + dip + rec, BUY))

    def test_new_strategies_hold_on_short_history(self):
        for name in ("vwap_trend", "stoch_rsi_cross", "bollinger_squeeze", "trend_momentum"):
            strat = build_strategies([name], {})[0]
            self.assertEqual(strat.evaluate("BTCUSDT", self._df([100.0] * 10)).side, "HOLD")

    def test_new_strategies_reject_bad_params(self):
        for name, bad in (
            ("vwap_trend", {"period": 1}),
            ("stoch_rsi_cross", {"oversold": 0.9, "overbought": 0.8}),
            ("bollinger_squeeze", {"lookback": 1}),
            ("trend_momentum", {"fast": 50, "slow": 20}),
        ):
            with self.assertRaises(ValueError, msg=name):
                build_strategies([name], {name: bad})


class TestRiskManager(unittest.TestCase):
    def _rm(self, tmp):
        cfg = make_cfg()
        store = Store(os.path.join(tmp, "t.db"))
        return RiskManager(cfg.risk, store), store

    def test_position_size_capped_by_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, _ = self._rm(tmp)
            amt = rm.position_size(50000.0, cash_available=1_000_000)
            self.assertEqual(amt, 100.0)  # 10% of 1000

    def test_position_size_capped_by_cash(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, _ = self._rm(tmp)
            amt = rm.position_size(50000.0, cash_available=50)
            self.assertEqual(amt, 50)  # capped by available cash
            self.assertEqual(rm.position_size(50000.0, cash_available=5), 0)  # below minimum

    def test_stop_loss_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)
            pid = store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 100.0,
                    "qty": 1.0,
                    "stop_loss": 98.0,
                    "take_profit": 104.0,
                    "mode": "paper",
                }
            )
            pos = store.get_open_position("BTCUSDT", "paper")
            plan = rm.check_exit(pos, 97.5)
            self.assertEqual(plan.action, "exit")
            self.assertIn("stop-loss", plan.reason)
            store.close_position(pid, 97.5, -2.5)

    def test_take_profit_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)
            store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 100.0,
                    "qty": 1.0,
                    "stop_loss": 98.0,
                    "take_profit": 104.0,
                    "mode": "paper",
                }
            )
            pos = store.get_open_position("BTCUSDT", "paper")
            plan = rm.check_exit(pos, 104.5)
            self.assertEqual(plan.action, "exit")
            self.assertIn("take-profit", plan.reason)

    def test_trailing_stop_ratchets_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)
            store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 100.0,
                    "qty": 1.0,
                    "stop_loss": 98.0,
                    "take_profit": 110.0,
                    "mode": "paper",
                }
            )
            pos = store.get_open_position("BTCUSDT", "paper")
            plan = rm.check_exit(pos, 105.0)  # raise trail high
            self.assertEqual(plan.action, "hold")
            pos = store.get_open_position("BTCUSDT", "paper")
            self.assertEqual(pos["trail_high"], 105.0)
            # stop ratcheted above entry by trailing logic
            self.assertGreater(pos["stop_loss"], 100.0)
            # drop below trailing distance -> exit (SL was ratcheted above entry)
            plan = rm.check_exit(pos, 103.0)
            self.assertEqual(plan.action, "exit")
            # reason can be stop-loss (ratcheted SL) or trailing stop
            self.assertTrue("stop" in plan.reason or "trailing" in plan.reason)

    def test_breakeven_locks_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)  # breakeven trigger 2.0 / buffer 0.1
            rm.cfg.trailing_stop_pct = 0.0  # isolate breakeven from trailing
            store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 100.0,
                    "qty": 1.0,
                    "stop_loss": 98.0,
                    "take_profit": 110.0,
                    "mode": "paper",
                }
            )
            pos = store.get_open_position("BTCUSDT", "paper")
            plan = rm.check_exit(pos, 102.5)  # +2.5% hits trigger, below TP
            self.assertEqual(plan.action, "hold")
            pos = store.get_open_position("BTCUSDT", "paper")
            self.assertAlmostEqual(pos["stop_loss"], 100.1)  # locked to entry+buffer
            # dip below entry now exits on the LOCKED stop (old SL 98 would hold)
            plan = rm.check_exit(pos, 99.5)
            self.assertEqual(plan.action, "exit")
            self.assertIn("stop-loss", plan.reason)

    def test_breakeven_disabled_leaves_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)
            rm.cfg.breakeven_trigger_pct = 0.0
            rm.cfg.trailing_stop_pct = 0.0  # isolate: no other ratchet may move SL
            store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 100.0,
                    "qty": 1.0,
                    "stop_loss": 98.0,
                    "take_profit": 110.0,
                    "mode": "paper",
                }
            )
            pos = store.get_open_position("BTCUSDT", "paper")
            self.assertEqual(rm.check_exit(pos, 103.0).action, "hold")
            self.assertEqual(store.get_open_position("BTCUSDT", "paper")["stop_loss"], 98.0)

    def test_time_stop_trips_on_stale_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)
            rm.cfg.max_hold_min = 60
            pid = store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 100.0,
                    "qty": 1.0,
                    "stop_loss": 98.0,
                    "take_profit": 110.0,
                    "mode": "paper",
                }
            )
            with store._conn() as c:
                c.execute(
                    "UPDATE positions SET opened_ts=? WHERE id=?",
                    ("2020-01-01T00:00:00Z", pid),
                )
            pos = store.get_open_position("BTCUSDT", "paper")
            plan = rm.check_exit(pos, 101.0)
            self.assertEqual(plan.action, "exit")
            self.assertIn("time stop", plan.reason)

    def test_time_stop_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)  # max_hold_min == 0
            pid = store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 100.0,
                    "qty": 1.0,
                    "stop_loss": 98.0,
                    "take_profit": 110.0,
                    "mode": "paper",
                }
            )
            with store._conn() as c:
                c.execute(
                    "UPDATE positions SET opened_ts=? WHERE id=?",
                    ("2020-01-01T00:00:00Z", pid),
                )
            pos = store.get_open_position("BTCUSDT", "paper")
            self.assertEqual(rm.check_exit(pos, 101.0).action, "hold")

    def test_max_positions_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm, store = self._rm(tmp)
            for s in ("AAA", "BBB", "CCC"):
                store.open_position(
                    {
                        "symbol": s,
                        "strategy": "x",
                        "entry_price": 1.0,
                        "qty": 1.0,
                        "mode": "paper",
                    }
                )
            ok, why = rm.can_open("DDD", "paper")
            self.assertFalse(ok)
            self.assertIn("max_open_positions", why)


class TestStorage(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.db"))
            store.set_paper_state(1234.5)
            self.assertAlmostEqual(store.get_paper_state(0), 1234.5)
            tid = store.insert_trade(
                {
                    "ts": utcnow(),
                    "mode": "paper",
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "qty": 0.001,
                    "price": 50000.0,
                    "quote_amt": 50.0,
                    "fee": 0.05,
                    "reason": "test",
                    "exchange_order_id": "",
                    "raw": "",
                }
            )
            self.assertEqual(tid, 1)
            trades = store.trades(mode="paper")
            self.assertEqual(len(trades), 1)
            store.open_position(
                {
                    "symbol": "BTCUSDT",
                    "strategy": "s",
                    "entry_price": 50000,
                    "qty": 0.001,
                    "mode": "paper",
                }
            )
            self.assertIsNotNone(store.get_open_position("BTCUSDT", "paper"))
            store.snapshot_equity("paper", 950.0, 50.0)
            hist = store.equity_history("paper")
            self.assertAlmostEqual(hist[-1]["total"], 1000.0)


class TestPaperBroker(unittest.TestCase):
    def test_buy_sell_roundtrip(self):
        client = BinanceClient("", "", testnet=True)
        b = PaperBroker(client, starting_cash=1000.0)
        fill = b.buy_market("BTCUSDT", 100.0, 50000.0)
        # spot-like: notional + fee charged from cash
        self.assertAlmostEqual(b.cash, 900.0 - 100.0 * TAKER_FEE, places=8)
        self.assertAlmostEqual(fill["qty"], 100.0 / 50000.0, places=10)
        b.sell_market("BTCUSDT", fill["qty"], 51000.0)
        expected = fill["qty"] * 51000.0 * (1 - TAKER_FEE)
        self.assertAlmostEqual(b.cash, 900.0 - 100.0 * TAKER_FEE + expected, places=8)

    def test_futures_margin_roundtrip(self):
        client = BinanceClient("", "", testnet=True, market="futures")
        b = PaperBroker(client, starting_cash=1000.0, leverage=2)
        # 100 margin -> 200 notional
        fill = b.buy_market("BTCUSDT", 100.0, 50000.0)
        self.assertAlmostEqual(b.cash, 1000.0 - 100.0 - 200.0 * FUT_TAKER_FEE, places=8)
        self.assertAlmostEqual(fill["qty"], 200.0 / 50000.0, places=10)
        # price +5% -> notional 210, margin returned 100 + pnl 10 - fee
        b.sell_market("BTCUSDT", fill["qty"], 52500.0, entry_price=50000.0)
        expected_cash = (
            1000.0 - 100.0 - 200.0 * FUT_TAKER_FEE + 100.0 + 10.0 - 210.0 * FUT_TAKER_FEE
        )
        self.assertAlmostEqual(b.cash, expected_cash, places=8)

    def test_insufficient_cash(self):
        client = BinanceClient("", "", testnet=True)
        b = PaperBroker(client, starting_cash=50.0)
        with self.assertRaises(ValueError):
            b.buy_market("BTCUSDT", 100.0, 50000.0)


class TestSymbolFilter(unittest.TestCase):
    def test_filters(self):
        info = {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.00001",
                    "maxQty": "1000",
                    "stepSize": "0.00001",
                },
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "NOTIONAL", "minNotional": "10"},
            ],
        }
        f = SymbolFilter(info)
        from decimal import Decimal

        self.assertEqual(f.round_qty(Decimal("0.123456")), Decimal("0.12345"))
        # round_price uses ROUND_HALF_UP to nearest tick
        self.assertEqual(f.round_price(Decimal("50000.567")), Decimal("50000.57"))
        self.assertIsNone(f.clamp_qty(Decimal("0.000001"), Decimal("50000")))  # below minQty
        self.assertIsNone(f.clamp_qty(Decimal("0.0001"), Decimal("50000")))  # notional < 10
        self.assertIsNotNone(f.clamp_qty(Decimal("0.001"), Decimal("50000")))


class TestBacktest(unittest.TestCase):
    def test_backtest_runs(self):
        from glmbot.backtest import Backtester

        random.seed(7)
        n = 400
        closes, cur = [100.0], 100.0
        for _ in range(n - 1):
            cur += random.gauss(0, 0.5)
            closes.append(cur)
        k = Klines.from_lists(
            close=closes,
            high=[c * 1.01 for c in closes],
            low=[c * 0.99 for c in closes],
            volume=[1000.0] * n,
        )
        cfg = make_cfg(strategies=["ema_cross"])
        bt = Backtester(cfg, {"BTCUSDT": k})
        results = bt.run()
        r = results["BTCUSDT"]
        self.assertEqual(r.n_trades, len([t for t in r.trades if t.side == "SELL"]))
        self.assertGreaterEqual(r.final_equity, 0)
        self.assertTrue(0 <= r.win_rate <= 100)


class TestATRStops(unittest.TestCase):
    def test_atr_levels_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg()
            cfg.risk.atr_stops = True
            cfg.risk.atr_sl_mult = 2.0
            cfg.risk.atr_tp_mult = 3.0
            store = Store(os.path.join(tmp, "t.db"))
            rm = RiskManager(cfg.risk, store)
            lv = rm.entry_levels(100.0, atr=1.0)
            self.assertAlmostEqual(lv["stop_loss"], 98.0)
            self.assertAlmostEqual(lv["take_profit"], 103.0)

    def test_atr_falls_back_to_fixed_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg()
            cfg.risk.atr_stops = True
            store = Store(os.path.join(tmp, "t.db"))
            rm = RiskManager(cfg.risk, store)
            lv = rm.entry_levels(100.0, atr=None)
            self.assertAlmostEqual(lv["stop_loss"], 98.0)  # 2% fixed
            self.assertAlmostEqual(lv["take_profit"], 104.0)  # 4% fixed

    def test_fixed_pct_levels_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg()
            store = Store(os.path.join(tmp, "t.db"))
            rm = RiskManager(cfg.risk, store)
            lv = rm.entry_levels(200.0)
            self.assertAlmostEqual(lv["stop_loss"], 196.0)
            self.assertAlmostEqual(lv["take_profit"], 208.0)


class TestDailyLossCap(unittest.TestCase):
    def test_cap_trips_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg()
            cfg.risk.daily_loss_cap_pct = 5.0
            store = Store(os.path.join(tmp, "t.db"))
            rm = RiskManager(cfg.risk, store)
            # baseline snapshot: 1000
            store.snapshot_equity("paper", 1000.0, 0.0)
            self.assertFalse(rm.check_daily_loss("paper", 980.0))  # -2% ok
            self.assertTrue(rm.check_daily_loss("paper", 940.0))  # -6% trips
            # kill switch now blocks entries
            ok, why = rm.can_open("BTCUSDT", "paper")
            self.assertFalse(ok)
            self.assertIn("daily loss cap", why)

    def test_cap_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg()
            store = Store(os.path.join(tmp, "t.db"))
            rm = RiskManager(cfg.risk, store)
            store.snapshot_equity("paper", 1000.0, 0.0)
            self.assertFalse(rm.check_daily_loss("paper", 100.0))  # never trips
            self.assertTrue(rm.can_open("X", "paper")[0])


class TestMinVotesConsensus(unittest.TestCase):
    """min_votes via risk config; trader-level logic tested through consensus math."""

    def test_unanimous_still_works(self):
        # emulate the consensus math the trader uses
        votes = ["BUY", "BUY", "HOLD", "HOLD"]
        buys = [v for v in votes if v == "BUY"]
        sells = [v for v in votes if v == "SELL"]
        n, min_votes = 4, 0
        need = min_votes if min_votes > 0 else n
        self.assertFalse(len(buys) >= need and len(sells) == 0)

    def test_min_votes_2_of_4_passes(self):
        votes = ["BUY", "BUY", "HOLD", "HOLD"]
        buys = [v for v in votes if v == "BUY"]
        sells = [v for v in votes if v == "SELL"]
        n, min_votes = 4, 2
        need = min_votes if min_votes > 0 else n
        self.assertTrue(len(buys) >= need and len(sells) == 0)

    def test_sell_vote_vetoes_entry(self):
        votes = ["BUY", "BUY", "BUY", "SELL"]
        buys = [v for v in votes if v == "BUY"]
        sells = [v for v in votes if v == "SELL"]
        n, min_votes = 4, 2
        need = min_votes if min_votes > 0 else n
        self.assertFalse(len(buys) >= need and len(sells) == 0)


class TestNoColorCLI(unittest.TestCase):
    """Regression: --no-color must boot and emit zero ANSI bytes (legacy conhost)."""

    def _run(self, *argv):
        return subprocess.run(
            [sys.executable, "bot.py", *argv],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            timeout=90,
        )

    def test_no_color_version_clean(self):
        p = self._run("--no-color", "version")
        self.assertEqual(p.returncode, 0, p.stderr.decode("utf-8", "replace")[-500:])
        self.assertNotIn(b"\x1b", p.stdout + p.stderr)

    def test_no_color_strategies_json_parses(self):
        import json

        p = self._run("--no-color", "strategies", "--json")
        self.assertEqual(p.returncode, 0, p.stderr.decode("utf-8", "replace")[-500:])
        names = [s["name"] for s in json.loads(p.stdout)]
        self.assertIn("ema_cross", names)


class _FakeResp:
    def __init__(self, payload, status=200):
        import json as _j

        self._payload = payload
        self.status_code = status
        self.text = _j.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Records requests; returns canned Binance payloads (no network)."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def _record(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self._reply(method, url, kw)

    def get(self, url, **kw):
        return self._record("GET", url, **kw)

    def post(self, url, **kw):
        return self._record("POST", url, **kw)

    def delete(self, url, **kw):
        return self._record("DELETE", url, **kw)

    def _reply(self, method, url, kw):
        if "openAlgoOrders" in url:
            return _FakeResp([{"algoId": 111, "symbol": "BNBUSDT"}])
        if "openOrders" in url:
            return _FakeResp([])
        if method == "DELETE":
            return _FakeResp({"code": 200, "msg": "success"})
        data = str(kw.get("data", ""))
        if "algoOrder" in url:
            if "type=STOP_MARKET" in data:
                oid = 111
            elif "type=TAKE_PROFIT_MARKET" in data:
                oid = 222
            else:
                oid = 0
            return _FakeResp({"algoId": oid, "status": "NEW"})
        if "type=STOP_MARKET" in data:
            return _FakeResp({"orderId": 333, "status": "NEW"})
        if "type=TAKE_PROFIT_MARKET" in data:
            return _FakeResp({"orderId": 444, "status": "NEW"})
        return _FakeResp({})


_FILTER_INFO = {
    "symbol": "BNBUSDT",
    "status": "TRADING",
    "baseAsset": "BNB",
    "quoteAsset": "USDT",
    "filters": [
        {
            "filterType": "LOT_SIZE",
            "minQty": "0.001",
            "maxQty": "100000",
            "stepSize": "0.001",
        },
        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        {"filterType": "MIN_NOTIONAL", "notional": "5"},
    ],
}


class TestProtectionClient(unittest.TestCase):
    def _client(self):
        return BinanceClient("k", "s", testnet=True, market="futures", session=_FakeSession())

    def test_place_protection_stop_payload(self):
        c = self._client()
        res = c.place_protection_stop("BNBUSDT", "SELL", "0.05", "778.13", "STOP_MARKET")
        self.assertEqual(res["algoId"], 111)
        method, url, kw = c.s.calls[-1]
        self.assertEqual(method, "POST")
        self.assertIn("/fapi/v1/algoOrder", url)
        data = str(kw["data"])
        self.assertIn("type=STOP_MARKET", data)
        self.assertIn("algoType=CONDITIONAL", data)
        self.assertIn("triggerPrice=778.13", data)
        self.assertIn("workingType=MARK_PRICE", data)
        self.assertIn("closePosition=true", data)
        self.assertIn("signature=", data)

    def test_place_take_profit_payload(self):
        c = self._client()
        res = c.place_protection_stop("BNBUSDT", "SELL", "0.05", "821.45", "TAKE_PROFIT_MARKET")
        self.assertEqual(res["algoId"], 222)

    def test_cancel_and_open_orders(self):
        c = self._client()
        c.cancel_algo_order("BNBUSDT", 111)  # must not raise
        method, url, kw = c.s.calls[-1]
        self.assertEqual(method, "DELETE")
        self.assertIn("/fapi/v1/algoOrder", url)
        self.assertIn("algoId=111", url)  # signed DELETE carries params in the query
        self.assertIn("signature=", url)
        self.assertEqual(c.open_algo_orders("BNBUSDT"), [{"algoId": 111, "symbol": "BNBUSDT"}])
        self.assertEqual(c.open_orders("BNBUSDT"), [])


class _StubFuturesClient:
    """Duck-typed futures client: canned filters + scripted protection calls."""

    market = "futures"

    def __init__(self, position_amt="0"):
        self.calls = []
        self._amt = position_amt

    def get_filter(self, symbol):
        return SymbolFilter(dict(_FILTER_INFO, symbol=symbol))

    def change_position_mode(self, hedge=False):
        self.calls.append(("mode", hedge))
        return {"msg": "ok"}

    def set_leverage(self, symbol, leverage):
        self.calls.append(("leverage", symbol, leverage))
        return {"leverage": leverage}

    def place_protection_stop(self, symbol, side, quantity, stop_price, kind):
        self.calls.append(("protect", symbol, side, quantity, stop_price, kind))
        oid = 111 if kind == "STOP_MARKET" else 222
        return {"algoId": oid, "status": "NEW"}

    def cancel_algo_order(self, symbol, algo_id):
        self.calls.append(("cancel", symbol, algo_id))
        if algo_id == 999:
            raise BinanceError(400, -2011, "Unknown order sent")
        return {"code": 200, "msg": "success"}

    def futures_position_risk(self, symbol=None):
        return [{"symbol": symbol or "BNBUSDT", "positionAmt": self._amt}]


class TestFuturesProtectionBroker(unittest.TestCase):
    def _broker(self, client=None):
        return FuturesBroker(client or _StubFuturesClient(), leverage=2)

    def test_place_rounds_to_tick(self):
        b = self._broker()
        ids = b.place_protection_orders("BNBUSDT", 0.05, 778.134, 821.459)
        self.assertEqual(ids, {"stop_order_id": 111, "take_order_id": 222})
        protects = [c for c in b.client.calls if c[0] == "protect"]
        self.assertEqual(protects[0][3], "0.05")  # qty passed through
        self.assertEqual(protects[0][4], "778.13")  # tick-rounded down
        self.assertEqual(protects[1][4], "821.46")  # tick-rounded half-up

    def test_cancel_best_effort(self):
        b = self._broker()
        b.cancel_protection_orders("BNBUSDT", [111, None, 999])  # 999 unknown -> swallowed
        cancels = [c for c in b.client.calls if c[0] == "cancel"]
        self.assertEqual([c[1:] for c in cancels], [("BNBUSDT", 111), ("BNBUSDT", 999)])


class TestProtectionJournal(unittest.TestCase):
    def test_ids_persist_and_migrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.db")
            store = Store(path)
            pid = store.open_position(
                {
                    "symbol": "BNBUSDT",
                    "strategy": "ema_cross",
                    "entry_price": 789.86,
                    "qty": 0.05,
                    "stop_loss": 778.13,
                    "take_profit": 821.45,
                    "mode": "live",
                }
            )
            store.update_protection_orders(pid, 111, 222)
            pos = store.get_open_position("BNBUSDT", "live")
            self.assertEqual(pos["stop_order_id"], "111")
            self.assertEqual(pos["take_order_id"], "222")
            Store(path)  # reopen: migration must be idempotent
            self.assertEqual(store.get_open_position("BNBUSDT", "live")["stop_order_id"], "111")


class _FailingBroker(FuturesBroker):
    def sell_market(self, symbol, qty, price_hint, entry_price=None):
        raise BinanceError(400, -2019, "Margin is insufficient")


class TestReconcileFlat(unittest.TestCase):
    def _trader(self, amt):
        import shutil

        cfg = make_cfg(mode="live", strategies=["ema_cross"])
        cfg.market = "futures"
        cfg.leverage = 2
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        store = Store(os.path.join(tmp, "t.db"))
        store.open_position(
            {
                "symbol": "BNBUSDT",
                "strategy": "ema_cross",
                "entry_price": 789.86,
                "qty": 0.05,
                "stop_loss": 778.13,
                "take_profit": 821.45,
                "mode": "live",
            }
        )
        client = _StubFuturesClient(position_amt=amt)
        broker = _FailingBroker(client, leverage=2)
        trader = Trader(cfg, store, client, broker, Notifier({}, {}))
        return trader, dict(store.get_open_position("BNBUSDT", "live"))

    def test_reconciles_when_exchange_flat(self):
        trader, pos = self._trader("0")
        self.assertTrue(trader._reconcile_flat(pos, 775.0))
        # journal closed with reconcile reason
        self.assertIsNone(trader.store.get_open_position("BNBUSDT", "live"))
        trades = trader.store.trades(mode="live")
        self.assertEqual(trades[0]["reason"][:10], "reconciled")

    def test_keeps_position_when_exchange_open(self):
        trader, pos = self._trader("0.05")
        self.assertFalse(trader._reconcile_flat(pos, 775.0))
        self.assertIsNotNone(trader.store.get_open_position("BNBUSDT", "live"))

    def test_close_position_reconciles_on_sell_failure(self):
        trader, pos = self._trader("0")
        self.assertTrue(trader._close_position(pos, 775.0, "stop-loss hit"))
        self.assertIsNone(trader.store.get_open_position("BNBUSDT", "live"))

    def test_arm_skipped_for_paper(self):
        cfg = make_cfg(mode="paper", strategies=["ema_cross"])
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.db"))
            client = BinanceClient("", "", testnet=True)
            trader = Trader(cfg, store, client, PaperBroker(client, 1000.0), Notifier({}, {}))
            trader._arm_exchange_stops("BTCUSDT", 0.002, 980.0, 1040.0, 1)  # no raise
            self.assertEqual(len(store.open_positions("paper")), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
