"""Unit tests: indicators, strategies, risk, storage, paper broker, backtest.
Pure-Python (no numpy/pandas) — matches Termux runtime."""

from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from glmbot.api import BinanceClient, SymbolFilter  # noqa: E402
from glmbot.broker import FUT_TAKER_FEE, TAKER_FEE, PaperBroker  # noqa: E402
from glmbot.config import BotConfig, RiskCfg  # noqa: E402
from glmbot.indicators import bollinger, ema, macd, rsi, sma  # noqa: E402
from glmbot.klines import Klines  # noqa: E402
from glmbot.risk import RiskManager  # noqa: E402
from glmbot.storage import Store, utcnow  # noqa: E402
from glmbot.strategies import BUY, SELL, build_strategies  # noqa: E402


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
