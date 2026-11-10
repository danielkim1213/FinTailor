import numpy as np
import pandas as pd

from fintailor.strategy.simulator import Market, run_book, tilt_weights
from fintailor.strategy.stats import block_bootstrap, deflated_sharpe, newey_west_t, probabilistic_sharpe, probability_of_overfitting


def _market(close, opens=None):
    close = np.asarray(close, float)
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    tick = [f"T{i}" for i in range(close.shape[1])]
    spy = pd.Series(close[:, 0], index=dates).pct_change().fillna(0.0)
    return Market(dates, tick, close.copy() if opens is None else np.asarray(opens, float), close, spy)


def test_book_trades_at_next_open_not_the_decision_close():
    # T0 jumps on the decision day's close; a decision made that close must not earn that jump
    close = [[100, 100], [200, 100], [200, 100], [200, 100]]
    m = _market(close)
    r = run_book(m, {m.dates[1]: pd.Series({"T0": 1.0})}, cost=0.0, min_trade=0.0)
    assert abs(r["ret"].sum()) < 1e-12


def test_overnight_gap_goes_to_the_old_position():
    close = np.array([[100, 100], [100, 100], [110, 100], [110, 100]], float)
    opens = np.array([[100, 100], [100, 100], [110, 100], [110, 100]], float)
    m = _market(close, opens)
    # hold T1 from day 1, switch to T0 decided on day 1's close: trades at day 2's open, after the gap
    t = {m.dates[0]: pd.Series({"T1": 1.0}), m.dates[1]: pd.Series({"T0": 1.0})}
    assert abs(run_book(m, t, cost=0.0, min_trade=0.0)["ret"].sum()) < 1e-12


def test_costs_are_charged_on_traded_weight():
    close = np.full((4, 2), 100.0)
    m = _market(close)
    t = {m.dates[0]: pd.Series({"T0": 1.0}), m.dates[1]: pd.Series({"T1": 1.0})}
    r = run_book(m, t, cost=0.002, min_trade=0.0)
    # entry 100% (0.1%), then sell 100% + buy 100% (0.2%)
    assert abs(r["ret"].sum() + 0.003) < 1e-5


def test_delisted_position_keeps_last_value_until_rebalanced():
    close = np.array([[100, 100], [100, 100], [100, np.nan], [100, np.nan]], float)
    closes = pd.DataFrame(close).ffill().to_numpy()
    m = Market(pd.bdate_range("2024-01-01", periods=4), ["T0", "T1"], close, closes, pd.Series(0.0, index=pd.bdate_range("2024-01-01", periods=4)))
    r = run_book(m, {m.dates[0]: pd.Series({"T1": 1.0})}, cost=0.0, min_trade=0.0)
    assert abs(r["ret"].sum()) < 1e-12


def test_tilt_respects_limits_and_is_neutral_without_a_score():
    f = pd.DataFrame({"ticker": list("ABCD"), "mcap": [700.0, 200.0, 50.0, 50.0], "z": [3.0, -1.0, 0.0, 2.0]})
    base = f["mcap"] / f["mcap"].sum()
    w = tilt_weights(f, 1.0, max_mult=3.0, max_active=0.02).reindex(list("ABCD")).to_numpy()
    assert abs(w.sum() - 1) < 1e-9 and (w <= np.minimum(base * 3, base + 0.02) + 1e-3).all()
    flat = tilt_weights(f.assign(z=0.0), 1.0).reindex(list("ABCD")).to_numpy()
    assert np.allclose(flat, base)


def test_stats_on_known_signals():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 5000)
    assert abs(newey_west_t(noise)) < 3
    strong = noise + 0.002
    assert newey_west_t(strong) > 5
    assert block_bootstrap(strong)["p_positive"] > 0.99 and block_bootstrap(noise - 0.002)["p_positive"] < 0.01
    assert probabilistic_sharpe(strong) > 0.99
    assert deflated_sharpe(noise, 200, 0.05) < 0.2  # pure noise cannot survive 200 attempts


def test_overfitting_probability_separates_luck_from_skill():
    rng = np.random.default_rng(2)
    luck = pd.DataFrame(rng.normal(0, 0.01, (1000, 30)))
    assert probability_of_overfitting(luck) > 0.25
    skill = luck.copy()
    skill[0] += 0.01
    assert probability_of_overfitting(skill) < 0.05
