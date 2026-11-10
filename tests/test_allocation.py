import numpy as np
import pandas as pd

from fintailor.strategy.allocation import Settings, book_returns, current_allocation, exposure_target, scale, sector_weights


def _prices(n=900, seed=4):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    names = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLU"]
    drift = dict(zip(names, [0.0004, 0.0012, 0.0002, -0.0002, 0.0003, 0.0001]))
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(drift[t], 0.01, n))) for t in names}, index=dates)


def test_month_picks_ignore_the_prior_month_and_later():
    p = _prices()
    months = p.index.to_period("M").unique()
    m = months[20]
    before = sector_weights(p).loc[m]
    changed = p.copy()
    start_of_previous = months[19].start_time
    changed.loc[changed.index >= start_of_previous, ["XLK", "XLE"]] *= 3.0  # huge moves from the skipped month onward
    assert before.equals(sector_weights(changed).loc[m])


def test_book_return_on_a_day_does_not_use_that_days_or_later_prices_to_choose():
    p = _prices()
    full = book_returns(p)["book"]
    cut = p.index[700]
    shorter = book_returns(p[p.index <= cut])["book"]
    assert np.allclose(full[shorter.index].dropna(), shorter.dropna())


def test_exposure_respects_the_borrowing_limit_and_scales_down_in_turbulence():
    s = Settings()
    calm = pd.Series(np.tile([0.002, -0.002], 100))
    wild = pd.Series(np.tile([0.05, -0.05], 100))
    assert exposure_target(calm, s).dropna().eq(s.max_exposure).all()
    assert exposure_target(wild, s).dropna().lt(0.25).all()


def test_trend_gate_only_removes_borrowing():
    from fintailor.strategy.allocation import leverage_cap
    n = 600
    calm = pd.Series(np.tile([0.002, -0.002], n // 2))
    up = pd.Series(np.full(n, 0.001))                      # rising index: always above its average
    down = pd.Series(np.r_[np.full(300, 0.001), np.full(300, -0.004)])  # falls below the average in the second half
    s = Settings(trend_ma=200)
    assert leverage_cap(up, s).eq(s.max_exposure).all()
    cap = leverage_cap(down, s)
    assert cap.iloc[:200].eq(s.max_exposure).all()        # warm-up: no gate
    assert cap.iloc[-50:].eq(1.0).all()
    assert exposure_target(calm, s, down).iloc[-50:].eq(1.0).all()
    wild = pd.Series(np.tile([0.05, -0.05], n // 2))
    assert exposure_target(wild, s, down).dropna().lt(0.25).all()   # the gate never lifts the vol target
    assert leverage_cap(down, Settings(trend_ma=0)).eq(1.5).all()


def test_scale_charges_for_borrowing_and_keeps_cash_earning():
    dates = pd.bdate_range("2024-01-01", periods=60)
    book = pd.Series(np.tile([0.001, -0.001], 30), index=dates)
    rf = pd.Series(0.0001, index=dates)
    s = Settings(max_exposure=1.5, band=0.0, cost_exposure=0.0)
    r = scale(book, rf, s)
    assert r["exposure"].max() <= 1.5 + 1e-9
    last = r.iloc[-1]
    assert np.isclose(last["ret"], last["exposure"] * book.iloc[-1] + (1 - last["exposure"]) * 0.0001 - max(last["exposure"] - 1, 0) * s.borrow_spread / 252)


def test_current_allocation_holds_the_stated_shares():
    r = current_allocation(_prices(), Settings())
    total = sum(r["holdings"].values())
    assert abs(total + r["cash"] - 1) < 1e-9
    assert np.isclose(r["holdings"]["SPY"], 0.5 * r["exposure"])
    assert len(r["sectors"]) == 3 and "SPY" not in r["sectors"]
    assert r["exposure"] <= r["leverage_cap"] + 1e-9 and r["leverage_cap"] in (1.0, 1.5)


def test_scale_with_core_matches_exposure_target_gate():
    p = _prices()
    parts = book_returns(p).dropna(subset=["book"])
    rf = pd.Series(0.0001, index=parts.index)
    gated = scale(parts["book"], rf, Settings(), parts["core"])
    free = scale(parts["book"], rf, Settings(trend_ma=0), parts["core"])
    assert gated["exposure"].max() <= 1.5 + 1e-9
    assert (gated["exposure"] <= free["exposure"] + 1e-9).all()   # the gate can only lower exposure
