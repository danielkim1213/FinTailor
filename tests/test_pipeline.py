import json

import numpy as np
import pandas as pd


def test_label_definition(panel):
    """y_h = (close[t+h] / open[t+1] - 1) - SPY same window - cost."""
    from fintailor.config import cost
    from fintailor.db import connect
    with connect(read_only=True) as con:
        px = con.execute("select ticker, date, open, adj_close from prices_daily").df()
    px["date"] = pd.to_datetime(px["date"])
    w_o = px.pivot(index="date", columns="ticker", values="open")
    w_c = px.pivot(index="date", columns="ticker", values="adj_close")
    row = panel[(panel["ticker"] == "S005") & panel["y_5"].notna()].iloc[100]
    i = w_c.index.get_loc(row["date"])
    stock = w_c["S005"].iloc[i + 5] / w_o["S005"].iloc[i + 1] - 1
    spy = w_c["SPY"].iloc[i + 5] / w_o["SPY"].iloc[i + 1] - 1
    assert abs((stock - spy - cost()) - row["y_5"]) < 1e-5


def test_no_lookahead_in_features(panel):
    """Changing prices after date X must not change any feature on or before X."""
    from fintailor.db import connect
    from fintailor.features.panel import build_panel
    cut = panel["date"].sort_values().unique()[-200]
    feats = panel.attrs["meta"]["features"]
    before = panel[panel["date"] <= cut].set_index(["date", "ticker"])[feats]
    with connect() as con:
        con.execute("update prices_daily set adj_close = adj_close * 1.5, close = close * 1.5, open = open * 0.7, "
                    "high = high * 1.6, low = low * 0.6, volume = volume * 3 where date > ?", [cut])
        con.execute("update news_scores set sentiment = -sentiment where news_id in "
                    "(select id from news where received_at > ? + interval 1 day)", [cut])
    try:
        p2 = build_panel(save=False)
        after = p2[p2["date"] <= cut].set_index(["date", "ticker"])[feats]
        a, b = before.sort_index(), after.reindex(before.sort_index().index)
        diff = (a - b).abs().max()
        # ranks among eligible names can shift only if eligibility changed; it must not here
        assert diff.max() < 1e-5, diff[diff > 1e-5]
    finally:
        with connect() as con:
            con.execute("update prices_daily set adj_close = adj_close / 1.5, close = close / 1.5, open = open / 0.7, "
                        "high = high / 1.6, low = low / 0.6, volume = volume / 3 where date > ?", [cut])
            con.execute("update news_scores set sentiment = -sentiment where news_id in "
                        "(select id from news where received_at > ? + interval 1 day)", [cut])


def test_survivorship_membership(panel):
    from fintailor.db import connect
    with connect(read_only=True) as con:
        u = con.execute("select ticker, start_date, end_date from universe where end_date is not null and not is_etf").df()
    assert len(u) > 0
    r = u.iloc[0]
    rows = panel[panel["ticker"] == r["ticker"]]
    assert rows["date"].max() <= pd.Timestamp(r["end_date"])


def test_train_window_purge(panel):
    from fintailor.config import get_config
    from fintailor.models.registry import train_window
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    end = dates[-100]
    for h in (1, 5, 30):
        tr = train_window(panel, end, h)
        last = tr["date"].max()
        gap = dates.get_loc(end) - dates.get_loc(last)
        assert gap > h + get_config()["labels"]["embargo_extra_days"]


def test_assign_trading_date():
    from fintailor.utils import assign_trading_date
    td = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06"])
    ts = pd.Series(pd.to_datetime(["2024-03-04 20:00", "2024-03-04 21:30", "2024-03-02 12:00", "2024-03-07 12:00"]))
    out = assign_trading_date(ts, td)  # UTC; 16:00 ET = 21:00 UTC in March (EST)
    assert out.iloc[0] == pd.Timestamp("2024-03-04")
    assert out.iloc[1] == pd.Timestamp("2024-03-05")
    assert out.iloc[2] == pd.Timestamp("2024-03-04")
    assert pd.isna(out.iloc[3])


def test_model_learns_injected_signal(panel):
    from fintailor.eval.backtest import walk_forward
    from fintailor.eval.metrics import evaluate
    oos = walk_forward(panel, [5], start_test="2020-06-01", retrain_every=126, stride=3)
    r = evaluate(oos, 5, 10, 0.001)
    assert r["rank_ic"] > 0.02


def test_guardrails():
    from fintailor.llm.guardrails import check
    payload = {"prob": 0.5834, "exp_excess_net": 0.0123, "stop_price": 181.25, "horizon": 5, "max_position_local": 1234567}
    assert check("Probability 58%, expected 1.23%, stop $181.25, hold 5 days, cap 1,234,567 KRW.", payload)["ok"]
    bad = check("Probability 64% and a 9.9% gain", payload)
    assert not bad["ok"] and "64" in bad["invented_numbers"]
    # KR disabled (English-first): assert not check("수익 보장!", payload, "ko")["ok"]
    assert not check("Use margin to boost it", payload)["ok"]


def test_template_is_guardrail_clean():
    from fintailor.llm.explainer import template
    from fintailor.llm.guardrails import check
    from fintailor.signals.profiles import COUNTRY_PACKS
    for lang in ("en",):  # "ko" disabled (English-first)
        sig = {"ticker": "XLE", "direction": "long", "horizon": 5, "prob": 0.6123, "exp_excess_net": 0.0081,
               "stop_price": 88.4, "max_position_local": 3500, "currency": "USD",
               "drivers": [["ret_5", -0.1], ["vix", 0.05]], "risks": ["overnight gap risk"],
               "disclaimer": COUNTRY_PACKS["US"]["disclaimer"][lang]}
        t = template(sig, lang)
        assert check(t, sig, lang)["ok"], check(t, sig, lang)


def test_profiles_and_tiers():
    from fintailor.signals.engine import build_for_customer, tier_for
    from fintailor.signals.profiles import SAMPLE_CUSTOMERS, CustomerProfile
    assert tier_for(3)[0] == "none" and tier_for(6)[0] == "mid" and tier_for(9)[0] == "high"
    # beginner without an emergency fund (was the KR sample "kr_new") -> education only
    newbie = CustomerProfile(customer_id="us_new", country="US", age=24, experience="beginner",
                             investable_amount=3500, has_emergency_fund=False, has_high_interest_debt=False)
    cand = pd.DataFrame()
    st = {"open_positions": 0, "open_tickers": [], "weekly_pnl_pct": 0.0}
    rows = build_for_customer(newbie, cand, pd.Timestamp("2024-01-02"), {"USD": 1, "CAD": 1.35}, st, "v", "d")
    assert rows[0]["status"] == "education"
    aggr = next(c for c in SAMPLE_CUSTOMERS if c.customer_id == "us_aggr")
    assert aggr.risk_score >= 8
    st_loss = {"open_positions": 0, "open_tickers": [], "weekly_pnl_pct": -0.05}
    rows = build_for_customer(aggr, cand, pd.Timestamp("2024-01-02"), {"USD": 1}, st_loss, "v", "d")
    assert rows[0]["status"] == "blocked_loss_limit"
    c = CustomerProfile(customer_id="x", country="CA", age=40, experience="intermediate", investable_amount=1000,
                        has_emergency_fund=True, has_high_interest_debt=False)
    assert c.language == "en" and 1 <= c.risk_score <= 10


def test_signal_sizing_and_filters():
    from fintailor.signals.engine import build_for_customer
    from fintailor.signals.profiles import SAMPLE_CUSTOMERS
    aggr = next(c for c in SAMPLE_CUSTOMERS if c.customer_id == "us_aggr")
    cand = pd.DataFrame([
        dict(ticker="AAA", horizon=5, prob=0.62, conf=0.62, pred_excess=0.01, direction="long", raw_close=100.0,
             atr_pct_raw=0.02, is_etf_flag=False, megacap=True, earn_in_window=False, earn_unknown=False,
             macro_events="", sector="technology", drivers=json.dumps([["ret_5", -0.2]])),
        dict(ticker="BBB", horizon=5, prob=0.65, conf=0.65, pred_excess=0.012, direction="long", raw_close=50.0,
             atr_pct_raw=0.02, is_etf_flag=False, megacap=False, earn_in_window=True, earn_unknown=False,
             macro_events="", sector="energy", drivers=None)])
    st = {"open_positions": 0, "open_tickers": [], "weekly_pnl_pct": 0.0}
    rows = build_for_customer(aggr, cand, pd.Timestamp("2024-01-02"), {"USD": 1}, st, "v", "d")
    assert [r["ticker"] for r in rows] == ["AAA"]  # BBB excluded: earnings in window
    p = json.loads(rows[0]["payload"])
    assert p["stop_pct"] == 0.04 and p["stop_price"] == 96.0
    assert p["max_position_usd"] <= 50000 * 0.2 + 1e-6
    assert np.isclose(p["max_position_usd"], min(50000 * 0.01 / 0.04, 50000 * 0.2))


def test_nightly_loop_end_to_end(panel):
    from fintailor.loop.nightly import run_nightly
    from fintailor.signals.profiles import SAMPLE_CUSTOMERS, save_customer
    for c in SAMPLE_CUSTOMERS:
        save_customer(c)
    rep = run_nightly(collect=False, challengers=False, asof=str(panel["date"].max().date()))
    bad = {k: v for k, v in rep["steps"].items() if v["status"] != "ok"}
    assert not bad, bad
    assert rep["signals"]


def test_neutralize_and_rank_target():
    from fintailor.models.lgbm import _neutralize, _rank_gauss
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"date": np.repeat(pd.to_datetime(["2024-01-02", "2024-01-03"]), n // 2),
                       "sector": rng.choice(["a", "b", "c"], n), "log_dvol": rng.random(n), "beta_60": rng.random(n)})
    df["y"] = 0.05 * (df["sector"] == "a") + 0.1 * df["log_dvol"] - 0.08 * df["beta_60"] + rng.normal(0, 0.01, n)
    r = _neutralize(df, "y", ["sector", "size", "beta"])
    for _, g in df.assign(r=r).groupby("date"):
        assert abs(np.corrcoef(g["r"], g["log_dvol"])[0, 1]) < 1e-6
        assert abs(g.groupby("sector")["r"].mean()).max() < 1e-9
    z = _rank_gauss(df["y"], df["date"])
    assert abs(np.nanmean(z)) < 1e-9 and 0.9 < np.nanstd(z) < 1.1


def test_portfolio_buffer_cost_and_tilt():
    from fintailor.eval.portfolio import simulate, simulate_tilt
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=60)
    tick = [f"T{i}" for i in range(200)]
    sc = pd.DataFrame([(d, t, rng.normal()) for d in dates for t in tick], columns=["date", "ticker", "score"])
    rets = sc[["date", "ticker"]].assign(r=0.001, vol=0.02, sector="s", mcap=1e9, dvol=1e7)
    tight = simulate(sc, rets, k=20, buffer=1.0, cost=0.001, sector_cap=None, weighting="equal")
    loose = simulate(sc, rets, k=20, buffer=4.0, cost=0.001, sector_cap=None, weighting="equal")
    assert loose["turnover"].iloc[1:].mean() < tight["turnover"].iloc[1:].mean()
    # every name returns +0.1%: net = 0.1% minus turnover * round-trip cost
    assert np.allclose(tight["ret"], 0.001 - tight["turnover"] * 0.001)
    t = simulate_tilt(sc, rets, strength=0.5)
    assert np.allclose(t["gross"], 0.001) and (t["n"] == 200).all()


def test_seed_ensemble_roundtrip(tmp_path):
    from fintailor.models.lgbm import AvgBooster, HorizonModel, fit_horizon
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=400)
    df = pd.DataFrame([(d, f"T{i}") for d in dates for i in range(40)], columns=["date", "ticker"])
    df["f1"], df["f2"] = rng.random(len(df)), rng.random(len(df))
    df["y_5"] = 0.02 * df["f1"] + rng.normal(0, 0.01, len(df))
    m = fit_horizon(df, ["f1", "f2"], 5, {"n_seeds": 3})
    assert isinstance(m.reg, AvgBooster) and len(m.reg.boosters) == 3
    a = m.predict(df.tail(40))
    m.save(tmp_path)
    b = HorizonModel.load(tmp_path, 5).predict(df.tail(40))
    assert np.allclose(a["pred_raw"], b["pred_raw"]) and np.allclose(a["prob"], b["prob"])

def test_days_to_earn_estimate_ignores_future_reports():
    from fintailor.features.panel import _days_to_earn_estimate
    dates = pd.bdate_range("2024-01-01", "2024-12-31")
    known = pd.to_datetime(["2024-02-01", "2024-05-01"]).values
    with_future = pd.to_datetime(["2024-02-01", "2024-05-01", "2024-07-15"]).values
    a, b = _days_to_earn_estimate(known, dates), _days_to_earn_estimate(with_future, dates)
    before = dates < "2024-07-15"
    assert np.allclose(a[before], b[before], equal_nan=True)
    assert np.isnan(a[dates < "2024-02-01"]).all()
    assert a[list(dates).index(pd.Timestamp("2024-05-01"))] == 90  # 90-day cycle, reported today


def test_fit_options_fixed_rounds_and_neutral_clf():
    from fintailor.models.lgbm import fit_horizon
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=400)
    df = pd.DataFrame([(d, f"T{i}") for d in dates for i in range(40)], columns=["date", "ticker"])
    df["f1"], df["f2"] = rng.random(len(df)), rng.random(len(df))
    df["log_dvol"], df["beta_60"] = rng.random(len(df)), rng.random(len(df))
    df["sector"] = "x"
    df["y_5"] = 0.02 * df["f1"] + rng.normal(0, 0.01, len(df))
    m = fit_horizon(df, ["f1", "f2"], 5, {"early_stop": False, "num_boost_round": 25})
    assert m.reg.num_trees() == 25 and m.clf.num_trees() == 25
    n = fit_horizon(df, ["f1", "f2"], 5, {"neutralize": ["size", "beta"], "clf_neutral": True, "target": "rank"})
    assert np.isfinite(n.predict(df.tail(40))["prob_raw"]).all()


def test_exclude_prefixes_are_opt_in():
    from fintailor.models.lgbm import select_features
    feats = ["a", "rel_guide"]
    assert select_features(feats, {}) == ["a"]
    assert select_features(feats, {"include_prefixes": ["rel_"]}) == feats


def test_ranking_score_mode_override():
    from fintailor.models.lgbm import ranking_score
    p = pd.DataFrame({"pred_raw": [1.0, 2.0, 3.0], "prob_raw": [0.3, 0.2, 0.1]})
    assert list(ranking_score(p, "reg")) == [1.0, 2.0, 3.0]
