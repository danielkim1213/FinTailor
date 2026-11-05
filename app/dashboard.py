"""Internal test dashboard (function over form). Run: python -m fintailor dashboard
Set FINTAILOR_STORE to point at another store (e.g. store_demo)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fintailor.config import get_config, store_dir  # noqa: E402
from fintailor.db import connect  # noqa: E402

st.set_page_config(page_title="FinTailor (internal)", layout="wide")


@st.cache_data(ttl=30)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(sql, list(params)).df()


def status_tab():
    from fintailor.status import status
    s = status()
    c1, c2, c3, c4 = st.columns(4)
    ks = s["kill_switch"]
    c1.metric("Kill switch", "HALTED" if ks.get("halted") else "running")
    c2.metric("Champion", s["champion"] or "-")
    c3.metric("Last price date", str(s["last_price_date"]))
    c4.metric("Model versions", s["n_versions"])
    if ks.get("halted"):
        st.error(f"Halted: {ks.get('reasons')}")
        note = st.text_input("Resume note (what you checked)")
        if st.button("Resume signals") and note:
            from fintailor.loop.steps import resume
            resume(note)
            st.cache_data.clear()
            st.rerun()
    st.write("Row counts", s["counts"])
    runs = q("select run_id, step, status, started_at, finished_at, detail from run_log order by started_at desc limit 200")
    if len(runs):
        st.subheader("Recent runs")
        st.dataframe(runs.drop(columns="detail"), width="stretch", height=260)
        err = runs[runs.status == "error"]
        if len(err):
            with st.expander(f"{len(err)} errors"):
                for r in err.head(10).itertuples():
                    st.code(f"{r.step}: {json.loads(r.detail).get('error')}")
    rep = sorted(store_dir("reports").glob("*.md"))
    if rep:
        with st.expander(f"Latest report ({rep[-1].stem})"):
            st.markdown(rep[-1].read_text(encoding="utf-8"))


def predictions_tab():
    dates = q("select distinct run_date from predictions where official order by run_date desc limit 120")
    if dates.empty:
        st.info("No predictions yet (run `fintailor train` then `fintailor predict`).")
        return
    d = st.selectbox("Run date", dates["run_date"].tolist())
    p = q("select ticker, horizon, prob, pred_excess, score, rank_pct, drivers, model_version from predictions "
          "where run_date = ? and official", (d,))
    h = st.radio("Horizon", sorted(p["horizon"].unique()), horizontal=True)
    sub = p[p["horizon"] == h].sort_values("score", ascending=False)
    sub["drivers"] = sub["drivers"].map(lambda x: ", ".join(f"{f}{'+' if v > 0 else '-'}" for f, v in json.loads(x)[:3]) if x else "")
    c1, c2 = st.columns(2)
    c1.write("Top")
    c1.dataframe(sub.head(25), width="stretch")
    c2.write("Bottom")
    c2.dataframe(sub.tail(15), width="stretch")
    try:
        from fintailor.features.panel import load_panel
        pan = load_panel(["date", "ticker", "sector"])
        sec = pan[pan["date"] == pd.Timestamp(d)][["ticker", "sector"]]
        agg = sub.merge(sec, on="ticker").groupby("sector", observed=True)[["pred_excess", "prob"]].mean().sort_values("pred_excess")
        st.write("Sector view (mean prediction)")
        st.bar_chart(agg["pred_excess"])
    except Exception:  # noqa: BLE001
        pass


def live_tab():
    from fintailor.loop.steps import live_metrics
    h = st.selectbox("Horizon", get_config()["labels"]["horizons"], index=1)
    lb = st.slider("Lookback (matured days)", 20, 500, 60)
    m = live_metrics(lb, h)
    if not m.get("n_trades"):
        st.info("No matured official predictions yet (`fintailor simulate` creates a paper record).")
        return
    cols = st.columns(5)
    for c, k in zip(cols, ("net_per_trade", "hit_rate", "rank_ic", "sharpe", "max_drawdown")):
        c.metric(k, f"{m.get(k, float('nan')):.4f}")
    df = q("""select p.run_date, p.ticker, p.score, p.prob, o.realized_net from predictions p
              join outcomes o using(run_date, ticker, horizon) where p.official and p.horizon = ?
              order by p.run_date""", (h,))
    df["rank"] = df.groupby("run_date")["score"].rank(ascending=False)
    top = df[df["rank"] <= get_config()["backtest"]["top_k"]].groupby("run_date")["realized_net"].mean()
    st.line_chart((1 + top / h).cumprod().rename("top-K cumulative (approx)"))
    from fintailor.eval.metrics import calibration_table
    st.write("Calibration (live)")
    st.dataframe(calibration_table(df.rename(columns={"realized_net": "y"})), width="stretch")


def backtest_tab():
    files = sorted(store_dir("backtests").glob("*.json"), reverse=True)
    if not files:
        st.info("Run `fintailor backtest --fast`.")
    else:
        f = st.selectbox("Report", files, format_func=lambda p: p.stem)
        rep = json.loads(Path(f).read_text())
        from fintailor.eval.metrics import summary_table
        st.dataframe(summary_table({k: v for k, v in rep.items() if not k.startswith("_")}), width="stretch")
        for k, v in rep.items():
            if isinstance(v, dict) and "gate" in v:
                st.write(k, v["gate"])
                c1, c2 = st.columns(2)
                c1.write("by year (net/trade)")
                c1.bar_chart(pd.Series(v.get("by_year", {})))
                c2.write("IC by year")
                c2.bar_chart(pd.Series(v.get("ic_by_year", {})))
    st.divider()
    st.write("Run a fast backtest now (blocks this page until done)")
    cfgs = ["base"] + [c["name"] for c in get_config().get("challengers", [])]
    cname = st.selectbox("Config", cfgs)
    if st.button("Run fast backtest"):
        from fintailor.eval.backtest import run_backtest
        from fintailor.features.panel import load_panel
        with st.spinner("running..."):
            r = run_backtest(load_panel(), fast=True, config_name=cname)
        st.success(r.get("_files"))


def customers_tab():
    from fintailor.signals.profiles import COUNTRY_PACKS, CustomerProfile, Questionnaire, load_customers, save_customer
    custs = load_customers()
    st.dataframe(pd.DataFrame([{**c.model_dump(exclude={"questionnaire"})} for c in custs]), width="stretch")
    with st.expander("Add / edit customer"):
        with st.form("cust"):
            cid = st.text_input("customer_id")
            country = st.selectbox("country", list(COUNTRY_PACKS))
            age = st.number_input("age", 18, 110, 35)
            exp = st.selectbox("experience", ["none", "beginner", "intermediate", "advanced"], index=2)
            amt = st.number_input("investable amount (local currency)", 1.0, 1e12, 10000.0)
            ef = st.checkbox("has emergency fund", True)
            debt = st.checkbox("has high-interest debt", False)
            short = st.checkbox("allow short", False)
            lt = st.slider("loss tolerance", 1, 5, 3)
            hz = st.slider("horizon years", 0, 30, 5)
            if st.form_submit_button("Save") and cid:
                p = CustomerProfile(customer_id=cid, country=country, age=int(age), experience=exp, investable_amount=amt,
                                    has_emergency_fund=ef, has_high_interest_debt=debt, allow_short=short,
                                    questionnaire=Questionnaire(loss_tolerance=lt, horizon_years=hz))
                save_customer(p)
                st.success(f"saved, risk score {p.risk_score}")
                st.cache_data.clear()


def signals_tab():
    sig = q("select run_date, customer_id, status, ticker, direction, horizon, explanation, payload from signals "
            "order by run_date desc, customer_id limit 500")
    if sig.empty:
        st.info("No signals yet.")
        return
    d = st.selectbox("Date", sorted(sig["run_date"].unique(), reverse=True))
    s = sig[sig["run_date"] == d]
    st.write(s["status"].value_counts())
    def _s(x):
        return "" if x is None or (not isinstance(x, str) and pd.isna(x)) else (str(int(x)) if isinstance(x, float) else str(x))

    for r in s.itertuples():
        with st.expander(f"{r.customer_id} | {r.status} | {_s(r.ticker)} {_s(r.direction)} {_s(r.horizon)}"):
            st.write(r.explanation)
            st.json(json.loads(r.payload))
    from fintailor.signals.ledger import mark_positions
    with connect(read_only=True) as con:
        last = con.execute("select max(date) from prices_daily").fetchone()[0]
        sent = con.execute("select * from signals where status = 'sent'").df()
        if len(sent):
            pos = mark_positions(con, sent, pd.Timestamp(last))
            st.subheader("Paper ledger (all clients)")
            st.dataframe(pos.groupby(["customer_id", "state"]).agg(n=("ret", "size"), mean_ret=("ret", "mean"),
                                                                    pnl_usd=("pnl_usd", "sum")), width="stretch")


def news_tab():
    n = q("""select n.received_at, n.ticker, n.headline, s.scorer, s.event_type, s.sentiment, s.severity
             from news n left join news_scores s on s.news_id = n.id order by n.received_at desc limit 300""")
    st.dataframe(n, width="stretch", height=500)
    v = q("select * from sector_views order by date desc limit 60")
    if len(v):
        st.write("Sector views")
        st.dataframe(v, width="stretch")


def reviews_tab():
    r = q("select run_date, ticker, horizon, category, verified, payload from reviews order by created_at desc limit 300")
    if r.empty:
        st.info("No reviews yet.")
        return
    st.write(r["category"].value_counts())
    for x in r.head(40).itertuples():
        p = json.loads(x.payload)
        with st.expander(f"{x.run_date} {x.ticker} h={x.horizon} [{x.category}] verified={x.verified}"):
            st.write(p["review"]["summary"])
            st.json(p["checks"])
    fq = q("select * from finetune_queue order by created_at desc limit 100")
    st.write(f"Fine-tune queue: {len(fq)} misread cases")
    st.dataframe(fq, width="stretch")


def models_tab():
    from fintailor.models.registry import registry, rollback
    reg = registry()
    st.write("Champion:", reg.get("champion"))
    st.dataframe(pd.DataFrame(reg.get("versions", [])).drop(columns="metrics", errors="ignore"), width="stretch")
    st.dataframe(pd.DataFrame(reg.get("history", [])[-20:]), width="stretch")
    if st.button("Rollback to previous champion"):
        st.success(f"champion -> {rollback()}")
    v = reg.get("champion")
    if v:
        from fintailor.models.registry import ModelBundle
        b = ModelBundle.load(v)
        h = st.selectbox("Feature importance horizon", list(b.models))
        m = b.models[h]
        imp = pd.Series(m.clf.feature_importance("gain"), index=m.features).sort_values(ascending=False).head(30)
        st.bar_chart(imp)
        st.json(b.meta.get("train", {}))


TABS = {"Status": status_tab, "Predictions": predictions_tab, "Live record": live_tab, "Backtests": backtest_tab,
        "Signals": signals_tab, "Customers": customers_tab, "News": news_tab, "Reviews": reviews_tab,
        "Models": models_tab}
st.sidebar.caption(f"store: {get_config()['paths']['store']}")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
choice = st.sidebar.radio("Page", list(TABS))
try:
    TABS[choice]()
except Exception as e:  # noqa: BLE001
    st.exception(e)
