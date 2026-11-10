"""Point-in-time daily panel.

Row = (date t, ticker). Features use only information available at the close of t
(news by *received* time, macro by *available* date, filings by acceptance time).
Labels: enter at the open of t+1, exit at the close of t+h, minus SPY over the same window,
minus round-trip cost.

All heavy lifting is done on wide (date x ticker) matrices, then stacked once -> fast.
Cross-sectional features are ranked per day among eligible names (centered to [-0.5, 0.5]).
Market-level features (same for every ticker on a date) are left raw on purpose: ranking them
would make them constant.
"""
from __future__ import annotations

import json
import time
import zlib

import duckdb
import numpy as np
import pandas as pd

from ..config import cost, get_config, horizons, sector_etfs, store_dir
from ..db import connect
from ..utils import assign_trading_date, log

AUX = ["raw_close", "atr_pct_raw", "dvol20", "sector", "is_etf_flag", "days_to_earn_raw"]
EARN_FUND = ["last_surprise", "days_since_earn", "ret_since_earn", "days_to_earn", "earn_yield", "net_margin",
             "rev_yoy", "eps_yoy", "gp_assets", "asset_growth", "book_to_price"]


def label_cols(hs=None) -> list[str]:
    hs = hs or horizons()
    return [f"{p}_{h}" for h in hs for p in ("y", "g", "gap")]


# ----------------------------------------------------------------------------- helpers
def _asof(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    s = series[~series.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def _rank(W: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return W.where(mask).rank(axis=1, pct=True) - 0.5


def _take(etf_vals: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.take_along_axis(etf_vals, np.clip(idx, 0, None), axis=1)
    out[idx < 0] = np.nan
    return out


def _days_to_next(event_dates: pd.DatetimeIndex, dates: pd.DatetimeIndex, cap: int = 60) -> np.ndarray:
    ev = np.sort(pd.DatetimeIndex(event_dates).values)
    if len(ev) == 0:
        return np.full(len(dates), np.nan)
    i = np.searchsorted(ev, dates.values, side="left")
    out = np.full(len(dates), np.nan)
    ok = i < len(ev)
    out[ok] = (ev[i[ok]] - dates.values[ok]) / np.timedelta64(1, "D")
    return np.minimum(out, cap)


def _days_to_earn_estimate(event_dates, dates: pd.DatetimeIndex, cap: int = 90, stale_after: int = 150) -> np.ndarray:
    """Point-in-time guess of days to the next report: the last report already known plus the median of its
    recent gaps. The actual next date is not used, since historically it was not known in advance."""
    ev = np.unique(np.asarray(event_dates, dtype="datetime64[ns]"))
    out = np.full(len(dates), np.nan)
    if len(ev) == 0:
        return out
    gaps = pd.Series(np.diff(ev) / np.timedelta64(1, "D"), dtype=float)
    typical = np.r_[91.0, gaps.rolling(8, min_periods=1).median().clip(60, 120).to_numpy()]
    i = np.searchsorted(ev, dates.values, side="right") - 1
    ok = np.flatnonzero(i >= 0)
    elapsed = (dates.values[ok] - ev[i[ok]]) / np.timedelta64(1, "D")
    est = np.clip(typical[i[ok]] - elapsed, 0, cap)
    out[ok] = np.where(elapsed > stale_after, np.nan, est)
    return out


def _event_wide(df: pd.DataFrame, value: str, dates, tickers, agg="sum") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(np.nan, index=dates, columns=tickers)
    w = df.pivot_table(index="tdate", columns="ticker", values=value, aggfunc=agg)
    return w.reindex(index=dates, columns=tickers)


# ----------------------------------------------------------------------------- main
def build_panel(con=None, save: bool = True) -> pd.DataFrame:
    if con is None:
        with connect(read_only=True) as c:
            return build_panel(c, save)
    t0 = time.time()
    cfg = get_config()
    ucfg = cfg["universe"]
    bench = ucfg["benchmark"]
    etf_map = sector_etfs()
    H = sorted({*horizons(), *get_config()["labels"].get("research_horizons", [])})  # extras: labels only, never trained live
    c_ = cost()

    px = con.execute("select ticker, date, open, high, low, close, adj_close, raw_close, volume, vwap "
                     "from prices_daily").df()
    px["date"] = pd.to_datetime(px["date"])
    big = px.set_index(["date", "ticker"]).unstack("ticker")
    dates = pd.DatetimeIndex(big.index[big[("adj_close", bench)].notna()]) if ("adj_close", bench) in big else big.index
    big = big.reindex(dates)
    univ = con.execute("select * from universe").df()
    for c in ("start_date", "end_date"):
        univ[c] = pd.to_datetime(univ[c])
    tickers = sorted(set(px["ticker"]) & set(univ["ticker"]) - {bench})
    del px

    def W(col):
        return big[col].reindex(columns=tickers).astype("float64")

    fac = W("adj_close") / W("close")
    C = W("adj_close")
    O, Hh, L = W("open") * fac, W("high") * fac, W("low") * fac
    V, RC, CS = W("volume"), W("raw_close").fillna(W("close")), W("close")
    VW = (W("vwap") * fac).fillna((Hh + L + C) / 3)
    spyC = big[("adj_close", bench)]
    spyO = big[("open", bench)] * (big[("adj_close", bench)] / big[("close", bench)])
    del big

    # ---- eligibility (point-in-time membership + liquidity)
    T, N = len(dates), len(tickers)
    member = np.zeros((T, N), dtype=bool)
    col_of = {t: j for j, t in enumerate(tickers)}
    for r in univ.itertuples():
        j = col_of.get(r.ticker)
        if j is None or r.sector == "benchmark":
            continue
        s = np.searchsorted(dates.values, np.datetime64(r.start_date), side="left")
        e = T if pd.isna(r.end_date) else np.searchsorted(dates.values, np.datetime64(r.end_date), side="right")
        member[s:e, j] = True
    dvol20 = (CS * V).rolling(20, min_periods=10).mean()
    is_etf = np.array([t in etf_map for t in tickers])
    elig = pd.DataFrame(member, index=dates, columns=tickers) & C.notna() & (RC >= ucfg["min_price"]) & \
        ((dvol20 >= ucfg["min_dollar_volume"]) | is_etf[None, :])

    # ---- price / volume features
    r1 = C.pct_change(fill_method=None)
    F: dict[str, pd.DataFrame] = {}
    for k in (1, 3, 5, 10, 20):
        F[f"ret_{k}"] = C / C.shift(k) - 1
    F["mom_12_1"] = C.shift(21) / C.shift(252) - 1
    for k in (20, 50, 200):
        F[f"ma{k}_pos"] = C / C.rolling(k, min_periods=int(k * 0.75)).mean() - 1
    F["gap"] = O / C.shift(1) - 1
    F["vol_surge"] = V / V.rolling(20, min_periods=10).mean().shift(1)
    rng_ = (Hh - L)
    F["close_loc"] = ((C - L) / rng_).where(rng_ > 0)
    F["vwap_dev"] = C / VW - 1
    F["rv_20"] = r1.rolling(20, min_periods=15).std()
    F["rv_ratio"] = r1.rolling(5, min_periods=4).std() / F["rv_20"]
    tr = np.fmax(np.fmax(rng_, (Hh - C.shift(1)).abs()), (L - C.shift(1)).abs())
    atr = tr.rolling(14, min_periods=10).mean() / C
    F["atr_14"] = atr
    F["dist_52w_high"] = C / C.rolling(252, min_periods=126).max() - 1
    F["max_ret_20"] = r1.rolling(20, min_periods=15).max()
    F["log_dvol"] = np.log1p(dvol20)
    spy_r = spyC.pct_change(fill_method=None)
    m_s, v_s = spy_r.rolling(60, min_periods=40).mean(), spy_r.rolling(60, min_periods=40).var()
    cov = r1.mul(spy_r, axis=0).rolling(60, min_periods=40).mean() - r1.rolling(60, min_periods=40).mean().mul(m_s, axis=0)
    beta = cov.div(v_s, axis=0)
    F["beta_60"] = beta
    F["idio_vol_60"] = (r1 - beta.mul(spy_r, axis=0)).rolling(60, min_periods=40).std()

    # ---- point-in-time sector assignment: ETF with highest rolling corr of SPY-excess returns
    etf_list = [e for e in etf_map if e in tickers]
    sec_keys = [etf_map[e] for e in etf_list]
    X = r1.sub(spy_r, axis=0)
    win = ucfg["sector_corr_window"]
    mx, sx = X.rolling(win, min_periods=60).mean(), X.rolling(win, min_periods=60).std()
    best = np.full((T, N), -np.inf)
    sidx = np.full((T, N), -1, dtype=np.int16)
    for j, e in enumerate(etf_list):
        y = X[e]
        my, sy = y.rolling(win, min_periods=60).mean(), y.rolling(win, min_periods=60).std()
        corr = (X.mul(y, axis=0).rolling(win, min_periods=60).mean() - mx.mul(my, axis=0)).div(sx.mul(sy, axis=0)).values
        upd = corr > best
        best[upd] = corr[upd]
        sidx[upd] = j
    for j, e in enumerate(etf_list):
        sidx[:, col_of[e]] = j
    # A published GICS sector beats the estimate above, which puts Apple in consumer staples in some years. The estimate
    # remains for companies that left the index, whose GICS sector the source no longer lists.
    sector_pos = {k: j for j, k in enumerate(sec_keys)}
    for r in univ.itertuples():
        if r.sector in sector_pos and r.ticker in col_of and r.ticker not in etf_map:
            sidx[:, col_of[r.ticker]] = sector_pos[r.sector]
    for k in (5, 20):
        etf_ret = F[f"ret_{k}"][etf_list].values
        sec_ret = _take(etf_ret, sidx)
        F[f"vs_sector_{k}"] = F[f"ret_{k}"] - sec_ret
        spy_k = (spyC / spyC.shift(k) - 1).values[:, None]
        F[f"sector_vs_spy_{k}"] = pd.DataFrame(sec_ret - spy_k, index=dates, columns=tickers)

    RAW: dict[str, pd.DataFrame] = {"is_etf": pd.DataFrame(np.broadcast_to(is_etf.astype(float), (T, N)).copy(),
                                                           index=dates, columns=tickers)}
    news_cut = cfg["news"]["cutoff_et"]

    # ---- news (ticker-level), sector news + LLM sector view
    news = con.execute("""
        select n.ticker, n.received_at, s.sentiment, s.severity, s.novelty, s.payload, s.event_type,
               try_cast(json_extract_string(s.payload, '$.confidence') as double) as confidence
        from news n join (select *, row_number() over (partition by news_id order by
            case scorer when 'llm' then 0 when 'synthetic' then 1 else 2 end) rn from news_scores) s
          on s.news_id = n.id and s.rn = 1""").df()
    if len(news):
        news["tdate"] = assign_trading_date(news["received_at"], dates, news_cut)
        news = news.dropna(subset=["tdate"])
        first_news = news["tdate"].min()
        tk = news[news["ticker"].isin(tickers)].assign(
            one=1.0, sw=lambda d: d.sentiment * d.severity.fillna(2) / 5, new=lambda d: (d.novelty == "new").astype(float))
        cnt = _event_wide(tk, "one", dates, tickers).fillna(0)
        ssum = _event_wide(tk, "sentiment", dates, tickers).fillna(0)
        swsum = _event_wide(tk, "sw", dates, tickers).fillna(0)
        nsum = _event_wide(tk, "new", dates, tickers).fillna(0)
        pre = dates < first_news
        c5, s5 = cnt.rolling(5, min_periods=1).sum(), ssum.rolling(5, min_periods=1).sum()
        RAW["news_cnt_1d"] = np.log1p(cnt)
        RAW["news_cnt_5d"] = np.log1p(c5)
        RAW["news_sent_1d"] = (ssum / cnt).where(cnt > 0)
        RAW["news_sent_5d"] = (s5 / c5).where(c5 > 0)
        RAW["news_sevsent_5d"] = swsum.rolling(5, min_periods=1).sum()
        RAW["news_novel_5d"] = (nsum.rolling(5, min_periods=1).sum() / c5).where(c5 > 0)
        for k in ("news_cnt_1d", "news_cnt_5d", "news_sevsent_5d"):
            RAW[k].loc[pre] = np.nan

        # abnormal attention: coverage vs the ticker's own past 60 days (raw counts mostly say "megacap")
        base = cnt.shift(5).rolling(60, min_periods=20).mean()
        warm = dates < dates[min(int(np.searchsorted(dates, first_news)) + 25, len(dates) - 1)]
        RAW["news_attn_1d"] = np.log((cnt + 1) / (base + 1))
        RAW["news_attn_5d"] = np.log((c5 + 1) / (base * 5 + 1))
        # longer window: post-news drift plays out over weeks (we predict up to 30 days)
        c20, s20 = cnt.rolling(20, min_periods=1).sum(), ssum.rolling(20, min_periods=1).sum()
        RAW["news_sent_20d"] = (s20 / c20).where(c20 > 0)
        RAW["news_sent_chg"] = RAW["news_sent_5d"] - RAW["news_sent_20d"]  # tone shift
        # confidence-weighted tone (the scorer's own certainty; lexicon = constant 0.3)
        tk["conf"] = tk["confidence"].fillna(0.5).clip(0, 1)
        tk["sc"] = tk["sentiment"] * tk["conf"]
        csum = _event_wide(tk, "conf", dates, tickers).fillna(0).rolling(5, min_periods=1).sum()
        scsum = _event_wide(tk, "sc", dates, tickers).fillna(0).rolling(5, min_periods=1).sum()
        RAW["news_csent_5d"] = (scsum / csum).where(csum > 0)
        # event types: the scorer classifies them, so use them ("guidance cut" != "negative op-ed").
        # Sums over 5 days: 0 = no such news (informative), sign/size = direction and weight.
        et = tk["event_type"].fillna("other")
        for name, kinds, val in (("news_earn_sent_5d", ("earnings", "guidance"), "sentiment"),
                                 ("news_rating_sent_5d", ("analyst_rating",), "sentiment"),
                                 ("news_legal_5d", ("legal_regulatory",), "one"),
                                 ("news_ma_5d", ("m_and_a",), "one")):
            sub = tk[et.isin(kinds)]
            w = _event_wide(sub, val, dates, tickers).fillna(0).rolling(5, min_periods=1).sum()
            RAW[name] = np.log1p(w) if val == "one" else w
        for k in ("news_attn_1d", "news_attn_5d"):
            RAW[k].loc[warm] = np.nan
        for k in ("news_earn_sent_5d", "news_rating_sent_5d", "news_legal_5d", "news_ma_5d"):
            RAW[k].loc[pre] = np.nan
        # sector impact from macro / world news payloads
        gen = news[news["payload"].fillna("").str.contains("sector_impact")]
        if len(gen):
            recs = []
            for d, p, sev in zip(gen["tdate"], gen["payload"], gen["severity"].fillna(2)):
                try:
                    imp = json.loads(p).get("sector_impact") or {}
                except (ValueError, TypeError):
                    continue
                for sk, v in imp.items():
                    sign = {"+": 1, "-": -1}.get(v, 0) if isinstance(v, str) else float(np.sign(v))
                    if sk in sec_keys and sign:
                        recs.append((d, sk, sign * sev / 5))
            if recs:
                sd = pd.DataFrame(recs, columns=["tdate", "sector", "v"]).pivot_table(
                    index="tdate", columns="sector", values="v", aggfunc="sum")
                sd = sd.reindex(index=dates, columns=sec_keys).fillna(0).rolling(5, min_periods=1).sum()
                sd.loc[dates < first_news] = np.nan
                RAW["sector_news_5d"] = pd.DataFrame(_take(sd.values, sidx), index=dates, columns=tickers)
    views = con.execute("select date, sector, avg(score) score from sector_views group by 1, 2").df()
    if len(views):
        views["date"] = pd.to_datetime(views["date"])
        vw = views.pivot(index="date", columns="sector", values="score").reindex(columns=sec_keys)
        vw = vw.reindex(vw.index.union(dates)).ffill(limit=3).reindex(dates)
        RAW["llm_sector_view"] = pd.DataFrame(_take(vw.values, sidx), index=dates, columns=tickers)

    # ---- earnings
    earn = con.execute("select ticker, date, announced_at, eps_actual, eps_estimate from earnings").df()
    days_to_earn = pd.DataFrame(np.nan, index=dates, columns=tickers)
    # yfinance has no history for most delisted names: there, the 10-Q/10-K filing date is a slightly late report date
    # that was public at the time, so the timing features exist for them too.
    reported = set(earn.loc[earn["eps_actual"].notna(), "ticker"])
    filed_events = con.execute("select distinct ticker, filed as date from fundamentals").df()
    filed_events = filed_events[~filed_events["ticker"].isin(reported)]
    if len(filed_events):
        earn = pd.concat([earn, filed_events.assign(announced_at=pd.NaT, eps_actual=np.nan, eps_estimate=np.nan)], ignore_index=True)
    if len(earn):
        earn["date"] = pd.to_datetime(earn["date"])
        earn["announced_at"] = pd.to_datetime(earn["announced_at"]).fillna(earn["date"] + pd.Timedelta(hours=21))
        earn["tdate"] = assign_trading_date(earn["announced_at"], dates, news_cut)
        earn = earn[earn["ticker"].isin(tickers)]
        dte = np.full((T, N), np.nan)
        for t, g in earn.groupby("ticker"):
            dte[:, col_of[t]] = _days_to_next(pd.DatetimeIndex(g["date"]), dates + pd.Timedelta(days=1), 90) + 1
        days_to_earn = pd.DataFrame(dte, index=dates, columns=tickers)
        known = earn.dropna(subset=["tdate"])
        known = known.assign(surp=((known.eps_actual - known.eps_estimate) / known.eps_estimate.abs().clip(lower=0.01)).clip(-1, 1),
                             one=1.0)
        flag = _event_wide(known, "one", dates, tickers).notna()
        RAW["last_surprise"] = _event_wide(known, "surp", dates, tickers, "mean").ffill(limit=90)
        last_d = pd.DataFrame(np.where(flag, dates.values[:, None], np.datetime64("NaT")), index=dates, columns=tickers).ffill(limit=90)
        RAW["days_since_earn"] = pd.DataFrame(
            (dates.values[:, None] - last_d.values.astype("datetime64[ns]")) / np.timedelta64(1, "D"),
            index=dates, columns=tickers)
        F["ret_since_earn"] = C / C.where(flag).ffill(limit=90) - 1
        est_days = np.full((T, N), np.nan)
        for t, g in known.groupby("ticker"):
            est_days[:, col_of[t]] = _days_to_earn_estimate(g["tdate"].values, dates)
        RAW["days_to_earn"] = pd.DataFrame(est_days, index=dates, columns=tickers)  # the model feature; days_to_earn_raw

    # ---- filings (8-K, Form 4)
    fil = con.execute("select ticker, form, accepted_at, filing_date from filings").df()
    if len(fil):
        fil["accepted_at"] = pd.to_datetime(fil["accepted_at"]).fillna(pd.to_datetime(fil["filing_date"]) + pd.Timedelta(hours=22))
        fil["tdate"] = assign_trading_date(fil["accepted_at"], dates, news_cut)
        fil = fil.dropna(subset=["tdate"]).assign(one=1.0)
        start_f = fil["tdate"].min()
        for form, name, win_ in (("8-K", "n_8k_5d", 5), ("4", "n_form4_20d", 20)):
            w = _event_wide(fil[fil.form == form], "one", dates, tickers).fillna(0).rolling(win_, min_periods=1).sum()
            w.loc[dates < start_f] = np.nan
            RAW[name] = w

    # ---- fundamentals (as of filed date + 1 business day)
    MCAP = None
    SHARES = None
    fun = con.execute("select ticker, filed, rev_ttm, ni_ttm, eps_ttm, rev_ttm_yoy, eps_ttm_yoy, gp_ttm, assets, "
                      "assets_yoy, equity, shares from fundamentals").df()
    if len(fun):
        fun["tdate"] = pd.to_datetime(fun["filed"]) + pd.offsets.BDay(1)
        fun = fun[fun["ticker"].isin(tickers)].copy()
        # A share count or EPS filed before a stock split no longer matches later prices. raw / split-adjusted close is the
        # splits still to come, so multiplying shares by it on the filing date puts them in the split-adjusted close's units.
        split_f = (RC / CS).where(lambda x: x > 0).ffill().bfill().fillna(1.0)
        at_filing = split_f.values[np.clip(dates.searchsorted(fun["tdate"].values, side="right") - 1, 0, T - 1),
                                   fun["ticker"].map(col_of).to_numpy()]
        fun["shares_adj"] = fun["shares"].where(fun["shares"] > 0) * at_filing
        fun["eps_adj"] = fun["eps_ttm"] / at_filing

        def fw(col):
            w = fun.pivot_table(index="tdate", columns="ticker", values=col, aggfunc="last")
            return w.reindex(w.index.union(dates)).ffill(limit=400).reindex(index=dates, columns=tickers)
        F["net_margin"] = fw("ni_ttm") / fw("rev_ttm")
        F["rev_yoy"] = fw("rev_ttm_yoy")
        F["eps_yoy"] = fw("eps_ttm_yoy").clip(-5, 5)
        # classic cross-sectional factors (Novy-Marx gross profitability, asset growth, book-to-price)
        assets = fw("assets").where(lambda a: a > 0)
        F["gp_assets"] = fw("gp_ttm") / assets
        F["asset_growth"] = fw("assets_yoy").clip(-0.9, 5)
        adj_shares = fw("shares_adj")
        SHARES = adj_shares / split_f
        dv = dvol20.where(dvol20 > 0)
        known = (adj_shares * CS).where(lambda x: x > 0)
        # Some large multi-class filers publish no usable total share count. Those names, and counts wildly out of line with
        # trading volume (unit errors), get cap = volume x the day's typical cap/volume ratio among the 200 most-traded stocks.
        ratio = known / dv
        typical = ratio.where(dv.rank(axis=1, ascending=False) <= 200).median(axis=1)
        sane = ratio.gt(typical / 15, axis=0) & ratio.lt(typical * 15, axis=0)
        company_cap = known.where(sane).fillna(dv.mul(typical, axis=0))
        F["earn_yield"] = (fw("eps_adj") / CS).fillna(fw("ni_ttm") / company_cap)
        F["book_to_price"] = (fw("equity") / company_cap).clip(-5, 5)
        # Share classes of one company (GOOG/GOOGL) carry the same filing and share count; without this each class would be
        # weighted as the whole company. The company's cap is divided between its classes by trading volume.
        same = fun.dropna(subset=["shares"]).groupby(["filed", "shares"])["ticker"].agg(lambda s: tuple(sorted(set(s))))
        class_groups = [g for g, n in same[same.map(len) > 1].value_counts().items() if n >= 3]
        class_share = pd.DataFrame(1.0, index=dates, columns=tickers)
        for g in class_groups:
            cols = list(g)
            class_share[cols] = dv[cols].div(dv[cols].sum(axis=1, min_count=1), axis=0).fillna(1.0 / len(cols))
        MCAP = company_cap * class_share

    def _table(sql: str) -> pd.DataFrame:  # optional tables (absent in demo / fresh stores)
        try:
            return con.execute(sql).df()
        except duckdb.Error:
            return pd.DataFrame()

    # ---- insider open-market trades (Form 4, P/S), usable the trading day after the filing date
    ins = _table("select ticker, filing_date, code, value, officer, director from insider_trades")
    if len(ins):
        ins = ins[ins["ticker"].isin(tickers) & (ins["officer"] | ins["director"])]
        ins["tdate"] = assign_trading_date(pd.to_datetime(ins["filing_date"]) + pd.Timedelta(hours=22), dates, news_cut)
        ins = ins.dropna(subset=["tdate"]).assign(one=1.0, v=lambda d: d["value"].fillna(0).clip(0, 5e9))  # data sets contain typos up to $1e16
        start_i = ins["tdate"].min()
        buy, sell = ins[ins["code"] == "P"], ins[ins["code"] == "S"]
        bval = _event_wide(buy, "v", dates, tickers).fillna(0).rolling(63, min_periods=1).sum()
        sval = _event_wide(sell, "v", dates, tickers).fillna(0).rolling(63, min_periods=1).sum()
        bcnt = _event_wide(buy, "one", dates, tickers).fillna(0).rolling(63, min_periods=1).sum()
        month_dvol = (dvol20 * 20).where(dvol20 > 0)
        RAW["ins_buy_63"] = np.log1p(1e4 * bval / month_dvol)          # bp of a month's $ volume
        RAW["ins_nbuy_63"] = np.log1p(bcnt)                              # clustered buying
        RAW["ins_net_63"] = np.sign(bval - sval) * np.log1p(1e4 * (bval - sval).abs() / month_dvol)
        for k in ("ins_buy_63", "ins_nbuy_63", "ins_net_63"):
            RAW[k].loc[dates < start_i + pd.Timedelta(days=90)] = np.nan

    # ---- short interest (FINRA), usable from settlement + 10 business days
    si = _table("select ticker, settlement_date, short_qty, days_to_cover from short_interest")
    if len(si):
        si = si[si["ticker"].isin(tickers)].sort_values(["ticker", "settlement_date"])
        si["avail"] = pd.to_datetime(si["settlement_date"]) + pd.offsets.BDay(10)
        si["chg"] = np.log((si["short_qty"] + 1) / (si.groupby("ticker")["short_qty"].shift(1) + 1))

        def sw(col):
            w = si.pivot_table(index="avail", columns="ticker", values=col, aggfunc="last")
            return w.reindex(w.index.union(dates)).ffill(limit=30).reindex(index=dates, columns=tickers)
        qty = sw("short_qty")
        F["si_dtc"] = sw("days_to_cover").clip(0, 60)
        F["si_chg"] = sw("chg").clip(-3, 3)
        if SHARES is not None:
            F["si_pct_shares"] = (qty / SHARES.where(SHARES > 0)).clip(0, 1)

    # ---- LLM reading of 8-K earnings releases, usable from the SEC acceptance time; held until the next quarter
    rel = _table("""select r.ticker, r.accepted_at, s.g, s.t, s.s, s.x from earnings_releases r
                    join release_scores s using(accession) where s.model = 'anthropic/claude-haiku-5.5'""")
    if len(rel):
        rel = rel[rel["ticker"].isin(tickers)].copy()
        rel["tdate"] = assign_trading_date(pd.to_datetime(rel["accepted_at"]), dates, news_cut)
        rel = rel.dropna(subset=["tdate"]).assign(gd=lambda d: d["g"].map({"up": 1.0, "down": -1.0}).fillna(0.0))
        for name, col in (("rel_guide", "gd"), ("rel_tone", "t"), ("rel_result", "s"), ("rel_oneoff", "x")):
            RAW[name] = _event_wide(rel, col, dates, tickers, "mean").ffill(limit=63)

    # ---- market-level features
    M: dict[str, pd.Series] = {}
    for k in (5, 20, 60):
        M[f"spy_ret_{k}"] = spyC / spyC.shift(k) - 1
    M["spy_ma200_pos"] = spyC / spyC.rolling(200, min_periods=150).mean() - 1
    M["spy_rv_20"] = spy_r.rolling(20, min_periods=15).std()
    stock_elig = elig & ~is_etf[None, :]
    M["breadth_50"] = (F["ma50_pos"] > 0).where(stock_elig).mean(axis=1)
    M["xsec_disp_5"] = F["ret_5"].where(stock_elig).std(axis=1)
    macro = con.execute("select series_id, date, value, available_date from macro").df()
    if len(macro):
        macro["date"] = pd.to_datetime(macro["date"])
        macro["available_date"] = pd.to_datetime(macro["available_date"])

        def ser(sid, transform=None):
            m = macro[macro.series_id == sid].sort_values("date")
            if m.empty:
                return None
            v = m.set_index("date")["value"]
            if transform:
                v = transform(v)
            return _asof(pd.Series(v.values, index=m["available_date"].values), dates)

        vix = ser("YF_VIX")
        if vix is None:
            vix = ser("VIXCLS")
        if vix is not None:
            M["vix"] = vix
            M["vix_chg_5"] = vix / vix.shift(5) - 1
            M["vix_bucket"] = pd.cut(vix, [0, 15, 20, 30, 1e9], labels=False).astype(float)
            v3 = ser("YF_VIX3M")
            if v3 is not None:
                M["vix_term"] = vix / v3
        def first(*sids):
            for sid in sids:
                s = ser(sid)
                if s is not None:
                    return s
            return None

        # yfinance series first (known at the close, same source in history and live), FRED as fallback
        tnx, irx = first("YF_TNX", "DGS10"), ser("YF_IRX")
        curve = (tnx - irx) if (tnx is not None and irx is not None) else ser("T10Y2Y")
        for name, s in (("dgs10", tnx), ("curve", curve), ("dff", ser("DFF")), ("hy_spread", ser("BAMLH0A0HYM2"))):
            if s is not None:
                M[name] = s
                M[f"{name}_chg_20"] = s - s.shift(20)
        hyg, ief = ser("YF_HYG"), ser("YF_IEF")
        if hyg is not None and ief is not None:
            cr = np.log(hyg / ief)
            M["credit_hyg_ief_chg_20"] = cr - cr.shift(20)
        for name, s in (("usd_chg_20", first("YF_DXYNYB", "DTWEXBGS")), ("oil_chg_20", first("YF_CLF", "DCOILWTICO"))):
            if s is not None:
                M[name] = (s / s.shift(20) - 1).clip(-0.6, 0.6)
        s = ser("CPIAUCSL", lambda v: v.pct_change(12, fill_method=None))
        if s is not None:
            M["cpi_yoy"] = s
            M["cpi_mom_vs_trend"] = ser("CPIAUCSL", lambda v: v.pct_change(fill_method=None) -
                                        v.pct_change(fill_method=None).rolling(12).mean())
        s = ser("UNRATE", lambda v: v.diff(3))
        if s is not None:
            M["unrate_chg_3m"] = s
        s = ser("PAYEMS", lambda v: v.diff() - v.diff().rolling(12).mean())
        if s is not None:
            M["payems_vs_trend"] = s
        for sid, name in (("EPU_DAILY", "epu"), ("GPR_DAILY", "gpr")):
            s = ser(sid, lambda v: v.rolling(5, min_periods=1).mean())
            if s is not None:
                M[name] = s
                M[f"{name}_z"] = (s - s.rolling(252, min_periods=60).mean()) / s.rolling(252, min_periods=60).std()
    world = con.execute("select metric, date, value from world_daily").df()
    if len(world):
        world["date"] = pd.to_datetime(world["date"]) + pd.offsets.BDay(1)  # known next day
        for metric, g in world.groupby("metric"):
            M[f"gdelt_{metric}"] = _asof(g.set_index("date")["value"], dates).where(dates >= g["date"].min())
    ev = con.execute("select date, event from events").df()
    if len(ev):
        ev["date"] = pd.to_datetime(ev["date"])
        for name, g in ev.groupby("event"):
            dd = _days_to_next(pd.DatetimeIndex(g["date"]), dates)
            dd[dates < g["date"].min()] = np.nan
            M[f"days_to_{name.lower()}"] = pd.Series(dd, index=dates)

    # ---- labels
    ent = O.shift(-1)
    spy_ent = spyO.shift(-1)
    logg = np.log(O / C.shift(1))
    LBL: dict[str, pd.DataFrame] = {}
    for h in H:
        ex = C.ffill(limit=h).shift(-h)  # delisted inside the window -> exit at last close
        gross = (ex / ent - 1).sub(spyC.shift(-h) / spy_ent - 1, axis=0)
        LBL[f"g_{h}"] = gross
        LBL[f"y_{h}"] = gross - c_
        LBL[f"gap_{h}"] = (logg.rolling(h - 1).sum().shift(-h) if h > 1 else logg * 0)

    # ---- assemble long panel
    E = elig.values
    rows = np.flatnonzero(E.ravel())
    di, ti = rows // N, rows % N
    out = {"date": dates.values[di], "ticker": np.asarray(tickers, dtype=object)[ti]}
    for name, w in F.items():
        out[name] = _rank(w.reindex(index=dates, columns=tickers), elig).values.ravel()[rows].astype(np.float32)
    for name, w in RAW.items():
        out[name] = pd.DataFrame(w).reindex(index=dates, columns=tickers).values.ravel()[rows].astype(np.float32)
    for name, s in M.items():
        out[name] = pd.Series(s).reindex(dates).values[di].astype(np.float32)
    out["raw_close"] = RC.values.ravel()[rows]
    out["atr_pct_raw"] = atr.values.ravel()[rows].astype(np.float32)
    out["dvol20"] = dvol20.values.ravel()[rows]
    out["mcap"] = (MCAP.reindex(index=dates, columns=tickers).values.ravel()[rows] if MCAP is not None
                   else np.full(len(rows), np.nan))
    sec_arr = np.asarray(sec_keys + ["unknown"], dtype=object)
    s_flat = sidx.ravel()[rows]
    out["sector"] = sec_arr[np.where(s_flat < 0, len(sec_keys), s_flat)]
    out["is_etf_flag"] = is_etf[ti]
    out["days_to_earn_raw"] = days_to_earn.values.ravel()[rows].astype(np.float32)
    for name, w in LBL.items():
        out[name] = w.values.ravel()[rows].astype(np.float32)
    panel = pd.DataFrame(out)
    panel["sector"] = panel["sector"].astype("category")

    # Earnings/fundamentals coverage is far thinner for names later removed from the index, so a NaN would tell the
    # model which stocks eventually drop out. A constant fill would leave a spike a tree can isolate, and fresh random
    # draws each day would jitter scores and churn the book, so each ticker/feature keeps one pseudo-random percentile
    # and takes that percentile of the day's real values.
    # Whether real financials were public that day. The all-stocks universe uses it to skip funds and blank-check shells.
    panel["fund_known"] = panel["net_margin"].notna() | panel["rev_yoy"].notna() if "net_margin" in panel else False
    for c in ("gp_assets", "asset_growth", "book_to_price", "earn_yield", "net_margin"):  # unfilled copies for factor scores
        if c in panel:
            panel[f"{c}_nf"] = panel[c]
    fill = [c for c in EARN_FUND if c in panel]
    codes, uniq = pd.factorize(panel["ticker"])
    pct = {c: np.array([zlib.crc32(f"{t}|{c}".encode()) / 2**32 for t in uniq])[codes] for c in fill}
    vals = {c: panel[c].to_numpy(copy=True) for c in fill}
    for idx in panel.groupby("date", observed=True).indices.values():
        for c in fill:
            v = vals[c]
            gaps, seen = idx[np.isnan(v[idx])], idx[~np.isnan(v[idx])]
            if len(gaps) and len(seen):
                v[gaps] = np.quantile(v[seen], pct[c][gaps], method="inverted_cdf")
    for c in fill:
        panel[c] = vals[c]

    feats = list(F) + list(RAW) + list(M)
    meta = {"features": feats, "ranked": list(F), "raw": list(RAW), "market": list(M),
            "labels": list(LBL), "built_at": pd.Timestamp.now().isoformat(),
            "n_rows": len(panel), "n_tickers": int(panel["ticker"].nunique()),
            "date_min": str(dates.min().date()), "date_max": str(dates.max().date())}
    if save:
        d = store_dir("features")
        panel.to_parquet(d / "panel.parquet", index=False)
        (d / "panel_meta.json").write_text(json.dumps(meta, indent=1))
    log.info("panel: %d rows x %d features, %d tickers, %s..%s (%.1fs)", len(panel), len(feats),
             meta["n_tickers"], meta["date_min"], meta["date_max"], time.time() - t0)
    panel.attrs["meta"] = meta
    return panel


def load_panel(columns: list[str] | None = None) -> pd.DataFrame:
    d = store_dir("features")
    meta = json.loads((d / "panel_meta.json").read_text())
    p = pd.read_parquet(d / "panel.parquet", columns=columns)
    p["date"] = pd.to_datetime(p["date"])
    p.attrs["meta"] = meta
    return p


def panel_meta() -> dict:
    return json.loads((store_dir("features") / "panel_meta.json").read_text())
