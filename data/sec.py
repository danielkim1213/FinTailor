"""SEC EDGAR: 8-K / Form 4 filings (event counts) and XBRL fundamentals (point-in-time by `filed`).

SEC requires a descriptive User-Agent with contact info: set SEC_USER_AGENT in .env,
e.g. "FinTailor research yourname@example.com".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import env
from ..db import connect, upsert
from ..utils import ET, RateLimiter, http_get, log

_lim = RateLimiter(500)  # SEC allows 10 req/s; stay below


def _hdr() -> dict:
    ua = env("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT not set (SEC requires contact info)")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def cik_map() -> dict[str, int]:
    js = http_get("https://www.sec.gov/files/company_tickers.json", headers=_hdr(), limiter=_lim).json()
    return {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in js.values()}


def historical_cik_map() -> dict[str, int]:
    """Ticker -> CIK for names no longer listed, which SEC's current ticker file omits. Built from the cached Form 3/4/5
    data sets (data.insider). If a symbol was reused, take the issuer with the most filings during its index membership."""
    import zipfile

    from ..config import store_dir
    frames = []
    for path in sorted(store_dir("raw", "insider").glob("*_form345.zip")):
        with zipfile.ZipFile(path) as z, z.open("SUBMISSION.tsv") as f:
            frames.append(pd.read_csv(f, sep="\t", usecols=["FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"], dtype=str))
    sub = pd.concat(frames, ignore_index=True).dropna(subset=["ISSUERCIK", "ISSUERTRADINGSYMBOL"])
    sub["ticker"] = sub["ISSUERTRADINGSYMBOL"].str.upper().str.strip().str.replace("-", ".", regex=False)
    sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    with connect(read_only=True) as con:
        spans = con.execute("""select ticker, min(start_date) lo, max(coalesce(end_date, current_date)) hi
                               from universe where not is_etf group by 1""").df()
    m = sub.merge(spans, on="ticker")
    year = pd.Timedelta(days=365)
    m = m[(m["filed"] >= pd.to_datetime(m["lo"]) - year) & (m["filed"] <= pd.to_datetime(m["hi"]) + year)]
    best = m.groupby(["ticker", "ISSUERCIK"]).size().rename("n").reset_index().sort_values("n").drop_duplicates("ticker", keep="last")
    return {t: int(c) for t, c in zip(best["ticker"], best["ISSUERCIK"])}


def _tickers(con) -> list[str]:
    return con.execute("select distinct ticker from universe where not is_etf").df()["ticker"].tolist()


def collect_filings(tickers: list[str] | None = None, forms=("8-K", "4")) -> int:
    hdr = _hdr()
    cmap = cik_map()
    with connect(read_only=True) as con:
        tickers = tickers or _tickers(con)
    rows = []
    for i, t in enumerate(tickers):
        cik = cmap.get(t)
        if not cik:
            continue
        try:
            js = http_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=hdr, limiter=_lim).json()
        except Exception as e:  # noqa: BLE001
            log.warning("sec %s: %s", t, e)
            continue
        rec = js.get("filings", {}).get("recent", {})
        df = pd.DataFrame({k: rec.get(k, []) for k in ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "items")})
        df = df[df["form"].isin(forms)]
        if df.empty:
            continue
        acc = pd.to_datetime(df["acceptanceDateTime"], errors="coerce", utc=True)
        rows.append(pd.DataFrame({
            "accession": df["accessionNumber"], "ticker": t, "cik": str(cik), "form": df["form"],
            "filing_date": pd.to_datetime(df["filingDate"]).dt.date,
            "accepted_at": acc.dt.tz_convert("UTC").dt.tz_localize(None), "items": df["items"]}))
        if i % 100 == 0:
            log.info("sec filings: %d/%d", i, len(tickers))
    if not rows:
        return 0
    with connect() as con:
        n = upsert(con, "filings", pd.concat(rows, ignore_index=True))
    log.info("sec filings: %d rows", n)
    return n


# A company can change tags over time, and banks use their own. Each quarter comes from the first tag that reports it.
REV_TAGS = ["RevenuesNetOfInterestExpense", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "RegulatedAndUnregulatedOperatingRevenue", "RealEstateRevenueNet",
            "InterestAndDividendIncomeOperating"]
NI_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]


def _quarterly(units: list[dict]) -> pd.DataFrame:
    """Quarterly (~90d) values with filed date; Q4 derived as FY - (Q1+Q2+Q3) when needed."""
    d = pd.DataFrame(units)
    if d.empty or "start" not in d:
        return pd.DataFrame()
    d = d[d["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])].copy()
    d["start"], d["end"], d["filed"] = (pd.to_datetime(d[c]) for c in ("start", "end", "filed"))
    d["days"] = (d["end"] - d["start"]).dt.days
    d = d.sort_values("filed").drop_duplicates(["start", "end"], keep="first")  # first report = PIT value
    q = d[(d["days"] > 75) & (d["days"] < 105)][["end", "val", "filed"]]
    fy = d[(d["days"] > 340) & (d["days"] < 380)]
    extra = []
    for _, r in fy.iterrows():
        if (q["end"] == r["end"]).any():
            continue
        inside = q[(q["end"] > r["start"]) & (q["end"] < r["end"])]
        if len(inside) == 3:
            extra.append(dict(end=r["end"], val=r["val"] - inside["val"].sum(), filed=r["filed"]))
    if extra:
        q = pd.concat([q, pd.DataFrame(extra)], ignore_index=True)
    return q.sort_values("end")


def _instant(units: list[dict]) -> pd.DataFrame:
    """Balance-sheet (point-in-time) values: end, val, filed; first report of each period end."""
    d = pd.DataFrame(units)
    if d.empty or "end" not in d or "filed" not in d:
        return pd.DataFrame()
    if "form" in d:
        d = d[d["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])]
    d = d.copy()
    d["end"], d["filed"] = pd.to_datetime(d["end"]), pd.to_datetime(d["filed"])
    return d.sort_values("filed").drop_duplicates("end", keep="first")[["end", "val", "filed"]].sort_values("end")


def _instant_asof(inst: pd.DataFrame, filed: pd.Series) -> pd.DataFrame:
    """For each filing date: latest known value and the value ~1 year before it (both known by then)."""
    out = []
    for f in filed:
        k = inst[inst["filed"] <= f]
        if k.empty:
            out.append((f, np.nan, np.nan))
            continue
        cur = k.iloc[-1] if k["end"].is_monotonic_increasing else k.loc[k["end"].idxmax()]
        prev = k[(k["end"] <= cur["end"] - pd.Timedelta(days=300)) & (k["end"] >= cur["end"] - pd.Timedelta(days=430))]
        out.append((f, cur["val"], prev["val"].iloc[-1] if len(prev) else np.nan))
    return pd.DataFrame(out, columns=["filed", "val", "prev"])


def _ttm_asof(q: pd.DataFrame) -> pd.DataFrame:
    """For each filing date, TTM and YoY using only quarters filed on/before it."""
    out = []
    for f in sorted(q["filed"].unique()):
        known = q[q["filed"] <= f].sort_values("end").drop_duplicates("end", keep="last")
        if len(known) < 4:
            continue
        last4 = known.tail(4)
        if (last4["end"].iloc[-1] - last4["end"].iloc[0]).days > 300:
            continue
        prev4 = known.iloc[-8:-4] if len(known) >= 8 else None
        ttm = last4["val"].sum()
        prev = prev4["val"].sum() if prev4 is not None and len(prev4) == 4 else np.nan
        out.append(dict(filed=f, period_end=last4["end"].iloc[-1], ttm=ttm, prev=prev))
    return pd.DataFrame(out)


def _by_priority(facts: dict, tags: list[str], unit: str) -> pd.DataFrame:
    out = pd.DataFrame()
    for tag in tags:
        q = _quarterly(facts.get(tag, {}).get("units", {}).get(unit, []))
        if not q.empty:
            out = q if out.empty else pd.concat([out, q[~q["end"].isin(out["end"])]], ignore_index=True)
    return out.sort_values("end") if len(out) else out


def _asof_recent(units: list[dict], filed: pd.Series, max_age: int = 460) -> np.ndarray:
    """Latest value known on each filing date, ignoring facts whose period ended long before it."""
    out = np.full(len(filed), np.nan)
    inst = _instant(units)
    if inst.empty:
        return out
    for i, f in enumerate(filed):
        k = inst[inst["filed"] <= f]
        if len(k):
            cur = k.loc[k["end"].idxmax()]
            if (f - cur["end"]).days <= max_age:
                out[i] = cur["val"]
    return out


def _share_count(facts: dict, dei: dict, filed: pd.Series) -> np.ndarray:
    """Total shares as of each filing. The cover-page count is the most current, but multi-class filers report it per class
    or for one class only, so it is used only when it agrees with the consolidated weighted average."""
    def get(src: dict, tag: str) -> np.ndarray:
        return _asof_recent(src.get(tag, {}).get("units", {}).get("shares", []), filed)

    cover = get(dei, "EntityCommonStockSharesOutstanding")
    total = get(facts, "WeightedAverageNumberOfDilutedSharesOutstanding")
    for tag in ("WeightedAverageNumberOfShareOutstandingBasicAndDiluted", "WeightedAverageNumberOfSharesOutstandingBasic",
                "CommonStockSharesOutstanding"):
        total = np.where(np.isnan(total), get(facts, tag), total)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = cover / total
    agrees = np.isnan(total) | ((ratio > 0.7) & (ratio < 1.4))
    return np.where(np.isfinite(cover) & agrees, cover, total)


def collect_fundamentals(tickers: list[str] | None = None, ciks: dict[str, int] | None = None) -> int:
    with connect(read_only=True) as con:
        tickers = tickers or _tickers(con)
    df = fetch_fundamentals(tickers, ciks if ciks is not None else cik_map())
    if df.empty:
        return 0
    with connect() as con:
        n = upsert(con, "fundamentals", df)
    log.info("fundamentals: %d rows", n)
    return n


def fetch_fundamentals(tickers: list[str], cmap: dict[str, int]) -> pd.DataFrame:
    hdr = _hdr()
    rows = []
    for i, t in enumerate(tickers):
        cik = cmap.get(t)
        if not cik:
            continue
        try:
            allfacts = http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=hdr,
                                limiter=_lim, timeout=60).json().get("facts", {})
            facts, dei = allfacts.get("us-gaap", {}), allfacts.get("dei", {})
        except Exception as e:  # noqa: BLE001
            log.warning("companyfacts %s: %s", t, e)
            continue
        series = {"rev": _by_priority(facts, REV_TAGS, "USD"), "ni": _by_priority(facts, NI_TAGS, "USD"),
                  "eps": _by_priority(facts, EPS_TAGS, "USD/shares")}
        gp = _quarterly(facts.get("GrossProfit", {}).get("units", {}).get("USD", []))
        if gp.empty and len(series["rev"]):  # many filers report cost of revenue instead of gross profit
            cogs = []
            for tag in ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"):
                cogs += facts.get(tag, {}).get("units", {}).get("USD", [])
            cq = _quarterly(cogs)
            if len(cq):
                gp = series["rev"].merge(cq, on="end", suffixes=("", "_c"))
                gp = gp.assign(val=gp["val"] - gp["val_c"], filed=gp[["filed", "filed_c"]].max(axis=1))[["end", "val", "filed"]]
        series["gp"] = gp
        ttm = {k: _ttm_asof(v) for k, v in series.items() if len(v)}
        first = next((k for k in ("rev", "ni") if k in ttm and len(ttm[k])), None)  # some filers have income but no usable revenue
        if first is None:
            continue
        base = ttm[first].rename(columns={"ttm": f"{first}_ttm", "prev": f"{first}_prev"})
        for k in ("rev", "ni", "eps", "gp"):
            if k != first and k in ttm and len(ttm[k]):
                base = pd.merge_asof(base.sort_values("filed"), ttm[k].sort_values("filed")[["filed", "ttm", "prev"]]
                                     .rename(columns={"ttm": f"{k}_ttm", "prev": f"{k}_prev"}), on="filed")
        f_dates = base["filed"]
        base["shares"] = _share_count(facts, dei, f_dates)
        for name, units in (("assets", facts.get("Assets", {}).get("units", {}).get("USD", [])),
                            ("equity", facts.get("StockholdersEquity", {}).get("units", {}).get("USD", []))):
            inst = _instant(units)
            if inst.empty:
                base[name] = np.nan
                if name == "assets":
                    base["assets_prev"] = np.nan
                continue
            a = _instant_asof(inst, f_dates)
            base[name] = a["val"].values
            if name == "assets":
                base["assets_prev"] = a["prev"].values
        base["ticker"] = t
        rows.append(base)
        if i % 100 == 0:
            log.info("fundamentals: %d/%d", i, len(tickers))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    for c in ("rev_ttm", "rev_prev", "ni_ttm", "eps_ttm", "eps_prev", "gp_ttm", "assets", "assets_prev", "equity", "shares"):
        if c not in df:
            df[c] = np.nan
    df["rev_ttm_yoy"] = df["rev_ttm"] / df["rev_prev"] - 1
    df["eps_ttm_yoy"] = (df["eps_ttm"] - df["eps_prev"]) / df["eps_prev"].abs()
    df["assets_yoy"] = df["assets"] / df["assets_prev"] - 1
    df["filed"] = pd.to_datetime(df["filed"]).dt.date
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df = df[["ticker", "filed", "period_end", "rev_ttm", "ni_ttm", "eps_ttm", "rev_ttm_yoy", "eps_ttm_yoy",
             "gp_ttm", "assets", "assets_yoy", "equity", "shares"]]
    return df.drop_duplicates(["ticker", "filed", "period_end"])


__all__ = ["collect_filings", "collect_fundamentals", "ET"]
