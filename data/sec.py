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


REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax"]


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


def collect_fundamentals(tickers: list[str] | None = None) -> int:
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
            facts = http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=hdr,
                             limiter=_lim, timeout=60).json().get("facts", {}).get("us-gaap", {})
        except Exception as e:  # noqa: BLE001
            log.warning("companyfacts %s: %s", t, e)
            continue
        series = {}
        rev_units = []
        for tag in REV_TAGS:
            rev_units += facts.get(tag, {}).get("units", {}).get("USD", [])
        series["rev"] = _quarterly(rev_units)
        series["ni"] = _quarterly(facts.get("NetIncomeLoss", {}).get("units", {}).get("USD", []))
        series["eps"] = _quarterly(facts.get("EarningsPerShareDiluted", {}).get("units", {}).get("USD/shares", []))
        ttm = {k: _ttm_asof(v) for k, v in series.items() if len(v)}
        if "rev" not in ttm or ttm["rev"].empty:
            continue
        base = ttm["rev"].rename(columns={"ttm": "rev_ttm", "prev": "rev_prev"})
        for k in ("ni", "eps"):
            if k in ttm and len(ttm[k]):
                base = pd.merge_asof(base.sort_values("filed"), ttm[k].sort_values("filed")[["filed", "ttm", "prev"]]
                                     .rename(columns={"ttm": f"{k}_ttm", "prev": f"{k}_prev"}), on="filed")
        base["ticker"] = t
        rows.append(base)
        if i % 100 == 0:
            log.info("fundamentals: %d/%d", i, len(tickers))
    if not rows:
        return 0
    df = pd.concat(rows, ignore_index=True)
    for c in ("ni_ttm", "eps_ttm", "eps_prev"):
        if c not in df:
            df[c] = np.nan
    df["rev_ttm_yoy"] = df["rev_ttm"] / df["rev_prev"] - 1
    df["eps_ttm_yoy"] = (df["eps_ttm"] - df["eps_prev"]) / df["eps_prev"].abs()
    df["filed"] = pd.to_datetime(df["filed"]).dt.date
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df = df[["ticker", "filed", "period_end", "rev_ttm", "ni_ttm", "eps_ttm", "rev_ttm_yoy", "eps_ttm_yoy"]]
    df = df.drop_duplicates(["ticker", "filed", "period_end"])
    with connect() as con:
        n = upsert(con, "fundamentals", df)
    log.info("fundamentals: %d rows", n)
    return n


__all__ = ["collect_filings", "collect_fundamentals", "ET"]
