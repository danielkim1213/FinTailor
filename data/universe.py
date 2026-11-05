"""S&P 500 membership history (survivorship-bias aware) + sector ETFs.

Source: Wikipedia current constituents + "Selected changes" table, walked backwards to rebuild
membership intervals. Free and decent, but not perfect (old changes are incomplete, recycled
tickers exist). For production, swap in a paid point-in-time constituent file.
"""
from __future__ import annotations

import io

import pandas as pd

from ..config import get_config, sector_etfs
from ..db import connect, replace_where
from ..utils import http_get, log

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

GICS_TO_KEY = {
    "Information Technology": "technology", "Financials": "financials", "Health Care": "health_care",
    "Consumer Discretionary": "consumer_discretionary", "Consumer Staples": "consumer_staples",
    "Energy": "energy", "Industrials": "industrials", "Materials": "materials",
    "Utilities": "utilities", "Real Estate": "real_estate", "Communication Services": "communication",
}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(dict.fromkeys(str(c) for c in col)).strip() for col in df.columns]
    return df


FJA_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"


def fetch_sp500_history(start_date: str) -> pd.DataFrame:
    """Primary: fja05680/sp500 membership intervals (maintained, includes removed members).
    Wikipedia's current table adds names/GICS sectors and catches changes after the file's last update."""
    html = http_get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 FinTailor research"}).text
    cur = _flatten(pd.read_html(io.StringIO(html))[0])
    cur["ticker"] = cur["Symbol"].astype(str).str.strip()
    try:
        iv = pd.read_csv(io.StringIO(http_get(FJA_URL).text), parse_dates=["start_date", "end_date"])
    except Exception as e:  # noqa: BLE001
        log.warning("fja05680 intervals unavailable (%s); falling back to Wikipedia change table", e)
        return _wiki_history(cur, html, start_date)
    last = max(iv["start_date"].max(), iv["end_date"].max())
    min_date = pd.Timestamp(start_date)
    iv = iv[iv["end_date"].isna() | (iv["end_date"] >= min_date)].copy()
    open_now = set(iv.loc[iv["end_date"].isna(), "ticker"])
    cur_set = set(cur["ticker"])
    iv.loc[iv["end_date"].isna() & ~iv["ticker"].isin(cur_set), "end_date"] = last  # removed after file update
    added = cur[~cur["ticker"].isin(open_now)]
    if len(added):
        iv = pd.concat([iv, pd.DataFrame({"ticker": added["ticker"],
                                          "start_date": pd.to_datetime(added["Date added"], errors="coerce").fillna(last),
                                          "end_date": pd.NaT})], ignore_index=True)
    meta = cur.set_index("ticker")
    iv["name"] = iv["ticker"].map(meta["Security"]).fillna(iv["ticker"])
    iv["sector"] = iv["ticker"].map(meta["GICS Sector"].map(GICS_TO_KEY)).fillna("unknown")
    iv["start_date"] = iv["start_date"].clip(lower=min_date).dt.date
    iv["end_date"] = iv["end_date"].dt.date
    iv["is_etf"] = False
    iv["source"] = "fja05680+wikipedia"
    log.info("membership file through %s; %d added / %d removed since", last.date(), len(added),
             len(open_now - cur_set))
    return iv[["ticker", "name", "sector", "start_date", "end_date", "is_etf", "source"]]


def _wiki_history(cur: pd.DataFrame, html: str, start_date: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    if len(tables) < 2 or not any("Added" in str(c) for c in _flatten(tables[1]).columns):
        raise RuntimeError("no membership history source available")
    chg = _flatten(tables[1])
    sym_col = next(c for c in cur.columns if "Symbol" in c)
    sec_col = next(c for c in cur.columns if "GICS Sector" in c)
    name_col = next(c for c in cur.columns if "Security" in c)
    date_col = next(c for c in chg.columns if "Date" in c)
    add_col = next(c for c in chg.columns if c.startswith("Added") and "Ticker" in c)
    rem_col = next(c for c in chg.columns if c.startswith("Removed") and "Ticker" in c)
    rem_name = next((c for c in chg.columns if c.startswith("Removed") and "Security" in c), None)

    chg = chg.assign(_d=pd.to_datetime(chg[date_col], errors="coerce")).dropna(subset=["_d"])
    chg = chg.sort_values("_d", ascending=False)
    min_date = pd.Timestamp(start_date)

    open_iv: dict[str, dict] = {}
    for _, r in cur.iterrows():
        t = str(r[sym_col]).strip()
        open_iv[t] = dict(ticker=t, name=r[name_col], sector=GICS_TO_KEY.get(r[sec_col], "unknown"), end_date=None)
    closed = []
    for _, r in chg.iterrows():
        d = r["_d"]
        if d < min_date:
            break
        added = str(r[add_col]).strip() if pd.notna(r[add_col]) else ""
        removed = str(r[rem_col]).strip() if pd.notna(r[rem_col]) else ""
        if added and added in open_iv:
            iv = open_iv.pop(added)
            iv["start_date"] = d
            closed.append(iv)
        if removed and removed not in open_iv:
            open_iv[removed] = dict(ticker=removed, name=r[rem_name] if rem_name else removed,
                                    sector="unknown", end_date=d - pd.Timedelta(days=1))
    for iv in open_iv.values():
        iv["start_date"] = min_date
        closed.append(iv)
    df = pd.DataFrame(closed)
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
    df["is_etf"] = False
    df["source"] = "wikipedia"
    return df[["ticker", "name", "sector", "start_date", "end_date", "is_etf", "source"]]


def etf_rows(start_date: str) -> pd.DataFrame:
    bench = get_config()["universe"]["benchmark"]
    rows = [dict(ticker=t, name=t, sector=s, start_date=pd.Timestamp(start_date).date(), end_date=None,
                 is_etf=True, source="config") for t, s in sector_etfs().items()]
    rows.append(dict(ticker=bench, name=bench, sector="benchmark", start_date=pd.Timestamp(start_date).date(),
                     end_date=None, is_etf=True, source="config"))
    return pd.DataFrame(rows)


def refresh_universe() -> pd.DataFrame:
    start = get_config()["universe"]["start_date"]
    stocks = fetch_sp500_history(start)
    df = pd.concat([stocks, etf_rows(start)], ignore_index=True)
    with connect() as con:
        replace_where(con, "universe", df, "source in ('wikipedia', 'fja05680+wikipedia', 'config')")
    log.info("universe: %d intervals (%d unique tickers, %d removed members)", len(df), df.ticker.nunique(),
             int(stocks.end_date.notna().sum()))
    return df


def load_universe(con) -> pd.DataFrame:
    df = con.execute("select * from universe").df()
    for c in ("start_date", "end_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def all_tickers(con) -> list[str]:
    return sorted(con.execute("select distinct ticker from universe").df()["ticker"].tolist())
