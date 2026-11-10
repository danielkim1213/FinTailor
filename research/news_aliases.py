"""Link the ticker tags on news articles to the panel's tickers.

Articles keep the symbol in use when they were written (FB in 2019, META now), and symbols get reused by other companies
(CTRA was Contura before it was Coterra). SEC Form 3/4/5 filings state which issuer used which symbol each quarter, so a tag on a
given date is resolved to an issuer first and to the panel ticker of that issuer second.

Writes store/research/news_aliases.parquet (symbol, quarter, ticker) and store/research/news_symbols.txt.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fintailor.data.sec import historical_cik_map  # noqa: E402

GOOD = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
QUARTERS = pd.period_range("2014Q1", "2026Q4", freq="Q")
REACH = 4  # a symbol's issuer is carried at most this many quarters past its nearest filing


def filings() -> pd.DataFrame:
    frames = []
    for path in sorted((ROOT / "store/raw/insider").glob("*_form345.zip")):
        with zipfile.ZipFile(path) as z, z.open("SUBMISSION.tsv") as f:
            frames.append(pd.read_csv(f, sep="\t", usecols=["FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"], dtype=str))
    s = pd.concat(frames, ignore_index=True).dropna()
    s["symbol"] = s["ISSUERTRADINGSYMBOL"].str.upper().str.strip().str.replace("-", ".", regex=False)
    s["filed"] = pd.to_datetime(s["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    s["cik"] = pd.to_numeric(s["ISSUERCIK"], errors="coerce")
    s = s.dropna(subset=["filed", "cik"])
    s = s[s["symbol"].str.match(GOOD)]
    return pd.DataFrame({"symbol": s["symbol"].to_numpy(), "cik": s["cik"].astype("int64").to_numpy(), "q": s["filed"].dt.to_period("Q").to_numpy()})


def main() -> None:
    panel = pd.read_parquet(ROOT / "store/features/panel.parquet", columns=["ticker", "is_etf_flag"]).drop_duplicates("ticker")
    tickers = set(panel["ticker"])
    cik_of = {t: c for t, c in historical_cik_map().items() if t in tickers}
    panel_of: dict[int, list[str]] = {}
    for t, c in cik_of.items():
        panel_of.setdefault(c, []).append(t)
    print(f"panel tickers {len(tickers)}, with an SEC issuer {len(cik_of)}", flush=True)

    f = filings()
    counts = f.groupby(["symbol", "q", "cik"]).size().rename("n").reset_index()
    counts = counts[counts["n"] >= 2].sort_values("n").drop_duplicates(["symbol", "q"], keep="last")  # the issuer behind each symbol, each quarter
    rows = []
    for sym, g in counts.groupby("symbol"):
        known = pd.Series(g["cik"].to_numpy(), index=pd.PeriodIndex(g["q"], freq="Q")).sort_index()
        full = known.reindex(QUARTERS)
        pos = pd.Series(range(len(QUARTERS)), index=QUARTERS)
        ahead = full.ffill(limit=REACH)
        behind = full.bfill(limit=REACH)
        issuer = full.fillna(ahead).fillna(behind)
        for q, c in issuer.dropna().items():
            for t in panel_of.get(int(c), []):
                rows.append((sym, str(q), t))
        if sym in tickers:  # no filings near this date: nothing says the symbol meant someone else, so keep the plain match
            for q in issuer[issuer.isna()].index:
                rows.append((sym, str(q), sym))
        del pos
    seen = set(counts["symbol"])
    for t in sorted(tickers - seen):  # e.g. ETFs, which file no Forms 3/4/5
        rows.extend((t, str(q), t) for q in QUARTERS)
    out = pd.DataFrame(rows, columns=["symbol", "quarter", "ticker"]).drop_duplicates()
    out.to_parquet(ROOT / "store/research/news_aliases.parquet", index=False)

    symbols = sorted(set(out["symbol"]) | tickers)
    (ROOT / "store/research/news_symbols.txt").write_text(",".join(symbols), encoding="utf-8")
    other = out[out["symbol"] != out["ticker"]]
    print(f"symbols to fetch {len(symbols)}; symbols that differ from the panel ticker {other['symbol'].nunique()}; panel tickers reached that way {other['ticker'].nunique()}")
    spans = other.groupby(["ticker", "symbol"])["quarter"].agg(["min", "max", "size"]).reset_index()
    for t in ["META", "ELV", "CTRA", "WTW", "RVTY", "GOOGL", "FI", "RTX", "BKNG", "CVS", "HII", "PSKY", "XYZ"]:
        print(t, spans[spans["ticker"] == t][["symbol", "min", "max"]].to_dict("records"))
    own = out[out["symbol"] == out["ticker"]].groupby("ticker")["quarter"].agg(["min", "max", "size"])
    print("own-symbol quarters for reused names:\n", own.loc[[t for t in ["CTRA", "META", "ELV", "FB"] if t in own.index]].to_string())
    no_issuer = sorted(tickers - set(cik_of))
    print(f"panel tickers without an SEC issuer ({len(no_issuer)}):", no_issuer[:60])


if __name__ == "__main__":
    main()
