"""Evaluate tradable books on the S&P 500 universe with the daily simulator.

Book = market-cap weights tilted by a score. The score's ingredients and signs come from long-run evidence on large stocks,
not from this sample. The control (no tilt) is the fair benchmark for skill: it shares every data quirk of the tilted books.

Usage: python research/strategy_eval.py [comma-separated ml oos names in store/research/oos]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.strategy.simulator import load_market, run_book, tilt_weights  # noqa: E402
from fintailor.strategy.stats import block_bootstrap, newey_west_t, sharpe  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 100, "display.float_format", lambda v: f"{v:.2f}")
END, DEV_END, START = pd.Timestamp("2025-10-31"), pd.Timestamp("2023-12-31"), pd.Timestamp("2019-01-01")
SIGNALS = {"mom": ("mom_12_1", 1), "prof": ("gp_assets_nf", 1), "inv": ("asset_growth_nf", -1), "value": ("book_to_price_nf", 1), "value2": ("earn_yield_nf", 1)}


def connect():
    for _ in range(40):
        try:
            return duckdb.connect(str(ROOT / "store/fintailor.duckdb"), read_only=True)
        except duckdb.Error:
            time.sleep(3)
    raise RuntimeError("database busy")


def zscore(s: pd.Series, dates: pd.Series) -> pd.Series:
    g = s.groupby(dates.values)
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-12)).fillna(0.0)


def composite(p: pd.DataFrame) -> pd.Series:
    """Average percentile of momentum, profitability, low investment and value; value is the mean of book and earnings yield."""
    parts = {k: p[c] * sign for k, (c, sign) in SIGNALS.items()}
    x = pd.DataFrame(parts)
    x["value"] = x[["value", "value2"]].mean(axis=1)
    x = x[["mom", "prof", "inv", "value"]]
    raw = x.mean(axis=1).where(x.notna().sum(axis=1) >= 2)
    return zscore(raw, p["date"])


def sector_momentum(p: pd.DataFrame, con) -> pd.Series:
    """12-month sector return, skipping the latest month, standardised across sectors each day and given to each stock in the sector."""
    from fintailor.config import sector_etfs
    etfs = sector_etfs()
    names = "','".join(etfs)
    px = con.execute(f"select ticker, date, adj_close from prices_daily where ticker in ('{names}')").df()
    px["date"] = pd.to_datetime(px["date"])
    C = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    mom = C.shift(21) / C.shift(252) - 1
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1) + 1e-12, axis=0).rename(columns=etfs)
    long = z.stack().rename("sec").rename_axis(["date", "sector"]).reset_index()
    return p[["date", "sector"]].assign(sector=p["sector"].astype(str)).merge(long, on=["date", "sector"], how="left")["sec"].set_axis(p.index).fillna(0.0)


def ml_scores(p: pd.DataFrame, name: str) -> pd.Series:
    o = pd.read_parquet(ROOT / "store/research/oos" / f"{name}.parquet")
    o = o[o["horizon"].isin([5, 10])]
    g = o.groupby(["date", "horizon"])["score"]
    o["z"] = (o["score"] - g.transform("mean")) / (g.transform("std") + 1e-12)
    w = o.groupby(["date", "ticker"])["z"].mean().rename("ml").reset_index()
    return p[["date", "ticker"]].merge(w, on=["date", "ticker"], how="left")["ml"].set_axis(p.index)


def summarize(r: pd.DataFrame, base: pd.Series, spy: pd.Series) -> dict:
    a = (r["ret"] - base).dropna()
    x = (r["ret"] - spy.reindex(r.index)).dropna()
    bb = block_bootstrap(a)
    return {"vs_spy": x.mean() * 25200, "active": a.mean() * 25200, "te": a.std() * np.sqrt(252) * 100, "ir": sharpe(a),
            "t_nw": newey_west_t(a), "ci_lo": bb["lo"] * 100, "ci_hi": bb["hi"] * 100, "p_pos": bb["p_positive"] * 100,
            "dev": a[a.index <= DEV_END].mean() * 25200, "val": a[a.index > DEV_END].mean() * 25200, "turn": r["turnover"].mean() * 252}


def main() -> None:
    cols = ["date", "ticker", "mcap", "is_etf_flag", "dvol20", "sector"] + sorted({c for c, _ in SIGNALS.values()})
    p = pd.read_parquet(ROOT / "store/features/panel.parquet", columns=cols)
    p = p[(~p["is_etf_flag"].astype(bool)) & (p["date"] >= START) & (p["date"] <= END)].reset_index(drop=True)
    con = connect()
    m = load_market(con, "2018-09-01", str(END.date()))
    sec = sector_momentum(p, con)
    con.close()
    book = pd.DataFrame({"date": p["date"], "ticker": p["ticker"], "mcap": p["mcap"]})
    scores = {"composite": composite(p), "sector": sec}
    scores["comp+sector"] = zscore(scores["composite"] + scores["sector"], p["date"])
    for name in [x for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else []) if x]:
        scores[f"ml:{name}"] = zscore(ml_scores(p, name), p["date"])
        scores[f"all3:{name}"] = zscore(scores[f"ml:{name}"] + scores["composite"] + scores["sector"], p["date"])
        scores[f"ml+sector:{name}"] = zscore(scores[f"ml:{name}"] + scores["sector"], p["date"])
    dates = np.sort(book["date"].unique())
    by_date = {d: g.index.to_numpy() for d, g in book.groupby("date")}
    base_rows, tick = book["mcap"].to_numpy(float), book["ticker"].to_numpy()

    def book_for(z: np.ndarray | None, lam: float, offset: int) -> dict:
        out = {}
        for d in dates[offset::5]:
            ix = by_date[d]
            f = pd.DataFrame({"ticker": tick[ix], "mcap": base_rows[ix], "z": z[ix] if z is not None else 0.0})
            out[pd.Timestamp(d)] = tilt_weights(f, lam)
        return out

    controls = [run_book(m, book_for(None, 0.0, i)) for i in range(5)]
    ctrl = pd.concat(controls, axis=1)["ret"].mean(axis=1)
    x = (ctrl - m.spy.reindex(ctrl.index)).dropna()
    print(f"control (no tilt) vs SPY: {x.mean() * 25200:+.2f}%/yr, tracking {x.std() * np.sqrt(252) * 100:.2f}%, turnover {np.mean([c['turnover'].mean() for c in controls]) * 252:.2f}x")
    rows = {}
    for name, z in scores.items():
        for lam in (0.25, 0.5, 1.0):
            runs = [run_book(m, book_for(z.to_numpy(float), lam, i)) for i in range(5)]
            avg = pd.DataFrame({"ret": pd.concat([r["ret"] for r in runs], axis=1).mean(axis=1), "turnover": pd.concat([r["turnover"] for r in runs], axis=1).mean(axis=1)})
            s = summarize(avg, ctrl, m.spy)
            rows[f"{name} lam={lam}"] = s
            print(f"{name} lam={lam}: active {s['active']:+.2f} (dev {s['dev']:+.2f}, val {s['val']:+.2f}), TE {s['te']:.1f}, t {s['t_nw']:+.2f}, turn {s['turn']:.1f}", flush=True)
    print("\n", pd.DataFrame(rows).T.to_string())
    best = {k: pd.concat([run_book(m, book_for(v.to_numpy(float), 0.5, i)) for i in range(5)], axis=1)["ret"].mean(axis=1) - ctrl for k, v in scores.items()}
    print("\nactive return by year at lam=0.5 (%):\n", pd.DataFrame({k: v.groupby(v.index.year).mean() * 25200 for k, v in best.items()}).T.round(1).to_string())
    cross = pd.DataFrame(best).dropna()
    print("\ncorrelation of active returns:\n", cross.corr().round(2).to_string())


if __name__ == "__main__":
    main()
