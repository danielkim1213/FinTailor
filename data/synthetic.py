"""Synthetic market for fast end-to-end tests (no network). Contains a weak, known signal
(short-term reversal + news sentiment) so tests can check the model learns something."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_config, sector_etfs
from ..db import connect, replace_where, upsert
from ..utils import ET, stable_id


def generate(n_stocks: int = 120, start: str = "2015-01-01", end: str = "2024-06-30", seed: int = 7,
             signal: float = 0.15) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    etfs = list(sector_etfs())
    S = len(etfs)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    sector_of = rng.integers(0, S, n_stocks)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    vol = rng.uniform(0.012, 0.03, n_stocks)

    mkt = rng.normal(0.0003, 0.01, T)
    sec = rng.normal(0, 0.006, (T, S))
    idio = rng.normal(0, 1, (T, n_stocks)) * vol
    r = mkt[:, None] * beta + sec[:, sector_of] + idio
    # inject predictability: tomorrow's idio return reverts part of the last-5-day idio move
    idio5 = pd.DataFrame(idio).rolling(5).sum().shift(1).fillna(0).values
    r = r - signal * idio5 / 5

    # news sentiment that leaks a little of the next-day idio return
    news_rows = []
    for t in range(1, T - 1):
        for j in rng.choice(n_stocks, size=3, replace=False):
            sent = np.tanh(idio[t + 1, j] / vol[j] * 0.5 + rng.normal(0, 1))
            ts = (dates[t] + pd.Timedelta(hours=rng.uniform(7, 20))).tz_localize(ET).tz_convert("UTC").tz_localize(None)
            word = "beats" if sent > 0 else "misses"
            news_rows.append(dict(id=f"syn_{t}_{j}", ticker=tickers[j], category="company",
                                  headline=f"{tickers[j]} {word} expectations, shares move", summary="",
                                  source="synthetic", url=None, published_at=ts, received_at=ts, backfilled=False,
                                  _sent=float(sent)))

    close = 50 * np.exp(np.cumsum(r, axis=0))
    on = rng.normal(0, 0.004, (T, n_stocks))
    openp = np.vstack([close[0], close[:-1]]) * np.exp(on)
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.006, (T, n_stocks))))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.006, (T, n_stocks))))
    volume = rng.lognormal(14, 0.5, (T, n_stocks))

    def frame(tk, c, o, h, lo, v):
        return pd.DataFrame({"ticker": tk, "date": dates.date, "open": o, "high": h, "low": lo, "close": c,
                             "adj_close": c, "raw_close": c, "volume": v, "vwap": np.nan, "source": "synthetic"})

    frames = [frame(tickers[j], close[:, j], openp[:, j], high[:, j], low[:, j], volume[:, j]) for j in range(n_stocks)]
    rs = np.column_stack([r[:, sector_of == s].mean(axis=1) if (sector_of == s).any() else mkt for s in range(S)])
    for s, e in enumerate(etfs):
        c = 30 * np.exp(np.cumsum(rs[:, s]))
        frames.append(frame(e, c, np.r_[c[0], c[:-1]], c * 1.004, c * 0.996, np.full(T, 5e7)))
    spy = 200 * np.exp(np.cumsum(r.mean(axis=1)))
    frames.append(frame(get_config()["universe"]["benchmark"], spy, np.r_[spy[0], spy[:-1]], spy * 1.003,
                        spy * 0.997, np.full(T, 1e8)))
    prices = pd.concat(frames, ignore_index=True)

    # membership: 80% full period, others join late or leave early (survivorship test)
    univ = []
    for j, t in enumerate(tickers):
        s0, e0 = dates[0], None
        u = rng.random()
        if u < 0.1:
            s0 = dates[rng.integers(T // 4, T // 2)]
        elif u < 0.2:
            e0 = dates[rng.integers(T // 2, T - 50)]
        univ.append(dict(ticker=t, name=t, sector=list(sector_etfs().values())[sector_of[j]], start_date=s0.date(),
                         end_date=e0.date() if e0 is not None else None, is_etf=False, source="synthetic"))
    for e, name in sector_etfs().items():
        univ.append(dict(ticker=e, name=e, sector=name, start_date=dates[0].date(), end_date=None, is_etf=True,
                         source="synthetic"))
    univ.append(dict(ticker=get_config()["universe"]["benchmark"], name="SPY", sector="benchmark",
                     start_date=dates[0].date(), end_date=None, is_etf=True, source="synthetic"))
    vix = pd.DataFrame({"series_id": "YF_VIX", "date": dates.date,
                        "value": 15 + 100 * pd.Series(mkt).rolling(20, min_periods=1).std().values,
                        "available_date": dates.date})
    earn = []
    for j, t in enumerate(tickers):
        off = int(rng.integers(0, 63))
        for k in range(off, T + 63, 63):  # quarterly; includes one scheduled date beyond the data
            d = dates[k] if k < T else dates[-1] + pd.offsets.BDay(k - T + 1)
            est = rng.normal(1, 0.3)
            earn.append(dict(ticker=t, date=d.date(), announced_at=(d + pd.Timedelta(hours=21)).to_pydatetime(),
                             eps_actual=est + rng.normal(0, 0.1) if k < T else None, eps_estimate=est,
                             source="synthetic"))
    return dict(prices=prices, universe=pd.DataFrame(univ), news=pd.DataFrame(news_rows), macro=vix,
                earnings=pd.DataFrame(earn))


def load_synthetic(**kw) -> dict[str, int]:
    data = generate(**kw)
    news = data["news"]
    scores = pd.DataFrame({"news_id": news["id"], "scorer": "synthetic",
                           "payload": "{}", "sentiment": news["_sent"], "severity": 2.0, "novelty": "new",
                           "event_type": "earnings", "created_at": news["received_at"]})
    with connect() as con:
        replace_where(con, "prices_daily", data["prices"], "source = 'synthetic'")
        replace_where(con, "universe", data["universe"], "source = 'synthetic'")
        upsert(con, "news", news.drop(columns="_sent"))
        upsert(con, "news_scores", scores)
        upsert(con, "macro", data["macro"])
        upsert(con, "earnings", data["earnings"])
    return {k: len(v) for k, v in data.items()}


__all__ = ["generate", "load_synthetic", "stable_id"]
