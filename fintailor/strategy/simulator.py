"""Daily portfolio simulator.

Positions drift with prices between rebalances. A decision made on one day's close trades at the next day's open, and each
day's return is split into overnight (held the old weights) and intraday (held the new ones). The result is a total return
that can be compared directly with SPY over the same days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Market:
    dates: pd.DatetimeIndex
    tickers: list[str]
    open: np.ndarray
    close: np.ndarray
    spy: pd.Series  # SPY open-to-open-free daily total return, close to close

    @property
    def index(self) -> dict[str, int]:
        return {t: j for j, t in enumerate(self.tickers)}


def load_market(con, start: str = "2016-01-01", end: str | None = None, benchmark: str = "SPY") -> Market:
    q = f"select ticker, date, open, close, adj_close from prices_daily where date >= '{start}'" + (f" and date <= '{end}'" if end else "")
    px = con.execute(q).df()
    px["date"] = pd.to_datetime(px["date"])
    fac = px["adj_close"] / px["close"]
    px["o"] = px["open"] * fac
    C = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    O = px.pivot_table(index="date", columns="ticker", values="o", aggfunc="last").reindex_like(C)
    days = C.index[C[benchmark].notna()]
    C, O = C.reindex(days), O.reindex(days)
    O = O.where(O > 0).fillna(C)  # a missing open trades at that day's close
    spy = C[benchmark].pct_change().fillna(0.0)
    return Market(days, list(C.columns), O.to_numpy(float), C.ffill().to_numpy(float), spy)


def run_book(m: Market, targets: dict[pd.Timestamp, pd.Series], cost: float | pd.Series = 0.001, min_trade: float = 0.0002) -> pd.DataFrame:
    """targets: decision date -> weights by ticker (sum to 1). cost: round-trip fraction, per name if a Series.
    A held name whose prices end stays at its last close until the next rebalance moves it to the new book."""
    idx = m.index
    T, N = len(m.dates), len(m.tickers)
    cost_v = np.full(N, float(cost)) if np.isscalar(cost) else np.array([float(cost.get(t, cost.median())) for t in m.tickers])
    pos = np.zeros(N)
    cash, nav = 1.0, 1.0
    started = False
    rows = np.empty((T, 3))
    date_pos = {d: i for i, d in enumerate(m.dates)}
    decisions = {date_pos[d] + 1: s for d, s in targets.items() if d in date_pos and date_pos[d] + 1 < T}
    for i in range(1, T):
        before = nav
        traded = 0.0
        if started:  # overnight gap on what was held
            with np.errstate(invalid="ignore", divide="ignore"):
                gap = m.open[i] / m.close[i - 1]
            pos = pos * np.where(np.isfinite(gap), gap, 1.0)
            nav = pos.sum() + cash
        s = decisions.get(i)
        if s is not None:
            want = np.zeros(N)
            ok = [(idx[t], w) for t, w in s.items() if t in idx and np.isfinite(m.open[i, idx[t]])]
            if ok:
                js, ws = np.array([a for a, _ in ok]), np.array([b for _, b in ok], float)
                want[js] = ws / ws.sum()
                delta = want * nav - pos
                delta[np.abs(delta) < min_trade * nav] = 0.0
                fee = float(np.abs(delta) @ cost_v) / 2
                pos = pos + delta
                cash = nav - fee - pos.sum()
                nav -= fee
                traded = float(np.abs(delta).sum()) / 2 / before
                started = True
        if started:  # the day's session on the current positions
            with np.errstate(invalid="ignore", divide="ignore"):
                day = m.close[i] / m.open[i]
            pos = pos * np.where(np.isfinite(day), day, 1.0)
            nav = pos.sum() + cash
        rows[i] = (nav / before - 1, traded, np.abs(pos).sum() / nav if nav else 0.0)
    out = pd.DataFrame(rows, index=m.dates, columns=["ret", "turnover", "invested"]).iloc[1:]
    first = next(iter(sorted(decisions)), None)
    return out.iloc[first - 1:] if first else out


def tilt_weights(frame: pd.DataFrame, strength: float, max_mult: float = 3.0, max_active: float = 0.02, cap: str = "mcap", score: str = "z") -> pd.Series:
    """Market-cap weights times exp(strength * score). No stock goes above max_mult times its cap weight, or more than
    max_active of the whole book above it, so a single forecast cannot dominate."""
    f = frame.dropna(subset=[cap])
    f = f[f[cap] > 0]
    base = f[cap] / f[cap].sum()
    z = f[score].fillna(0.0).clip(-3, 3) if score in f else 0.0
    limit = np.minimum(base * max_mult, base + max_active).to_numpy()
    w = (base * np.exp(strength * z)).to_numpy()
    w = w / w.sum()
    for _ in range(100):  # weight above a limit is handed to the stocks still under theirs, in proportion
        over = w > limit
        excess = (w[over] - limit[over]).sum()
        if excess <= 1e-12:
            break
        w[over] = limit[over]
        room = ~over
        if not room.any():
            break
        w[room] += excess * w[room] / w[room].sum()
    return pd.Series(w / w.sum(), index=f["ticker"].values)
