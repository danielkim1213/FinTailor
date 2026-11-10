"""Allocation program: what to hold and how much of it.

The book is half S&P 500 (SPY) and half the three strongest sector ETFs by 12-month return, skipping the latest month and reset
at each month start. The whole book is then scaled so its recent volatility sits near 17%, the market's own long-run level,
up to a borrowing limit. These settings were fixed from century-long evidence and published practice, before the test period.

Borrowing is allowed only while the market trades above its 200-day average (trend_ma). Below it the book is capped at 1.0x but
never cut below the volatility target because of the trend: the gate only decides whether leverage is permitted. This was one of
ten variants written down before testing (research/alloc_v2.py) and the only one that improved Sharpe and drawdown on the
century data, stayed ahead in all five eras, and was also better on real ETF prices since 2000.

Why this and not individual-stock forecasts: every stock-level forecast we tested on corrected data showed no reliable edge over
SPY, while sector momentum and volatility scaling each held up over about 100 years and across parameter choices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
CORE = "SPY"


@dataclass(frozen=True)
class Settings:
    top: int = 3                   # sectors held
    look_months: int = 12          # momentum window
    skip_months: int = 1           # most recent month left out
    sector_share: float = 0.5      # share of the book in the sector sleeve
    vol_target: float = 0.17       # annualised
    vol_days: int = 20
    max_exposure: float = 1.5      # 1.0 means never borrow
    trend_ma: int = 200            # borrowing only while the core index is above this moving average (days); 0 = always
    act_days_late: int = 1         # backtest assumes acting this many days after the signal, on top of the next-session trade
    band: float = 0.10             # exposure changes only when the target is this far from the position
    borrow_spread: float = 0.01    # above T-bills, per year
    cost_exposure: float = 0.0005  # on each change in exposure
    cost_sector: float = 0.001     # on sector trades, each side


def sector_weights(close: pd.DataFrame, s: Settings = Settings(), months: pd.PeriodIndex | None = None) -> pd.DataFrame:
    """Weights by calendar month. A month's picks use prices only up to the end of the month before last (when skip_months=1)."""
    names = [c for c in close.columns if c in SECTOR_ETFS]
    month_end = close[names].groupby(close.index.to_period("M")).last()
    rets = np.log(month_end / month_end.shift(1))
    index = month_end.index if months is None else month_end.index.union(months)
    rets = rets.reindex(index)
    needed = s.look_months
    mom = rets.shift(1 + s.skip_months).rolling(s.look_months, min_periods=needed).sum()
    pick = mom.rank(axis=1, ascending=False, method="first").le(s.top) & mom.notna()
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def book_returns(close: pd.DataFrame, s: Settings = Settings()) -> pd.DataFrame:
    """Daily returns of the unscaled book and of its two halves, net of sector trading costs."""
    r = close.pct_change(fill_method=None)
    w = sector_weights(close, s)
    days = pd.PeriodIndex(close.index, freq="M")
    daily_w = w.reindex(days).set_axis(close.index).fillna(0.0)
    names = daily_w.columns
    invested = daily_w.sum(axis=1) > 0
    sleeve = (daily_w * r[names]).sum(axis=1)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) / 2
    first_day = pd.Series(days, index=close.index).ne(pd.Series(days, index=close.index).shift(1))
    fee = turn.reindex(days).set_axis(close.index).where(first_day, 0.0).fillna(0.0) * s.cost_sector * 2
    sleeve = (sleeve - fee).where(invested)
    core = r[CORE]
    book = ((1 - s.sector_share) * core + s.sector_share * sleeve).where(invested, core)
    return pd.DataFrame({"core": core, "sleeve": sleeve, "book": book}).dropna(subset=["core"])


def leverage_cap(core: pd.Series, s: Settings = Settings()) -> pd.Series:
    """max_exposure while the core index is above its trend_ma-day average, 1.0 below it. Before the average exists (warm-up)
    the cap is max_exposure, so a short history behaves like the ungated program."""
    level = (1 + core.fillna(0.0)).cumprod()
    if not s.trend_ma:
        return pd.Series(s.max_exposure, index=core.index)
    ma = level.rolling(s.trend_ma, min_periods=s.trend_ma).mean()
    return pd.Series(np.where((level > ma) | ma.isna(), s.max_exposure, 1.0), index=core.index)


def exposure_target(book: pd.Series, s: Settings = Settings(), core: pd.Series | None = None) -> pd.Series:
    vol = book.rolling(s.vol_days, min_periods=s.vol_days).std() * np.sqrt(252)
    cap = leverage_cap(core.reindex(book.index), s) if core is not None else s.max_exposure
    return (s.vol_target / vol).clip(upper=cap)


def scale(book: pd.Series, rf: pd.Series, s: Settings = Settings(), core: pd.Series | None = None) -> pd.DataFrame:
    """Apply the exposure rule. The position changes after a day's close, using a signal already act_days_late old.
    `core` (daily returns of the core index) drives the trend gate; without it the gate is off."""
    target = exposure_target(book, s, core).shift(s.act_days_late).fillna(1.0).to_numpy()
    x, f = book.to_numpy(), rf.reindex(book.index).ffill().fillna(0.0).to_numpy()
    out, pos, held, spread = np.zeros(len(book)), np.zeros(len(book)), 1.0, s.borrow_spread / 252
    for i in range(len(book)):
        pos[i] = held
        out[i] = held * x[i] + (1 - held) * f[i] - max(held - 1, 0) * spread
        if abs(target[i] - held) > s.band:
            out[i] -= abs(target[i] - held) * s.cost_exposure
            held = target[i]
    return pd.DataFrame({"ret": out, "exposure": pos}, index=book.index)


def current_allocation(close: pd.DataFrame, s: Settings = Settings()) -> dict:
    """Target holdings for the next session, from prices through the latest close."""
    close = close.sort_index()
    last = close.index[-1]
    next_day = last + pd.offsets.BDay(1)
    month = pd.Period(next_day, "M")
    w = sector_weights(close, s, pd.PeriodIndex([month])).loc[month]
    picks = w[w > 0]
    parts = book_returns(close, s)
    book = parts["book"].dropna()
    cap = float(leverage_cap(parts["core"], s).iloc[-1])
    exposure = float(exposure_target(book, s, parts["core"]).iloc[-1])
    vol = float(book.tail(s.vol_days).std() * np.sqrt(252))
    holdings = {CORE: (1 - s.sector_share) * exposure, **{t: float(v) * s.sector_share * exposure for t, v in picks.items()}}
    return {"as_of": last.date(), "for_month": str(month), "sectors": list(picks.index), "recent_vol": vol, "exposure": exposure,
            "leverage_cap": cap, "above_trend": cap > 1.0 or not s.trend_ma,
            "holdings": holdings, "cash": 1 - exposure, "settings": s}
