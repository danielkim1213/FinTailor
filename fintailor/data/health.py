"""Pre-run data checks. Failures trip the kill switch (signals stop until a human resumes)."""
from __future__ import annotations

import pandas as pd

from ..config import get_config


def check_data(con, asof: pd.Timestamp | None = None) -> dict:
    cfg = get_config()
    bench = cfg["universe"]["benchmark"]
    issues: list[str] = []
    last = con.execute("select max(date) from prices_daily where ticker = ?", [bench]).fetchone()[0]
    if last is None:
        return {"ok": False, "issues": ["no benchmark prices"], "last_date": None}
    last = pd.Timestamp(last)
    asof = pd.Timestamp(asof or pd.Timestamp.today().normalize())
    stale = len(pd.bdate_range(last, asof)) - 1
    if stale > cfg["killswitch"]["max_stale_days"]:
        issues.append(f"benchmark data stale by {stale} business days (last {last.date()})")

    cov = con.execute("""
        with live as (select ticker from universe where (end_date is null or end_date >= ?) and start_date <= ?
                      and sector <> 'benchmark')
        select count(*) filter (where p.ticker is not null) * 1.0 / count(*) from live
        left join prices_daily p on p.ticker = live.ticker and p.date = ?""", [last.date(), last.date(), last.date()]).fetchone()[0]
    if cov is not None and cov < 0.95:
        issues.append(f"only {cov:.0%} of live universe has a bar on {last.date()}")

    bad = con.execute("""select count(*) from prices_daily where date = ? and
                         (close <= 0 or high < low or adj_close is null or volume < 0)""", [last.date()]).fetchone()[0]
    if bad:
        issues.append(f"{bad} invalid bars on {last.date()}")
    jumps = con.execute("""
        with p as (select ticker, date, adj_close / lag(adj_close) over (partition by ticker order by date) - 1 r
                   from prices_daily where date >= ? - interval 10 day)
        select count(*) from p where date = ? and abs(r) > 0.6""", [last.date(), last.date()]).fetchone()[0]
    if jumps > 3:
        issues.append(f"{jumps} tickers moved >60% on {last.date()} (possible bad adjustment)")
    return {"ok": not issues, "issues": issues, "last_date": str(last.date()), "coverage": cov}
