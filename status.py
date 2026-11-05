"""System status snapshot (CLI `status`, dashboard, API)."""
from __future__ import annotations

from .db import connect, get_state, table_count
from .models.registry import registry


def status() -> dict:
    with connect(read_only=True) as con:
        counts = {t: table_count(con, t) for t in ("prices_daily", "universe", "news", "news_scores", "macro", "events",
                                                   "earnings", "filings", "fundamentals", "predictions", "outcomes",
                                                   "signals", "reviews", "customers", "rag_docs")}
        last_price = con.execute("select max(date) from prices_daily").fetchone()[0]
        last_news = con.execute("select max(received_at) from news").fetchone()[0]
        last_run = con.execute("select run_id, min(started_at), max(finished_at), "
                               "count(*) filter (where status = 'error') from run_log group by run_id "
                               "order by 2 desc limit 1").fetchone()
        ks = get_state(con, "kill_switch", {"halted": False})
    reg = registry()
    return {"counts": counts, "last_price_date": last_price, "last_news": last_news,
            "last_run": {"id": last_run[0], "started": last_run[1], "finished": last_run[2], "errors": last_run[3]} if last_run else None,
            "kill_switch": ks, "champion": reg.get("champion"), "n_versions": len(reg.get("versions", []))}
