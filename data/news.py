"""News collection (Finnhub company + general news, RSS). Every item is stored with the time WE
received it (point-in-time). Backfilled history is flagged: its received_at = published_at + 15m."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ..config import env, get_config
from ..db import connect, upsert
from ..utils import RateLimiter, http_get, log, stable_id, utcnow

FINNHUB = "https://finnhub.io/api/v1"
_lim = RateLimiter(55)


def _finnhub(path: str, **params):
    tok = env("FINNHUB_API_KEY")
    if not tok:
        raise RuntimeError("FINNHUB_API_KEY not set")
    return http_get(f"{FINNHUB}/{path}", params={**params, "token": tok}, limiter=_lim).json()


def _rows_from_finnhub(items: list[dict], ticker: str | None, backfill: bool) -> list[dict]:
    now = utcnow()
    out = []
    for it in items or []:
        pub = pd.to_datetime(it.get("datetime"), unit="s", errors="coerce")
        if pd.isna(pub) or not it.get("headline"):
            continue
        out.append(dict(
            id=f"fh_{it.get('id') or stable_id(it.get('headline'), pub)}",
            ticker=ticker, category=it.get("category"), headline=it["headline"][:500],
            summary=(it.get("summary") or "")[:2000], source=it.get("source"), url=it.get("url"),
            published_at=pub.to_pydatetime(),
            received_at=(pub + pd.Timedelta(minutes=15)).to_pydatetime() if backfill else now,
            backfilled=backfill))
    return out


def collect_finnhub(days: int = 2, tickers: list[str] | None = None, backfill: bool = False) -> int:
    cfg = get_config()["news"]
    with connect(read_only=True) as con:
        if tickers is None:
            # most liquid live members first
            tickers = con.execute("""
                select p.ticker from prices_daily p join universe u using(ticker)
                where p.date >= current_date - interval 40 day and (u.end_date is null)
                group by p.ticker order by avg(p.close * p.volume) desc""").df()["ticker"].tolist()
    tickers = tickers[: cfg["finnhub_max_tickers"]]
    to = utcnow().date()
    frm = to - timedelta(days=days)
    rows: list[dict] = []
    try:
        rows += _rows_from_finnhub(_finnhub("news", category="general"), None, backfill=False)
    except Exception as e:  # noqa: BLE001
        log.warning("finnhub general news: %s", e)
        if "not set" in str(e):
            return 0
    for i, t in enumerate(tickers):
        # Finnhub caps results per call; walk in month chunks for backfills
        step = 30 if backfill else days + 1
        s = frm
        while s <= to:
            e = min(s + timedelta(days=step), to)
            try:
                rows += _rows_from_finnhub(_finnhub("company-news", symbol=t, **{"from": str(s), "to": str(e)}),
                                           t, backfill)
            except Exception as ex:  # noqa: BLE001
                log.warning("finnhub %s: %s", t, ex)
            s = e + timedelta(days=1)
        if i % 50 == 0:
            log.info("finnhub news: %d/%d tickers, %d items", i, len(tickers), len(rows))
    df = pd.DataFrame(rows).drop_duplicates("id") if rows else pd.DataFrame()
    with connect() as con:
        n = upsert(con, "news", df, mode="ignore")  # never overwrite the original received_at
    log.info("finnhub: %d news rows", n)
    return n


def collect_rss() -> int:
    import feedparser

    now = utcnow()
    rows = []
    for url in get_config()["news"].get("rss_feeds", []):
        try:
            feed = feedparser.parse(http_get(url, timeout=20).content)
        except Exception as e:  # noqa: BLE001
            log.warning("rss %s: %s", url, e)
            continue
        for en in feed.entries:
            title = getattr(en, "title", None)
            if not title:
                continue
            pub = None
            if getattr(en, "published_parsed", None):
                pub = pd.Timestamp(*en.published_parsed[:6]).to_pydatetime()
            rows.append(dict(id=f"rss_{stable_id(url, title)}", ticker=None, category="rss", headline=title[:500],
                             summary=(getattr(en, "summary", "") or "")[:2000], source=feed.feed.get("title", url),
                             url=getattr(en, "link", None), published_at=pub or now, received_at=now,
                             backfilled=False))
    df = pd.DataFrame(rows).drop_duplicates("id") if rows else pd.DataFrame()
    with connect() as con:
        n = upsert(con, "news", df, mode="ignore")
    log.info("rss: %d items", n)
    return n


def collect_news(backfill: bool = False) -> int:
    days = get_config()["news"]["backfill_days"] if backfill else 3
    n = collect_finnhub(days=days, backfill=backfill)
    return n + collect_rss()
