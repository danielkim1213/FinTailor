"""DuckDB storage. One file, short-lived connections (so the dashboard and nightly job coexist)."""
from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

from .config import get_config

SCHEMA = """
create table if not exists prices_daily(
  ticker varchar, date date, open double, high double, low double, close double,
  adj_close double, raw_close double, volume double, vwap double, source varchar,
  primary key(ticker, date));
create table if not exists universe(
  ticker varchar, name varchar, sector varchar, start_date date, end_date date,
  is_etf boolean, source varchar);
create table if not exists news(
  id varchar primary key, ticker varchar, category varchar, headline varchar, summary varchar,
  source varchar, url varchar, published_at timestamp, received_at timestamp, backfilled boolean);
create table if not exists news_scores(
  news_id varchar, scorer varchar, payload varchar, sentiment double, severity double,
  novelty varchar, event_type varchar, created_at timestamp, primary key(news_id, scorer));
create table if not exists macro(
  series_id varchar, date date, value double, available_date date, primary key(series_id, date));
create table if not exists events(date date, event varchar, primary key(date, event));
create table if not exists earnings(
  ticker varchar, date date, announced_at timestamp, eps_actual double, eps_estimate double,
  source varchar, primary key(ticker, date));
create table if not exists filings(
  accession varchar primary key, ticker varchar, cik varchar, form varchar, filing_date date,
  accepted_at timestamp, items varchar);
create table if not exists fundamentals(
  ticker varchar, filed date, period_end date, rev_ttm double, ni_ttm double, eps_ttm double,
  rev_ttm_yoy double, eps_ttm_yoy double, primary key(ticker, filed, period_end));
create table if not exists world_daily(
  source varchar, metric varchar, date date, value double, primary key(source, metric, date));
create table if not exists sector_views(
  date date, sector varchar, score double, source varchar, rationale varchar,
  primary key(date, sector, source));
create table if not exists predictions(
  run_date date, ticker varchar, horizon integer, model_version varchar, pred_excess double,
  prob double, score double, rank_pct double, official boolean, drivers varchar,
  created_at timestamp, primary key(run_date, ticker, horizon, model_version));
create table if not exists outcomes(
  run_date date, ticker varchar, horizon integer, realized_net double, realized_gross double,
  spy_ret double, matured_date date, primary key(run_date, ticker, horizon));
create table if not exists customers(customer_id varchar primary key, profile varchar, updated_at timestamp);
create table if not exists signals(
  signal_id varchar primary key, run_date date, customer_id varchar, ticker varchar,
  direction varchar, horizon integer, payload varchar, explanation varchar, status varchar,
  created_at timestamp);
create table if not exists reviews(
  review_id varchar primary key, run_date date, ticker varchar, horizon integer, category varchar,
  verified boolean, payload varchar, created_at timestamp);
create table if not exists finetune_queue(
  news_id varchar, headline varchar, reason varchar, payload varchar, created_at timestamp);
create table if not exists rag_docs(
  doc_id varchar primary key, kind varchar, text varchar, meta varchar, created_at timestamp);
create table if not exists system_state(key varchar primary key, value varchar, updated_at timestamp);
create table if not exists run_log(
  run_id varchar, step varchar, status varchar, detail varchar, started_at timestamp,
  finished_at timestamp);
alter table fundamentals add column if not exists gp_ttm double;
alter table fundamentals add column if not exists assets double;
alter table fundamentals add column if not exists assets_yoy double;
alter table fundamentals add column if not exists equity double;
alter table fundamentals add column if not exists shares double;
"""


def db_path() -> Path:
    p = Path(get_config()["paths"]["store"]) / "fintailor.duckdb"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_initialized: set[str] = set()
_local = threading.local()  # per-thread (con, read_only) stack for nested use


def _stack() -> list:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


@contextlib.contextmanager
def connect(read_only: bool = False, retries: int = 40):
    """Short-lived connection. Nested calls reuse the outer connection via a cursor (DuckDB does
    not allow two connections with different configs to one file in one process)."""
    _open = _stack()
    if _open:
        outer, outer_ro = _open[-1]
        if outer_ro and not read_only:
            raise RuntimeError("write connection requested inside a read-only connection")
        cur = outer.cursor()
        try:
            yield cur
        finally:
            cur.close()
        return
    path = db_path()
    ro = read_only and path.exists()
    con = None
    for i in range(retries):
        try:
            con = duckdb.connect(str(path), read_only=ro)
            break
        except duckdb.IOException:
            if i == retries - 1:
                raise
            time.sleep(0.5)
    _open.append((con, ro))
    try:
        if not ro and str(path) not in _initialized:
            con.execute(SCHEMA)
            _initialized.add(str(path))
        yield con
    finally:
        _open.pop()
        con.close()


def upsert(con, table: str, df: pd.DataFrame, mode: str = "replace") -> int:
    """mode: replace (INSERT OR REPLACE) | ignore (INSERT OR IGNORE) | append."""
    if df is None or len(df) == 0:
        return 0
    verb = {"replace": "INSERT OR REPLACE", "ignore": "INSERT OR IGNORE", "append": "INSERT"}[mode]
    con.register("_up_df", df)
    try:
        con.execute(f"{verb} INTO {table} BY NAME SELECT * FROM _up_df")
    finally:
        con.unregister("_up_df")
    return len(df)


def replace_where(con, table: str, df: pd.DataFrame, where: str, params: list | None = None) -> int:
    """Delete rows matching `where`, then append df (fast bulk replace)."""
    con.execute("BEGIN")
    try:
        con.execute(f"DELETE FROM {table} WHERE {where}", params or [])
        n = upsert(con, table, df, mode="append")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return n


def get_state(con, key: str, default=None):
    row = con.execute("select value from system_state where key = ?", [key]).fetchone()
    return json.loads(row[0]) if row else default


def set_state(con, key: str, value) -> None:
    from .utils import utcnow
    con.execute("insert or replace into system_state values (?, ?, ?)", [key, json.dumps(value, default=str), utcnow()])


def table_count(con, table: str) -> int:
    return con.execute(f"select count(*) from {table}").fetchone()[0]


def trading_dates(con) -> pd.DatetimeIndex:
    bench = get_config()["universe"]["benchmark"]
    d = con.execute("select distinct date from prices_daily where ticker = ? order by date", [bench]).df()
    return pd.DatetimeIndex(pd.to_datetime(d["date"])) if len(d) else pd.DatetimeIndex([])
