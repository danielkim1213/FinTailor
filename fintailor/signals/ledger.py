"""Paper ledger: assume each client follows their signals; mark positions to market to enforce
weekly loss limits and position counts. Entry = next open; exit = stop (or gap through it) or
close of the last holding day."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def mark_positions(con, signals: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """signals: rows from the signals table (status sent). Returns per-signal state."""
    if signals.empty:
        return signals.assign(state=[], entry=[], exit=[], ret=[], pnl_usd=[], exit_date=[])
    tick = sorted(signals["ticker"].unique())
    con.register("_t", pd.DataFrame({"t": tick}))
    px = con.execute("""select ticker, date, open, high, low, close from prices_daily
                        where ticker in (select t from _t) and date >= ? and date <= ?""",
                     [pd.to_datetime(signals["run_date"]).min().date(), asof.date()]).df()
    con.unregister("_t")
    px["date"] = pd.to_datetime(px["date"])
    by_t = {t: g.sort_values("date").reset_index(drop=True) for t, g in px.groupby("ticker")}
    out = []
    for s in signals.itertuples():
        p = json.loads(s.payload)
        g = by_t.get(s.ticker)
        rec = {"signal_id": s.signal_id, "customer_id": s.customer_id, "ticker": s.ticker, "state": "pending",
               "entry": np.nan, "exit": np.nan, "ret": 0.0, "pnl_usd": 0.0, "exit_date": None,
               "entry_date": None, "position_usd": p.get("max_position_usd", 0.0)}
        if g is None:
            out.append(rec)
            continue
        after = g[g["date"] > pd.Timestamp(s.run_date)]
        if after.empty:
            out.append(rec)
            continue
        hold = after.iloc[: int(s.horizon)]
        entry = float(hold["open"].iloc[0])
        sign = 1 if s.direction == "long" else -1
        stop = entry * (1 - sign * p["stop_pct"])
        rec.update(entry=entry, entry_date=hold["date"].iloc[0], state="open")
        exit_px, exit_d = float(hold["close"].iloc[-1]), hold["date"].iloc[-1]
        for i, r in enumerate(hold.itertuples()):
            hit = r.low <= stop if sign > 0 else r.high >= stop
            if hit:
                gap_through = (r.open <= stop) if sign > 0 else (r.open >= stop)
                exit_px, exit_d = (float(r.open) if (gap_through and i > 0) else stop), r.date
                rec["state"] = "stopped"
                break
        else:
            if len(hold) >= int(s.horizon):
                rec["state"] = "closed"
        rec["exit"], rec["exit_date"] = exit_px, (exit_d if rec["state"] != "open" else None)
        rec["ret"] = sign * (exit_px / entry - 1)
        rec["pnl_usd"] = rec["ret"] * rec["position_usd"]
        out.append(rec)
    return pd.DataFrame(out)


def customer_state(con, customer_id: str, asof: pd.Timestamp, capital_usd: float) -> dict:
    sig = con.execute("select * from signals where customer_id = ? and status = 'sent' and run_date >= ?",
                      [customer_id, (asof - pd.Timedelta(days=60)).date()]).df()
    if sig.empty:
        return {"open_positions": 0, "open_tickers": [], "weekly_pnl_pct": 0.0}
    st = mark_positions(con, sig, asof)
    week_start = asof - pd.Timedelta(days=asof.weekday())
    active = st[st["state"].isin(["open", "pending"])]
    wk = st[(st["state"] == "open") | (pd.to_datetime(st["exit_date"]) >= week_start)]
    return {"open_positions": int(len(active)), "open_tickers": active["ticker"].tolist(),
            "weekly_pnl_pct": float(wk["pnl_usd"].sum() / capital_usd) if capital_usd else 0.0,
            "positions": st}
