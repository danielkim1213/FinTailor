"""Read API for a future front-end / push notifications. Run: python -m fintailor api"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException

from ..db import connect
from ..signals.profiles import CustomerProfile, load_customers, save_customer
from ..status import status

app = FastAPI(title="FinTailor API (prototype)")


@app.get("/health")
def health():
    return status()


@app.get("/predictions/latest")
def latest_predictions(horizon: int = 5, limit: int = 20):
    with connect(read_only=True) as con:
        df = con.execute("""select * from predictions where official and horizon = ? and run_date =
                            (select max(run_date) from predictions where official) order by score desc limit ?""",
                         [horizon, limit]).df()
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/signals/{customer_id}")
def signals(customer_id: str, limit: int = 20):
    with connect(read_only=True) as con:
        df = con.execute("select * from signals where customer_id = ? order by run_date desc limit ?",
                         [customer_id, limit]).df()
    if df.empty:
        raise HTTPException(404, "no signals")
    df["payload"] = df["payload"].map(json.loads)
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/customers")
def customers():
    return [c.model_dump() for c in load_customers()]


@app.post("/customers")
def upsert_customer(p: CustomerProfile):
    save_customer(p)
    return {"ok": True, "risk_score": p.risk_score}
