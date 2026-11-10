"""Which stock characteristics earned more than the market among LARGE companies, over as long as the data goes?

Uses Ken French's value-weighted big-cap portfolios, long-only and net of nothing, so this is a prior on what to tilt toward.
Excess is big-cap portfolio minus the market. Data after 2025-10 excluded.

Usage: python -I research/factor_study.py <french_directory>
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 240, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.2f}")
END = pd.Period("2025-10", "M")


def section(folder: Path, name: str) -> tuple[list[str], pd.DataFrame]:
    """First block of monthly rows in a French CSV (value-weighted returns)."""
    with zipfile.ZipFile(folder / name) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head, rows, on = None, [], False
    for line in lines:
        p = [x.strip() for x in line.split(",")]
        if p[0].isdigit() and len(p[0]) == 6:
            on = True
            rows.append(p)
        elif on:
            break
        elif len(p) > 3 and p[0] == "":
            head = p[1:]
    d = pd.DataFrame([r[1:] for r in rows], index=pd.PeriodIndex([f"{r[0][:4]}-{r[0][4:]}" for r in rows], freq="M"), columns=head).astype(float)
    return head, d.where(d > -99) / 100


_, ff = section(Path(sys.argv[1]), "F-F_Research_Data_Factors_CSV.zip")
mkt = ff["Mkt-RF"] + ff["RF"]
folder = Path(sys.argv[1])
SOURCES = {
    "momentum (past winners)": ("6_Portfolios_ME_Prior_12_2_CSV.zip", -1, "big high prior"),
    "profitability (robust)": ("6_Portfolios_ME_OP_2x3_CSV.zip", -1, "big robust"),
    "low investment (conservative)": ("6_Portfolios_ME_INV_2x3_CSV.zip", -3, "big low inv"),
    "value (high book/price)": ("6_Portfolios_2x3_CSV.zip", -1, "big high bm"),
}
ports = {}
for label, (fname, col, _) in SOURCES.items():
    h, d = section(folder, fname)
    ports[label] = d.iloc[:, col]
    print(f"{label:32s} uses column '{d.columns[col]}', from {d.index[0]}")
P = pd.DataFrame(ports)
P = P[P.index <= END]
m = mkt[mkt.index <= END]
P["blend: mom+prof+inv"] = P[["momentum (past winners)", "profitability (robust)", "low investment (conservative)"]].mean(axis=1, skipna=False)
P["blend: all four"] = P[list(ports)].mean(axis=1, skipna=False)


def nw_t(x, L=12):
    x = x.dropna().values
    e, T = x - x.mean(), len(x)
    v = e @ e / T
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / T
    return x.mean() / np.sqrt(v / T)


rows = {}
for c in P.columns:
    ex = (P[c] - m).dropna()
    r = {"from": str(ex.index[0]), "ex_%/yr": ex.mean() * 1200, "t": nw_t(ex), "TE_%": ex.std() * np.sqrt(12) * 100}
    for lab, a, b in (("1963-99", "1963-07", "1999-12"), ("2000-09", "2000-01", "2009-12"), ("2010-19", "2010-01", "2019-12"), ("2020-25", "2020-01", "2025-10")):
        s = ex[a:b]
        r[lab] = s.mean() * 1200
    r["t_2000+"] = nw_t(ex["2000-01":])
    r["IR"] = ex.mean() / ex.std() * np.sqrt(12)
    r["beat_10y_%"] = float(np.log1p(P[c]).sub(np.log1p(m)).dropna().rolling(120).sum().dropna().gt(0).mean() * 100)
    rows[c] = r
print("\n=== big-cap portfolio minus market, %/yr (long-only, value-weighted, before costs)\n", pd.DataFrame(rows).T.to_string())

t = P[list(ports)].sub(m, axis=0).dropna()
print("\ncorrelation of excess returns (since first common month):\n", t.corr().round(2).to_string())
