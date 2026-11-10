"""Download Ken French's public return files (market since 1926, factor-sorted portfolios) into their own folder.

Usage: python -I research/french_fetch.py <empty_directory>
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = ["F-F_Research_Data_Factors_daily_CSV.zip", "F-F_Research_Data_Factors_CSV.zip",
         "6_Portfolios_2x3_CSV.zip", "6_Portfolios_ME_Prior_12_2_CSV.zip",
         "6_Portfolios_ME_OP_2x3_CSV.zip", "6_Portfolios_ME_INV_2x3_CSV.zip"]

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name in FILES:
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    (out / name).write_bytes(blob)
    print(name, f"{len(blob) / 1e3:.0f} KB", flush=True)
