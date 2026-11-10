"""Download Ken French's industry portfolio files into their own folder.   python -I research/french_fetch2.py <empty_directory>"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name in ("10_Industry_Portfolios_CSV.zip", "30_Industry_Portfolios_CSV.zip"):
    with urllib.request.urlopen(urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0 (research)"}), timeout=120) as r:
        (out / name).write_bytes(r.read())
    print(name, flush=True)
