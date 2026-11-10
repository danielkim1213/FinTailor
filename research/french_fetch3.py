"""Download Ken French's daily 10-industry file into its own folder.   python -I research/french_fetch3.py <empty_directory>"""
import sys
import urllib.request
from pathlib import Path

name = "10_Industry_Portfolios_daily_CSV.zip"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + name
with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"}), timeout=120) as r:
    (out / name).write_bytes(r.read())
print(name)
