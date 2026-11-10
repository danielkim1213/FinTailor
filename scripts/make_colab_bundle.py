"""Build fintailor_colab.zip for training/colab_finetune.ipynb.

Contains only what Colab needs: the package code, training scripts, config, and the three dataset files.
Never includes .env (API keys), store/ (DB), .venv/ or caches.

    python scripts/make_colab_bundle.py
Then upload fintailor_colab.zip to Google Drive at MyDrive/fintailor/fintailor_colab.zip
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = ["news_train.jsonl", "news_test.jsonl", "explain_train.jsonl"]
CODE = ["fintailor/**/*.py", "training/*.py", "training/colab_finetune.ipynb", "config.toml", "requirements.txt"]


def main() -> None:
    data_dir = ROOT / "training" / "data"
    missing = [f for f in DATA_FILES if not (data_dir / f).exists() or (data_dir / f).stat().st_size == 0]
    if missing:
        sys.exit(f"missing dataset files in training/data: {missing} - build them first")
    out = ROOT / "fintailor_colab.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in CODE:
            for f in sorted(ROOT.glob(pattern)):
                rel = f.relative_to(ROOT)
                if "__pycache__" in rel.parts or f.name.startswith(".env"):
                    continue
                z.write(f, Path("repo") / rel)
                n += 1
        for name in DATA_FILES:
            z.write(data_dir / name, Path("data") / name)
            n += 1
        names = z.namelist()
    assert not any(Path(x).name.startswith(".env") for x in names), "refusing: .env in bundle"
    rows = {name: sum(1 for _ in open(data_dir / name, encoding="utf-8")) for name in DATA_FILES}
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} files)")
    for name, k in rows.items():
        print(f"  data/{name}: {k} rows")
    print("Upload it to Google Drive as MyDrive/fintailor/fintailor_colab.zip")


if __name__ == "__main__":
    main()
