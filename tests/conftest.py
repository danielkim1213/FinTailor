import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_STORE = Path(tempfile.mkdtemp(prefix="ft_test_"))
os.environ["FINTAILOR_STORE"] = str(_STORE)
os.environ["FINTAILOR_OFFLINE"] = "1"
for k in ("LLM_BASE_URL", "LLM_MODEL"):
    os.environ.pop(k, None)


@pytest.fixture(scope="session")
def panel():
    from fintailor.config import reset_config
    reset_config()
    from fintailor.data.synthetic import load_synthetic
    from fintailor.features.panel import build_panel
    load_synthetic(n_stocks=60, start="2017-01-01", end="2021-12-31")
    return build_panel()
