"""Pytest config + fixtures for BTC H1 AutoRL tests."""
import sys
from pathlib import Path

# Add project root to sys.path so `from configs import ...` works without install
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from configs import Settings  # noqa: E402


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def settings(project_root: Path) -> Settings:
    """Default settings loaded from committed config files."""
    return Settings.from_files(
        freqtrade_path=project_root / "configs" / "freqtrade" / "config.json",
        experiment_path=project_root / "configs" / "experiments" / "default.json",
        research_path=project_root / "configs" / "research" / "default.json",
    )


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Per-test results directory — write metadata here, never to real results/."""
    d = tmp_path / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
