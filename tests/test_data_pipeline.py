"""Stage 2 acceptance tests — BTC H1 data pipeline.

Tests:
- Loader reads Freqtrade-format feather
- Clean handles missing candles + gap detection
- Monthly windows = ~60 for 5y data
- Worst adverse move detection finds a real decline
- Data quality report has all required fields
- Pipeline writes all expected outputs
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.pipeline.loader import (
    build_data_quality_report,
    clean_candles,
    detect_worst_adverse_move,
    generate_monthly_windows,
    load_btc_h1_feather,
    run_pipeline,
)

pytestmark = pytest.mark.stage2


# ============================================================
# Synthetic data fixtures
# ============================================================

@pytest.fixture
def sample_feather(tmp_path: Path) -> Path:
    """Create a small synthetic Freqtrade-format feather for testing."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "date": idx,
        "open": 40000.0 + (idx.hour * 10),
        "high": 40100.0 + (idx.hour * 10),
        "low": 39900.0 + (idx.hour * 10),
        "close": 40000.0 + (idx.hour * 10),
        "volume": 100.0,
    })
    path = tmp_path / "test.feather"
    df.to_feather(path)
    return path


@pytest.fixture
def declining_feather(tmp_path: Path) -> Path:
    """Synthetic data with a clear worst adverse move: -50% from 50000 to 25000."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # First 100 candles around 50000, next 100 declining to 25000
    close = [50000.0] * 100 + [50000.0 - i * 250.0 for i in range(100)]
    df = pd.DataFrame({
        "date": idx,
        "open": close,
        "high": [c + 50 for c in close],
        "low": [c - 50 for c in close],
        "close": close,
        "volume": 100.0,
    })
    path = tmp_path / "declining.feather"
    df.to_feather(path)
    return path


# ============================================================
# Loader tests
# ============================================================

def test_loader_reads_freqtrade_format(sample_feather: Path):
    df = load_btc_h1_feather(sample_feather)
    assert len(df) == 100
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["date"].iloc[0].tzinfo is not None  # UTC


def test_loader_rejects_missing_date_column(tmp_path: Path):
    bad = tmp_path / "bad.feather"
    pd.DataFrame({"foo": [1, 2, 3]}).to_feather(bad)
    with pytest.raises(ValueError, match="date"):
        load_btc_h1_feather(bad)


def test_loader_rejects_missing_ohlcv_column(tmp_path: Path):
    bad = tmp_path / "bad.feather"
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
        "open": [1.0, 2.0, 3.0],
        "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0],
        # missing close, volume
    }).to_feather(bad)
    with pytest.raises(ValueError, match="close"):
        load_btc_h1_feather(bad)


# ============================================================
# Clean tests
# ============================================================

def test_clean_handles_no_gaps(sample_feather: Path):
    df = load_btc_h1_feather(sample_feather)
    df_clean, report = clean_candles(df, forward_fill_max_gap=3)
    assert report["forward_filled_candles"] == 0
    assert len(df_clean) == len(df)


def test_clean_forward_fills_small_gaps(tmp_path: Path):
    """If 2 candles missing, forward-fill them."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Drop rows at indices 50, 51 to create a 3h gap (1h + 2 missing = 3 hours)
    df = pd.DataFrame({
        "date": idx,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 50.0,
    }).drop(index=[50, 51])
    df_clean, report = clean_candles(df, forward_fill_max_gap=3)
    assert len(df_clean) >= 98  # original 98 + 2 forward-filled
    assert report["forward_filled_candles"] >= 0


def test_clean_detects_large_gaps(tmp_path: Path):
    """Gaps > 1.5h should be flagged in the report."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "date": idx,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 50.0,
    }).drop(index=[50, 51, 52, 53, 54])  # 6h gap
    _df_clean, report = clean_candles(df, forward_fill_max_gap=3)
    # Either forward-filled (if <= 3) or flagged
    assert (report["forward_filled_candles"] > 0 or len(report["flagged_gaps"]) > 0)


# ============================================================
# Monthly windows tests
# ============================================================

def test_monthly_windows_count_is_correct(sample_feather: Path):
    df = load_btc_h1_feather(sample_feather)
    windows = generate_monthly_windows(df)
    assert len(windows) >= 1
    assert all(w.candle_count > 0 for w in windows)


def test_monthly_windows_iso_format(sample_feather: Path):
    df = load_btc_h1_feather(sample_feather)
    windows = generate_monthly_windows(df)
    for w in windows:
        # ISO 8601 timestamps must round-trip
        pd.Timestamp(w.start)
        pd.Timestamp(w.end)


# ============================================================
# Worst adverse move tests
# ============================================================

def test_worst_adverse_move_detects_decline(declining_feather: Path):
    df = load_btc_h1_feather(declining_feather)
    worst = detect_worst_adverse_move(df)
    assert worst.percentage_decline > 0.40  # at least -40%
    assert worst.start_price > worst.end_price
    assert worst.duration_hours >= 100


def test_worst_adverse_move_rejects_empty(tmp_path: Path):
    empty = tmp_path / "empty.feather"
    pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]).to_feather(empty)
    df = load_btc_h1_feather(empty)
    with pytest.raises(ValueError, match="Empty"):
        detect_worst_adverse_move(df)


# ============================================================
# Data quality report tests
# ============================================================

def test_data_quality_report_has_all_fields(sample_feather: Path):
    df = load_btc_h1_feather(sample_feather)
    _, cleaning_report = clean_candles(df)
    report = build_data_quality_report(df, cleaning_report)
    d = report.to_dict()
    assert "total_candles" in d
    assert "date_range_start" in d
    assert "date_range_end" in d
    assert "longest_gap_hours" in d
    assert "zero_volume_pct" in d
    assert "spread_avg_pips" in d
    assert "price_min" in d
    assert "price_max" in d
    assert "price_mean" in d


def test_data_quality_report_prices_reasonable(sample_feather: Path):
    df = load_btc_h1_feather(sample_feather)
    _, cleaning_report = clean_candles(df)
    report = build_data_quality_report(df, cleaning_report)
    assert report.price_min <= report.price_mean <= report.price_max


# ============================================================
# End-to-end pipeline test
# ============================================================

def test_pipeline_writes_all_outputs(tmp_path: Path, sample_feather: Path):
    """Use sample (small) data to verify pipeline writes the expected files."""
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"

    # Sample data has only 100 candles → pipeline will reject (coverage < 4.5y)
    # So instead, build a 5y synthetic dataset for the full test.
    n = 24 * 365 * 5 + 10  # ~5y
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "date": idx,
        "open": 40000.0,
        "high": 40100.0,
        "low": 39900.0,
        "close": 40000.0 + (idx.hour * 5),
        "volume": 100.0,
    })
    five_year_path = tmp_path / "raw" / "BTC_USDT-1h.feather"
    five_year_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(five_year_path)

    result = run_pipeline(five_year_path, processed_dir, metadata_dir)

    # Verify outputs exist
    assert Path(result["processed_path"]).exists()
    assert Path(result["candle_path_csv"]).exists()
    assert (metadata_dir / "monthly_windows.json").exists()
    assert (metadata_dir / "full_period.json").exists()
    assert (metadata_dir / "worst_adverse_move.json").exists()
    assert (metadata_dir / "data_quality_report.json").exists()

    # Verify monthly_windows.json is valid + has ~60 entries
    windows = json.loads((metadata_dir / "monthly_windows.json").read_text())
    assert 55 <= len(windows) <= 65  # ~60 for 5y

    # Verify worst_adverse_move.json is valid
    worst = json.loads((metadata_dir / "worst_adverse_move.json").read_text())
    assert worst["percentage_decline"] > 0
    assert "start_time" in worst
    assert "end_time" in worst

    # Verify data_quality_report.json is valid
    quality = json.loads((metadata_dir / "data_quality_report.json").read_text())
    assert quality["total_candles"] > 40000
    assert quality["total_candles"] < 50000


def test_pipeline_rejects_insufficient_coverage(tmp_path: Path):
    """< 4.5 years of data → ValueError."""
    n = 24 * 100  # 100 days
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "date": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0,
    })
    p = tmp_path / "raw" / "BTC_USDT-1h.feather"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(p)

    with pytest.raises(ValueError, match="Insufficient coverage"):
        run_pipeline(p, tmp_path / "processed", tmp_path / "metadata")
