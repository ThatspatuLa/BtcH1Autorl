"""Stage 2 — BTC H1 data pipeline.

Reads Freqtrade-format feather file, validates 5y coverage, detects worst adverse
move (H1 peak-to-trough decline), produces monthly windows + data quality report.

Output files written to data/metadata/:
- monthly_windows.json
- full_period.json
- worst_adverse_move.json
- data_quality_report.json

Output processed file to data/processed/:
- btc_h1_5y.feather (Freqtrade-compatible)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Constants
# ============================================================

EXPECTED_CANDLES_5Y = 5 * 365 * 24  # ~43800, allowing for leap years
EXPECTED_START = pd.Timestamp("2021-06-01", tz="UTC")
EXPECTED_END = pd.Timestamp("2026-06-20", tz="UTC")
TOLERANCE_DAYS = 30  # Allow ±30 days for data provider quirks


# ============================================================
# Result types
# ============================================================

@dataclass
class MonthlyWindow:
    index: int
    start: str  # ISO 8601
    end: str
    candle_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorstAdverseMove:
    start_time: str
    end_time: str
    duration_hours: int
    duration_days: float
    percentage_decline: float  # positive = decline magnitude
    start_price: float
    end_price: float
    recovery_time_hours: int | None  # None if never recovered in dataset
    candle_path_csv: str  # path to CSV with full price path
    replay_file_path: str  # path to replay file (same as candle_path_csv)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataQualityReport:
    total_candles: int
    date_range_start: str
    date_range_end: str
    missing_pct: float
    longest_gap_hours: int
    zero_volume_pct: float
    session_gaps_count: int
    spread_avg_pips: float
    spread_max_pips: float
    price_min: float
    price_max: float
    price_mean: float
    price_std: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Loader
# ============================================================

def load_btc_h1_feather(path: Path | str) -> pd.DataFrame:
    """Load Freqtrade-format feather into a typed DataFrame.

    Freqtrade format: columns = [date, open, high, low, close, volume]
    `date` is datetime64[ns, UTC].
    """
    df = pd.read_feather(Path(path))
    if "date" not in df.columns:
        raise ValueError(f"Expected 'date' column in feather file, got: {list(df.columns)}")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing OHLCV column: {col}")
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_candles(
    df: pd.DataFrame,
    forward_fill_max_gap: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Clean candles: forward-fill small gaps, flag large gaps.

    Returns (cleaned_df, cleaning_report).
    """
    df = df.copy()
    cleaning_report = {
        "original_count": len(df),
        "forward_filled_candles": 0,
        "flagged_gaps": [],
    }

    if df.empty:
        return df, cleaning_report

    # Detect gaps by computing diff between consecutive dates
    expected_freq = pd.Timedelta(hours=1)
    time_diffs = df["date"].diff()
    gap_mask = time_diffs > expected_freq * 1.5  # any gap > 90 min
    if gap_mask.any():
        for idx in df.index[gap_mask]:
            gap_hours = time_diffs.iloc[idx].total_seconds() / 3600
            cleaning_report["flagged_gaps"].append({
                "after_date": df["date"].iloc[idx - 1].isoformat(),
                "before_date": df["date"].iloc[idx].isoformat(),
                "gap_hours": int(gap_hours),
            })

    # Forward-fill small gaps (insert synthetic candles)
    if forward_fill_max_gap > 0:
        full_idx = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="1h", tz="UTC")
        df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
        # Forward-fill OHLC from last close, then fill volume=0 for filled candles
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].ffill()
        df["volume"] = df["volume"].fillna(0.0)
        cleaning_report["forward_filled_candles"] = full_idx.size - cleaning_report["original_count"]
        # Drop leading NaN (before first real candle)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    return df, cleaning_report


def generate_monthly_windows(df: pd.DataFrame) -> list[MonthlyWindow]:
    """Split 5y data into ~60 monthly windows."""
    if df.empty or len(df) < 2:
        return []
    windows = []
    # Group by calendar month
    df_temp = df.copy()
    df_temp["year_month"] = df_temp["date"].dt.to_period("M")
    grouped = df_temp.groupby("year_month")
    for i, (_year_month, group) in enumerate(grouped):
        windows.append(MonthlyWindow(
            index=i,
            start=group["date"].iloc[0].isoformat(),
            end=group["date"].iloc[-1].isoformat(),
            candle_count=len(group),
        ))
    return windows


def detect_worst_adverse_move(df: pd.DataFrame) -> WorstAdverseMove:
    """Find the largest H1 peak-to-trough decline in the dataset.

    Uses close prices. Scans all (peak_idx, trough_idx) pairs where peak_idx < trough_idx,
    and returns the pair with maximum percentage decline.

    Returns the start (peak), end (trough), duration, percentage decline, and a CSV path
    with the full price path around the event.
    """
    if df.empty or len(df) < 2:
        raise ValueError("Empty or single-row dataframe — cannot detect worst adverse move")

    close = df["close"].values
    dates = df["date"].values
    n = len(close)

    # Two-pointer scan: walk pointer i forward, track min_close_so_far AFTER i
    # For each peak at i, the worst future trough is at argmin of close[i+1:]
    # We want max((peak - trough) / peak)
    worst_dd = 0.0
    worst_peak_idx = 0
    worst_trough_idx = 0

    # Brute O(n²) is fine for n=44k (would take ~5 seconds); but O(n) is cleaner.
    # Track minimum close seen so far starting from each i.
    # Forward pass: for each i, compute (close[i] - min_close_after_i) / close[i]
    # We need min close strictly after i.
    suffix_min = np.full(n, np.inf)
    suffix_min_idx = np.full(n, -1, dtype=int)
    suffix_min[n - 1] = close[n - 1]
    suffix_min_idx[n - 1] = n - 1
    for k in range(n - 2, -1, -1):
        if close[k + 1] < suffix_min[k + 1]:
            suffix_min[k] = close[k + 1]
            suffix_min_idx[k] = k + 1
        else:
            suffix_min[k] = suffix_min[k + 1]
            suffix_min_idx[k] = suffix_min_idx[k + 1]

    for i in range(n - 1):
        peak_val = close[i]
        if peak_val <= 0:
            continue
        trough_val = suffix_min[i]
        dd = (peak_val - trough_val) / peak_val
        if dd > worst_dd:
            worst_dd = dd
            worst_peak_idx = i
            worst_trough_idx = suffix_min_idx[i]

    if worst_dd == 0:
        raise ValueError("No adverse move detected (data may be monotonically increasing)")

    start_date = pd.Timestamp(dates[worst_peak_idx])
    end_date = pd.Timestamp(dates[worst_trough_idx])

    # Recovery time: first candle after trough where close >= peak_val
    recovery_hours = None
    for j in range(worst_trough_idx + 1, len(close)):
        if close[j] >= peak_val:
            recovery_hours = j - worst_trough_idx
            break

    return WorstAdverseMove(
        start_time=start_date.isoformat(),
        end_time=end_date.isoformat(),
        duration_hours=int((end_date - start_date).total_seconds() // 3600),
        duration_days=float((end_date - start_date).total_seconds() / 86400),
        percentage_decline=float(worst_dd),
        start_price=float(peak_val),
        end_price=float(close[worst_trough_idx]),
        recovery_time_hours=int(recovery_hours) if recovery_hours is not None else None,
        candle_path_csv="",  # filled in by run_pipeline()
        replay_file_path="",  # filled in by run_pipeline()
    )


def build_data_quality_report(df: pd.DataFrame, cleaning_report: dict) -> DataQualityReport:
    """Build a data quality summary."""
    if df.empty:
        return DataQualityReport(
            total_candles=0,
            date_range_start="",
            date_range_end="",
            missing_pct=100.0,
            longest_gap_hours=0,
            zero_volume_pct=0.0,
            session_gaps_count=0,
            spread_avg_pips=0.0,
            spread_max_pips=0.0,
            price_min=0.0,
            price_max=0.0,
            price_mean=0.0,
            price_std=0.0,
            notes=["EMPTY DATAFRAME"],
        )

    total = len(df)
    zero_vol = int((df["volume"] == 0).sum())
    zero_vol_pct = (zero_vol / total) * 100

    # Detect gaps
    time_diffs = df["date"].diff().dropna()
    gap_mask = time_diffs > pd.Timedelta(hours=1, minutes=30)
    longest_gap = int(time_diffs.max().total_seconds() / 3600) if not time_diffs.empty else 0
    session_gaps = int(gap_mask.sum())

    # Spread approximation: (high - low) / close * 10000 (in pips)
    spread_bps = ((df["high"] - df["low"]) / df["close"]) * 10000
    spread_avg = float(spread_bps.mean())
    spread_max = float(spread_bps.max())

    notes = []
    if cleaning_report["forward_filled_candles"] > 0:
        notes.append(f"Forward-filled {cleaning_report['forward_filled_candles']} missing candles")
    if cleaning_report["flagged_gaps"]:
        notes.append(f"Detected {len(cleaning_report['flagged_gaps'])} gaps > 90 min")
    if longest_gap > 24:
        notes.append(f"WARNING: longest gap is {longest_gap}h")

    return DataQualityReport(
        total_candles=total,
        date_range_start=df["date"].iloc[0].isoformat(),
        date_range_end=df["date"].iloc[-1].isoformat(),
        missing_pct=0.0,  # after cleaning
        longest_gap_hours=longest_gap,
        zero_volume_pct=zero_vol_pct,
        session_gaps_count=session_gaps,
        spread_avg_pips=spread_avg,
        spread_max_pips=spread_max,
        price_min=float(df["close"].min()),
        price_max=float(df["close"].max()),
        price_mean=float(df["close"].mean()),
        price_std=float(df["close"].std()),
        notes=notes,
    )


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    raw_path: Path | str,
    processed_dir: Path | str,
    metadata_dir: Path | str,
) -> dict:
    """Run the full Stage 2 pipeline.

    Args:
        raw_path: input feather file (Freqtrade format)
        processed_dir: where to write btc_h1_5y.feather
        metadata_dir: where to write metadata JSONs

    Returns:
        dict with all generated paths + summary metrics
    """
    raw_path = Path(raw_path)
    processed_dir = Path(processed_dir)
    metadata_dir = Path(metadata_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw
    df_raw = load_btc_h1_feather(raw_path)

    # 2. Clean
    df_clean, cleaning_report = clean_candles(df_raw, forward_fill_max_gap=3)

    # 3. Validate 5y coverage
    coverage_years = (df_clean["date"].iloc[-1] - df_clean["date"].iloc[0]).days / 365.25
    if coverage_years < 4.5:
        raise ValueError(
            f"Insufficient coverage: {coverage_years:.2f} years (need ≥ 4.5). "
            f"Range: {df_clean['date'].iloc[0]} to {df_clean['date'].iloc[-1]}"
        )

    # 4. Generate monthly windows
    monthly = generate_monthly_windows(df_clean)

    # 5. Detect worst adverse move
    worst = detect_worst_adverse_move(df_clean)

    # 6. Build data quality report
    quality = build_data_quality_report(df_clean, cleaning_report)

    # 7. Write outputs
    # 7a. Processed feather (Freqtrade-compatible)
    processed_path = processed_dir / "btc_h1_5y.feather"
    df_clean.to_feather(processed_path)

    # 7b. Worst adverse move CSV path (full candle path around the event)
    recovery_window = 168
    peak_idx_recovery = int((df_clean["date"] == pd.Timestamp(worst.start_time)).idxmax())
    trough_idx_recovery = int((df_clean["date"] == pd.Timestamp(worst.end_time)).idxmax())
    end_window = min(len(df_clean), trough_idx_recovery + 1 + recovery_window)
    path_df = df_clean.iloc[peak_idx_recovery:end_window].copy()
    peak_val_at_start = float(df_clean["close"].iloc[peak_idx_recovery])
    path_df["drawdown_pct"] = (peak_val_at_start - path_df["close"]) / peak_val_at_start
    candle_path_path = metadata_dir / "worst_adverse_move_candles.csv"
    path_df.to_csv(candle_path_path, index=False)
    worst.candle_path_csv = str(candle_path_path)
    worst.replay_file_path = str(candle_path_path)

    # 7c. Metadata JSONs
    full_period = {
        "start": df_clean["date"].iloc[0].isoformat(),
        "end": df_clean["date"].iloc[-1].isoformat(),
        "candle_count": len(df_clean),
        "coverage_years": coverage_years,
    }
    (metadata_dir / "monthly_windows.json").write_text(json.dumps(
        [w.to_dict() for w in monthly], indent=2
    ))
    (metadata_dir / "full_period.json").write_text(json.dumps(full_period, indent=2))
    (metadata_dir / "worst_adverse_move.json").write_text(json.dumps(worst.to_dict(), indent=2))
    (metadata_dir / "data_quality_report.json").write_text(json.dumps({
        **quality.to_dict(),
        "cleaning_report": cleaning_report,
        "generated_at": datetime.now(UTC).isoformat(),
    }, indent=2))

    return {
        "processed_path": str(processed_path),
        "candle_path_csv": str(candle_path_path),
        "monthly_windows": len(monthly),
        "total_candles": len(df_clean),
        "coverage_years": coverage_years,
        "worst_adverse_pct": worst.percentage_decline,
    }


def main():
    """CLI entry point for Stage 2 pipeline."""
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "BTC_USDT-1h.feather"
    processed_dir = project_root / "data" / "processed"
    metadata_dir = project_root / "data" / "metadata"
    result = run_pipeline(raw_path, processed_dir, metadata_dir)
    print("Stage 2 pipeline complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
