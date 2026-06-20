"""Stage 3 end-to-end smoke test.

Runs one backtest with the placeholder sizing against real BTC H1 5y data,
feeds the result to Stage 5 reward engine, prints the score breakdown.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from dca_engine import backtest_candidate  # noqa: E402
from reward import ScoreResult, compute_score  # noqa: E402


def main() -> int:
    feather = PROJECT_ROOT / "data" / "processed" / "btc_h1_5y.feather"
    if not feather.exists():
        print(f"FAIL: {feather} does not exist. Run `python -m data.pipeline.loader` first.")
        return 1
    df = pd.read_feather(feather)
    print(f"[1/4] Loaded {len(df)} candles from {feather.name}")

    print("[2/4] Running 1 backtest (placeholder sizing: grid_pct=1.5%, tp_pct=2%, max_layers=3)...")
    t0 = time.time()
    result = backtest_candidate(
        df=df,
        candidate_id="stage3_smoke_001",
        genome_id="11111111-1111-4111-8111-111111111111",
        experiment_id="20210601_000000_gen0_stage3_smoke",
        grid_pct=0.015,
        tp_pct=0.02,
        max_layers=3,
        initial_deposit=10000.0,
        stake_amount=100.0,
    )
    elapsed = time.time() - t0
    print(f"       Completed in {elapsed:.2f}s")
    print(f"       Cycles opened: {result.n_cycles_opened}, closed: {result.n_cycles_closed}")
    print(f"       Final equity: ${result.final_equity:.2f}, peak: ${result.peak_equity:.2f}, trough: ${result.trough_equity:.2f}")
    print(f"       Trades: {len(result.trades_df)}")

    print("[3/4] Feeding result to Stage 5 reward engine...")
    score = compute_score(result.equity_curve, result.trades_df, candidate_id="stage3_smoke_001")
    if isinstance(score, ScoreResult):
        print(f"       SCORED — final_score={score.breakdown.final_score:.4f}")
        print(f"         profit: {score.breakdown.profit.contribution:.4f}")
        print(f"         dd_quality: {score.breakdown.dd_quality.contribution:.4f}")
        print(f"         sharpe: {score.breakdown.sharpe.contribution:.4f}")
        print(f"         profit_factor: {score.breakdown.profit_factor.contribution:.4f}")
        print(f"         tpm: {score.breakdown.tpm.contribution:.4f}")
        print(f"         dd_penalty_multiplier: {score.breakdown.dd_penalty_multiplier}")
        print(f"         total_trades: {score.total_trades}, months: {score.months_active:.2f}")
    else:
        print(f"       REJECTED — reason={score.reason}")
        print(f"         net_profit_pct={score.raw_metrics['net_profit_pct']:.3f}")
        print(f"         max_drawdown_pct={score.raw_metrics['max_drawdown_pct']:.3f}")
        print(f"         trades_per_month={score.raw_metrics['trades_per_month']:.1f}")

    print("[4/4] Verifying JSON serialisation...")
    d = result.to_dict()
    j = json.dumps(d, default=str)
    parsed = json.loads(j)
    print(f"       Serialised {len(j)} bytes, {len(parsed['trades_df'])} trade records")

    print("\nStage 3 smoke test: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
