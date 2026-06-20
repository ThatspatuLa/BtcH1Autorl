"""Backtest runner — top-level entry point that takes a candidate config and produces a BacktestResult.

Stage 3 skeleton: takes Settings + a simple grid_pct/tp_pct pair, runs the state machine,
returns equity_curve + trades_df in Stage 5-compatible shape.

Stage 8 will replace the simple params with full CandidateGenome → OrderManager conversion.
Stage 9 (TP baseline) lives here as a fixed_pct TP option.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dca_engine.indicators import compute_indicators
from dca_engine.order_manager import OrderManager
from dca_engine.state_machine import StateMachineResult, run_state_machine


@dataclass
class BacktestResult:
    """Final output of one candidate backtest — feeds into Stage 5 reward engine."""
    candidate_id: str
    genome_id: str
    experiment_id: str
    equity_curve: pd.Series
    trades_df: pd.DataFrame
    n_cycles_opened: int
    n_cycles_closed: int
    final_equity: float
    peak_equity: float
    trough_equity: float
    gross_exposure_series: list[float] = field(default_factory=list)
    free_margin_series: list[float] = field(default_factory=list)
    backtest_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Convert Timestamp keys to ISO strings for JSON serialisation
        eq_dict = {str(k): float(v) for k, v in self.equity_curve.to_dict().items()}
        trades_records = []
        for _, row in self.trades_df.iterrows():
            trades_records.append({
                k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.to_dict().items()
            })
        return {
            "candidate_id": self.candidate_id,
            "genome_id": self.genome_id,
            "experiment_id": self.experiment_id,
            "equity_curve": eq_dict,
            "trades_df": trades_records,
            "n_cycles_opened": self.n_cycles_opened,
            "n_cycles_closed": self.n_cycles_closed,
            "final_equity": self.final_equity,
            "peak_equity": self.peak_equity,
            "trough_equity": self.trough_equity,
            "backtest_meta": self.backtest_meta,
        }


def backtest_candidate(
    df: pd.DataFrame,
    candidate_id: str,
    genome_id: str,
    experiment_id: str,
    grid_pct: float = 0.015,
    tp_pct: float = 0.02,
    max_layers: int = 3,
    initial_deposit: float = 10000.0,
    stake_amount: float = 100.0,
    fee_pct: float = 0.001,
    symbol: str = "BTC/USDT",
    confirmation_indicators: list[str] | None = None,
    indicator_params: dict[str, dict[str, float]] | None = None,
    rsi_period: int = 14,
    ma_period: int = 200,
    vol_period: int = 20,
    ref_vol_period: int = 100,
) -> BacktestResult:
    """Run one candidate backtest through the OHLCV dataframe.

    Stage 3 placeholder: simple grid_pct + fixed TP. Stage 8 wires confirmations.
    """
    order_manager = OrderManager(
        grid_pct=grid_pct,
        tp_pct=tp_pct,
        max_layers=max_layers,
        symbol=symbol,
        confirmation_indicators=confirmation_indicators,
        indicator_params=indicator_params,
    )

    # Pre-compute indicators if confirmations are active
    indicators = None
    if confirmation_indicators:
        indicators = compute_indicators(
            df,
            rsi_period=rsi_period,
            ma_period=ma_period,
            vol_period=vol_period,
            ref_vol_period=ref_vol_period,
        )

    sm_result: StateMachineResult = run_state_machine(
        df=df,
        order_manager=order_manager,
        initial_deposit=initial_deposit,
        stake_amount=stake_amount,
        fee_pct=fee_pct,
        indicators=indicators,
    )

    trades_df = pd.DataFrame(sm_result.trades) if sm_result.trades else pd.DataFrame(
        columns=["open_time", "close_time", "cycle_id", "symbol", "qty", "avg_entry", "exit_price", "pnl", "close_reason", "n_layers"]
    )

    return BacktestResult(
        candidate_id=candidate_id,
        genome_id=genome_id,
        experiment_id=experiment_id,
        equity_curve=sm_result.equity_curve,
        trades_df=trades_df,
        n_cycles_opened=sm_result.n_cycles_opened,
        n_cycles_closed=sm_result.n_cycles_closed,
        final_equity=sm_result.final_equity,
        peak_equity=sm_result.peak_equity,
        trough_equity=sm_result.trough_equity,
        gross_exposure_series=sm_result.gross_exposure,
        free_margin_series=sm_result.free_margin,
        backtest_meta={
            "grid_pct": grid_pct,
            "tp_pct": tp_pct,
            "max_layers": max_layers,
            "initial_deposit": initial_deposit,
            "stake_amount": stake_amount,
            "fee_pct": fee_pct,
            "symbol": symbol,
            "stage": 3,
            "is_placeholder_sizing": True,
        },
    )
