from __future__ import annotations

import random
from pathlib import Path

from evolution.family_contracts import (
    clear_active_family_contract,
    set_active_family_contract,
)
from evolution.hyperopt_config import build_family_specs
from evolution.operators import ALL_GRID_METHODS, crossover, mutate, random_candidate_genome
from genome.schema import ConfirmationIndicator, GridMethod
import scripts.minato_stage1_queue_runner as queue_runner


def test_stage1_families_are_spacing_only_and_executable() -> None:
    specs = build_family_specs()
    names = [spec.name for spec in specs]

    assert len(specs) >= 8
    assert not any(name.startswith(("alloc_", "confirm_", "shallow_", "medium_", "deep_")) for name in names)

    executable = set(ALL_GRID_METHODS)
    for spec in specs:
        assert spec.forced_grid_methods
        assert set(spec.forced_grid_methods).issubset(executable)
        assert spec.forced_allocation is None
        assert spec.max_dca_layers_cap is None
        assert spec.forced_confirmations == ()
        assert spec.group in {"spacing", "hybrid_spacing"}


def test_pure_spacing_contract_blocks_grid_and_confirmation_leakage() -> None:
    spec = next(spec for spec in build_family_specs() if spec.name == "atr_spacing")
    rng = random.Random(7)
    parent = random_candidate_genome(
        rng=rng,
        forced_grid_method=GridMethod.RSI_OVERSOLD,
        generation_index=0,
    )
    parent.dca_genome.confirmation_indicators = [ConfirmationIndicator.RSI_BELOW]
    parent.dca_genome.indicator_params = {"rsi_below": {"threshold": 35.0}}

    set_active_family_contract(spec.mutation_contract)
    try:
        child = mutate(parent, rng=random.Random(8), mutation_rate=1.0)
    finally:
        clear_active_family_contract()

    assert child.dca_genome.grid_method == GridMethod.ATR
    assert child.dca_genome.confirmation_indicators == []
    assert child.dca_genome.indicator_params == {}


def test_hybrid_spacing_contract_allows_only_family_methods() -> None:
    spec = next(spec for spec in build_family_specs() if spec.name == "hybrid_atr_drawdown_spacing")
    rng = random.Random(11)
    parent_a = random_candidate_genome(rng=rng, forced_grid_method=GridMethod.Z_SCORE, generation_index=0)
    parent_b = random_candidate_genome(rng=rng, forced_grid_method=GridMethod.MA_DISTANCE, generation_index=0)
    allowed = set(spec.forced_grid_methods)

    set_active_family_contract(spec.mutation_contract)
    try:
        child = crossover(parent_a, parent_b, rng=random.Random(12))
        mutated = mutate(child, rng=random.Random(13), mutation_rate=1.0)
    finally:
        clear_active_family_contract()

    assert child.dca_genome.grid_method in allowed
    assert mutated.dca_genome.grid_method in allowed
    assert child.dca_genome.confirmation_indicators == []
    assert mutated.dca_genome.confirmation_indicators == []


def test_stage1_queue_runner_dry_run_starts_first_pending_family(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "stage1_state.json"
    monkeypatch.setattr(queue_runner, "_family_complete", lambda _family_name: False)

    result = queue_runner.run_queue_once(state_path, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["next_action"] == "would_start_family"
    assert result["current"]["family"] == build_family_specs()[0].name
    assert not state_path.exists()
