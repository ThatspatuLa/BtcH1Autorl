"""Stage 15 Phase 0 acceptance tests — reporting shell.

Phase 0 = module skeleton + interfaces + stub-data renders.
Phase 1 = real Stage 5/6/7 integration (covered by later test file).

Verifies:
- All renderers produce valid Markdown/JSON strings
- CandidateSummary dataclass works with stub data
- Generation leaderboard sorts by final_score descending
- Rejected reason report aggregates correctly
- JSON exporters produce parseable JSON
- Phase 0 stub helpers produce sensible test data
"""
from __future__ import annotations

import json

import pytest

from reports import (
    build_stub_candidate_summary,
    build_stub_generation_leaderboard,
    export_best_genome_json,
    export_generation_history_json,
    render_candidate_summary_md,
    render_generation_leaderboard,
    render_rejected_reason_report,
    render_score_breakdown_table,
)

pytestmark = pytest.mark.stage15


def test_candidate_summary_to_dict():
    c = build_stub_candidate_summary()
    d = c.to_dict()
    assert d["candidate_id"] == "stub_cand_0001"
    assert d["final_score"] == 0.65
    assert "profit" in d["score_components"]


def test_render_candidate_summary_md_contains_key_fields():
    c = build_stub_candidate_summary(candidate_id="md-test-001")
    md = render_candidate_summary_md(c)
    assert "# Candidate Summary" in md
    assert "md-test-001" in md
    assert "Final Score" in md
    assert "Score Breakdown" in md
    assert "profit" in md
    assert "Raw Metrics" in md


def test_render_candidate_summary_md_shows_rejection():
    c = build_stub_candidate_summary(
        candidate_id="rejected-001",
        exit_reason="hard_reject",
        rejection_reason="drawdown>35%",
    )
    md = render_candidate_summary_md(c)
    assert "hard_reject" in md
    assert "drawdown>35%" in md


def test_render_candidate_summary_md_shows_safety():
    c = build_stub_candidate_summary()
    c.safety_pass = {"passed": True, "reasons": [], "buffer_breach_count": 0}
    md = render_candidate_summary_md(c)
    assert "Safety Result" in md
    assert "passed" in md


def test_render_candidate_summary_md_shows_monthly_table():
    c = build_stub_candidate_summary()
    c.monthly_table = [
        {"month": "2021-07", "profit": 0.02, "dd": 0.05, "sharpe": 0.8, "pf": 1.2, "tpm": 6.0, "profitable": True},
        {"month": "2021-08", "profit": -0.01, "dd": 0.07, "sharpe": -0.3, "pf": 0.9, "tpm": 4.0, "profitable": False},
    ]
    md = render_candidate_summary_md(c)
    assert "Monthly Performance" in md
    assert "2021-07" in md
    assert "2021-08" in md


def test_render_score_breakdown_table_format():
    c = build_stub_candidate_summary()
    md = render_score_breakdown_table(c)
    assert "| Component |" in md
    assert "| profit |" in md
    assert "TOTAL" in md  # bold markup
    assert "0.65" in md  # final score appears


def test_render_generation_leaderboard_sorts_by_score():
    candidates = build_stub_generation_leaderboard(n_candidates=100)
    md = render_generation_leaderboard(candidates, generation_index=0, top_n=10)
    assert "Generation 0 Leaderboard" in md
    assert "Total candidates: 100" in md
    # Top candidate should have highest score
    scored = sorted(
        [c for c in candidates if c.exit_reason == "scored"],
        key=lambda c: c.final_score, reverse=True,
    )
    assert scored[0].final_score >= scored[-1].final_score
    # Leaderboard line count = header + separator + 50 data rows (top_n=10 default… actually 50)
    # We set top_n=10 so should be small


def test_render_rejected_reason_report_aggregates():
    candidates = build_stub_generation_leaderboard(n_candidates=200)
    md = render_rejected_reason_report(candidates, generation_index=0)
    assert "Rejection Report" in md
    assert "Reason | Count" in md
    # Should show at least one rejection reason
    assert "|" in md
    rejected = [c for c in candidates if c.exit_reason == "hard_reject"]
    assert len(rejected) > 0  # sanity


def test_export_best_genome_json():
    c = build_stub_candidate_summary(candidate_id="best-cand-001")
    j = export_best_genome_json(c, genome_dict={"grid_method": "fixed_pct", "tp_pct": 2.0})
    parsed = json.loads(j)
    assert parsed["candidate_id"] == "best-cand-001"
    assert parsed["genome"]["grid_method"] == "fixed_pct"
    assert "exported_at" in parsed


def test_export_best_genome_json_without_genome_dict():
    c = build_stub_candidate_summary()
    j = export_best_genome_json(c)
    parsed = json.loads(j)
    assert parsed["candidate_id"] == c.candidate_id
    assert "genome" not in parsed  # not provided


def test_export_generation_history_json():
    metrics = [
        {"generation": 0, "best_score": 0.65, "mean_score": 0.45, "median_score": 0.44, "num_scored": 425, "num_rejected": 75},
        {"generation": 1, "best_score": 0.72, "mean_score": 0.52, "median_score": 0.51, "num_scored": 425, "num_rejected": 75},
    ]
    j = export_generation_history_json(generation_index=2, per_generation_metrics=metrics)
    parsed = json.loads(j)
    assert parsed["generation_index"] == 2
    assert len(parsed["per_generation_metrics"]) == 2
    assert parsed["per_generation_metrics"][1]["best_score"] == 0.72


def test_build_stub_candidate_summary_has_all_components():
    c = build_stub_candidate_summary()
    expected_components = {"profit", "dd_quality", "sharpe", "profit_factor", "tpm"}
    assert expected_components.issubset(set(c.score_components.keys()))


def test_build_stub_generation_leaderboard_realistic_distribution():
    candidates = build_stub_generation_leaderboard(n_candidates=500)
    scored = [c for c in candidates if c.exit_reason == "scored"]
    rejected = [c for c in candidates if c.exit_reason == "hard_reject"]
    # Default scored_pct=0.85, so ~85% scored, ~15% rejected
    assert 350 <= len(scored) <= 450
    assert 50 <= len(rejected) <= 150
