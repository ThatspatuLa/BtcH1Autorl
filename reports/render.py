"""Report renderers — Phase 0 (synthetic stub-data) + Phase 1 (real Stage 5/6/7 outputs).

This module defines data classes for report content + renderer functions that produce
Markdown / JSON output. Phase 0 stubs work with hand-built CandidateSummary objects;
Phase 1 will replace stub helpers with real converters from ScoreResult / FitnessResult / SafetyResult.

Each renderer is a pure function: (data) → string. File-writing is the caller's responsibility.

Output formats:
- candidate_summary.md: per-candidate Markdown report
- score_breakdown_table.md: per-candidate score component table (Markdown)
- generation_leaderboard.md: per-generation leaderboard table (Markdown)
- rejected_reasons.md: per-generation rejection reason breakdown (Markdown)
- best_genome.json: best genome from generation (JSON)
- generation_history.json: per-generation metrics (JSON)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

# ============================================================
# Data classes
# ============================================================

@dataclass
class CandidateSummary:
    """All data needed to render a per-candidate summary."""
    candidate_id: str
    experiment_id: str
    genome_id: str
    final_score: float
    base_score: float
    dd_penalty_multiplier: float
    score_components: dict[str, dict[str, float]]  # component_name → {raw, normalised, weight, contribution}
    raw_metrics: dict[str, float]
    total_trades: int
    months_active: float
    exit_reason: str  # "scored", "hard_reject", "safety_fail", "error", "timeout"
    rejection_reason: str | None = None
    safety_pass: dict[str, Any] | None = None
    monthly_table: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Renderers — return strings (caller writes to file)
# ============================================================

def render_candidate_summary_md(c: CandidateSummary) -> str:
    """Render per-candidate Markdown summary."""
    lines = [
        f"# Candidate Summary — {c.candidate_id}",
        "",
        f"- **Experiment:** `{c.experiment_id}`",
        f"- **Genome ID:** `{c.genome_id}`",
        f"- **Exit Reason:** `{c.exit_reason}`",
        f"- **Final Score:** `{c.final_score:.6f}`",
        f"- **Base Score:** `{c.base_score:.6f}`",
        f"- **DD Penalty Multiplier:** `{c.dd_penalty_multiplier:.4f}`",
        f"- **Total Trades:** `{c.total_trades}`",
        f"- **Months Active:** `{c.months_active:.2f}`",
        "",
        "## Score Breakdown",
        "",
    ]
    for name, comp in c.score_components.items():
        lines.append(
            f"- **{name}**: raw=`{comp['raw_value']:.4f}`, "
            f"normalised=`{comp['normalised']:.4f}`, "
            f"weight=`{comp['weight']:.2f}`, "
            f"contribution=`{comp['contribution']:.6f}`"
        )
    lines.append("")
    lines.append("## Raw Metrics")
    lines.append("")
    for k, v in sorted(c.raw_metrics.items()):
        lines.append(f"- {k}: `{v}`")
    if c.rejection_reason:
        lines.append("")
        lines.append(f"## Rejection: {c.rejection_reason}")
    if c.safety_pass:
        lines.append("")
        lines.append("## Safety Result")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(c.safety_pass, indent=2))
        lines.append("```")
    if c.monthly_table:
        lines.append("")
        lines.append("## Monthly Performance")
        lines.append("")
        lines.append("| Month | Profit | DD | Sharpe | PF | TPM | Profitable |")
        lines.append("|-------|--------|-----|--------|-----|------|------------|")
        for row in c.monthly_table:
            lines.append(
                f"| {row.get('month', '')} "
                f"| {row.get('profit', 0):.4f} "
                f"| {row.get('dd', 0):.4f} "
                f"| {row.get('sharpe', 0):.4f} "
                f"| {row.get('pf', 0):.4f} "
                f"| {row.get('tpm', 0):.4f} "
                f"| {row.get('profitable', False)} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_score_breakdown_table(c: CandidateSummary) -> str:
    """Render just the score breakdown table — concise Markdown."""
    lines = [
        "| Component | Raw | Normalised | Weight | Contribution |",
        "|-----------|-----|------------|--------|--------------|",
    ]
    for name, comp in c.score_components.items():
        lines.append(
            f"| {name} "
            f"| {comp['raw_value']:.4f} "
            f"| {comp['normalised']:.4f} "
            f"| {comp['weight']:.2f} "
            f"| {comp['contribution']:.6f} |"
        )
    lines.append(
        f"| **TOTAL** | | | | **{c.final_score:.6f}** (× DD penalty {c.dd_penalty_multiplier:.4f}) |"
    )
    return "\n".join(lines)


def render_generation_leaderboard(
    candidates: list[CandidateSummary],
    generation_index: int,
    top_n: int = 50,
) -> str:
    """Render per-generation leaderboard Markdown table."""
    scored = [c for c in candidates if c.exit_reason == "scored"]
    scored_sorted = sorted(scored, key=lambda c: c.final_score, reverse=True)[:top_n]
    lines = [
        f"# Generation {generation_index} Leaderboard",
        "",
        f"Total candidates: {len(candidates)}, scored: {len(scored)}, shown: {len(scored_sorted)}",
        "",
        "| Rank | Candidate | Genome | Score | Trades | TPM | Max DD |",
        "|------|-----------|--------|-------|--------|-----|--------|",
    ]
    for rank, c in enumerate(scored_sorted, start=1):
        lines.append(
            f"| {rank} "
            f"| `{c.candidate_id}` "
            f"| `{c.genome_id[:8]}...` "
            f"| {c.final_score:.6f} "
            f"| {c.total_trades} "
            f"| {c.raw_metrics.get('trades_per_month', 0):.2f} "
            f"| {c.raw_metrics.get('max_drawdown_pct', 0)*100:.1f}% |"
        )
    return "\n".join(lines)


def render_rejected_reason_report(
    candidates: list[CandidateSummary],
    generation_index: int,
) -> str:
    """Aggregate rejection reasons for the generation."""
    rejected = [c for c in candidates if c.exit_reason in ("hard_reject", "safety_fail")]
    reasons: dict[str, int] = {}
    for c in rejected:
        key = c.rejection_reason or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    lines = [
        f"# Generation {generation_index} Rejection Report",
        "",
        f"Total candidates: {len(candidates)}, rejected: {len(rejected)}",
        f"Rejection rate: {len(rejected) / max(1, len(candidates)) * 100:.1f}%",
        "",
        "| Reason | Count | % |",
        "|--------|-------|---|",
    ]
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / max(1, len(rejected)) * 100
        lines.append(f"| `{reason}` | {count} | {pct:.1f}% |")
    return "\n".join(lines)


# ============================================================
# JSON exporters
# ============================================================

def export_best_genome_json(c: CandidateSummary, genome_dict: dict[str, Any] | None = None) -> str:
    """Export the best candidate's genome as JSON.

    If genome_dict is not provided, the export contains only the candidate metadata —
    the caller is expected to inject the actual genome from the CandidateGenome dataclass.
    """
    payload: dict[str, Any] = {
        "candidate_id": c.candidate_id,
        "genome_id": c.genome_id,
        "experiment_id": c.experiment_id,
        "final_score": c.final_score,
        "raw_metrics": c.raw_metrics,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    if genome_dict is not None:
        payload["genome"] = genome_dict
    return json.dumps(payload, indent=2, default=str)


def export_generation_history_json(
    generation_index: int,
    per_generation_metrics: list[dict[str, float]],
) -> str:
    """Export per-generation history as JSON.

    per_generation_metrics: list of {generation, best_score, mean_score, median_score, num_scored, num_rejected}
    """
    return json.dumps(
        {
            "generation_index": generation_index,
            "per_generation_metrics": per_generation_metrics,
            "exported_at": datetime.now(UTC).isoformat(),
        },
        indent=2,
        default=str,
    )


# ============================================================
# Stub-data builders (used by Phase 0 tests; replaced by Phase 1 converters)
# ============================================================

def build_stub_candidate_summary(
    candidate_id: str = "stub_cand_0001",
    final_score: float = 0.65,
    base_score: float = 0.72,
    dd_penalty_multiplier: float = 1.0,
    exit_reason: str = "scored",
    rejection_reason: str | None = None,
) -> CandidateSummary:
    """Build a CandidateSummary with hand-crafted stub data — used by Phase 0 tests."""
    return CandidateSummary(
        candidate_id=candidate_id,
        experiment_id="stub_experiment_gen0",
        genome_id="00000000-0000-4000-8000-000000000001",
        final_score=final_score,
        base_score=base_score,
        dd_penalty_multiplier=dd_penalty_multiplier,
        score_components={
            "profit": {"raw_value": 0.50, "normalised": 0.70, "weight": 0.55, "contribution": 0.385},
            "dd_quality": {"raw_value": 0.10, "normalised": 0.85, "weight": 0.15, "contribution": 0.1275},
            "sharpe": {"raw_value": 1.2, "normalised": 0.77, "weight": 0.10, "contribution": 0.077},
            "profit_factor": {"raw_value": 1.8, "normalised": 0.70, "weight": 0.10, "contribution": 0.070},
            "tpm": {"raw_value": 12.0, "normalised": 0.59, "weight": 0.10, "contribution": 0.059},
        },
        raw_metrics={
            "net_profit_pct": 0.50,
            "max_drawdown_pct": 0.10,
            "sharpe": 1.2,
            "profit_factor": 1.8,
            "trades_per_month": 12.0,
            "total_trades": 720,
            "months_active": 60.0,
        },
        total_trades=720,
        months_active=60.0,
        exit_reason=exit_reason,
        rejection_reason=rejection_reason,
    )


def build_stub_generation_leaderboard(n_candidates: int = 500, scored_pct: float = 0.85) -> list[CandidateSummary]:
    """Build n candidates with realistic score distribution for Phase 0 testing."""
    import random
    rng = random.Random(42)
    candidates = []
    for i in range(n_candidates):
        is_rejected = rng.random() > scored_pct
        if is_rejected:
            reason = rng.choice(["net_profit<=0", "drawdown>35%", "tpm<5", "too_few_trades"])
            c = build_stub_candidate_summary(
                candidate_id=f"stub_cand_{i:04d}",
                exit_reason="hard_reject",
                rejection_reason=reason,
                final_score=0.0,
            )
        else:
            score = rng.uniform(0.3, 0.9)
            c = build_stub_candidate_summary(
                candidate_id=f"stub_cand_{i:04d}",
                final_score=score,
                base_score=score,
            )
        candidates.append(c)
    return candidates
