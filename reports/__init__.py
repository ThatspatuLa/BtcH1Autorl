"""BTC H1 AutoRL — reports subpackage.

Stage 15 Phase 0 (reporting shell): module skeleton + interfaces + stub-data renders.
Stage 15 Phase 1 (minimum, unblocks Stage 10): real reports consuming Stage 5/6/7 outputs.
Stage 15 Phase 2 (post-evolution polish): PNG charts — does NOT block Stage 10.

Phase 0 is built early (after Stage 1 + Stage 5 interface draft) using synthetic stub data.
Phase 1 integration with real Stage 5/6/7 outputs is required before Stage 10 begins.
"""
from reports.render import (
    # Phase 0 interfaces + renders
    CandidateSummary,
    # Phase 0 stub-data helpers (used by tests)
    build_stub_candidate_summary,
    build_stub_generation_leaderboard,
    export_best_genome_json,
    export_generation_history_json,
    render_candidate_summary_md,
    render_generation_leaderboard,
    render_rejected_reason_report,
    render_score_breakdown_table,
)

__all__ = [
    "CandidateSummary",
    "build_stub_candidate_summary",
    "build_stub_generation_leaderboard",
    "export_best_genome_json",
    "export_generation_history_json",
    "render_candidate_summary_md",
    "render_generation_leaderboard",
    "render_rejected_reason_report",
    "render_score_breakdown_table",
]
