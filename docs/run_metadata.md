# Run Metadata Schema

Every candidate backtest produces a `run_metadata.json` file containing full provenance — what was run, with what config, in what environment, and what the outcome was.

## Location

`results/{experiment_id}/{run_id}.meta.json`

Where:
- `experiment_id` = `YYYYMMDD_HHMMSS_gen{N}_{slug}` (UTC timestamp at experiment start)
- `run_id` = `{experiment_id}_run{NNNN}_{NN}` (NNNN = run index, NN = attempt index)

## Schema

```json
{
  "experiment_id": "20260620_194500_gen0_smoke_v1",
  "run_id": "20260620_194500_gen0_smoke_v1_run0001_01",
  "candidate_id": "20260620_194500_gen0_smoke_v1_cand0042",
  "genome_id": "550e8400-e29b-41d4-a716-446655440000",
  "started_at": "2026-06-20T19:45:00.000000Z",
  "finished_at": "2026-06-20T19:45:03.214000Z",
  "exit_reason": "success",
  "git_commit": "abc123def456...",
  "git_dirty": false,
  "python_version": "3.11.15",
  "freqtrade_version": "2026.5.1",
  "settings_snapshot_path": "/home/spatula/Projects/BtcH1Autorl/results/20260620_194500_gen0_smoke_v1/settings_snapshot.json",
  "settings_snapshot_inline": { /* full Settings object as dict */ },
  "input_files": {
    "freqtrade_config": "configs/freqtrade/config.json",
    "experiment_config": "configs/experiments/default.json",
    "research_config": "configs/research/default.json"
  },
  "safety_pass": {
    "passed": true,
    "reasons": [],
    "margin_trajectory_summary": "...",
    "buffer_breach_count": 0
  },
  "extra": { /* per-stage metadata, e.g. mutation_ops, parent_ids */ }
}
```

## Exit Reasons

| Reason | Meaning |
|--------|---------|
| `success` | Candidate completed, scored, not rejected |
| `hard_reject` | Rejected by reward engine hard rules (net≤0, DD>35%, TPM<5) |
| `safety_fail` | DCA completion safety check failed (margin/buffer) |
| `error` | Backtest errored (Python exception, NaN propagation, etc.) |
| `timeout` | Wall-time budget exceeded |

## Determinism Guarantees

Given identical inputs (genome, settings, data), two runs MUST produce identical:
- Score components (deterministic from inputs)
- Trade log timestamps + prices + quantities (deterministic from data file)
- Equity curve values

Wall-clock duration will differ, but `finished_at - started_at` is captured for reproducibility analysis.

## Usage

```python
from configs import Settings, make_run_metadata, write_run_metadata, RunMetadata

settings = Settings.from_files("configs/freqtrade/config.json",
                                "configs/experiments/default.json",
                                "configs/research/default.json")
meta = make_run_metadata(
    experiment_id="20260620_194500_gen0_smoke_v1",
    run_index=1,
    attempt_index=1,
    settings=settings,
    candidate_id="20260620_194500_gen0_smoke_v1_cand0042",
    genome_id="550e8400-e29b-41d4-a716-446655440000",
)
# ... run candidate ...
meta.finished_at = _now_utc()
meta.exit_reason = "success"
write_run_metadata(meta)
```

See `configs/metadata.py` for the dataclass definition.
