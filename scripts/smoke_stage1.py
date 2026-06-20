"""Stage 1 end-to-end smoke test — exercises the full config → IDs → metadata pipeline.

This is the script version of the pytest tests. Used to verify Stage 1 deliverable
end-to-end before requesting Gate 1 review.

Run:
    /home/spatula/freqtrade/.venv/bin/python scripts/smoke_stage1.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import Settings  # noqa: E402
from configs.ids import (  # noqa: E402
    is_valid_candidate_id,
    is_valid_experiment_id,
    is_valid_genome_id,
    make_candidate_id,
    make_experiment_id,
    make_genome_id,
    make_run_metadata_id,
)
from configs.metadata import make_run_metadata, write_run_metadata  # noqa: E402


def main() -> int:
    print("=" * 70)
    print("STAGE 1 SMOKE TEST — config → IDs → metadata end-to-end")
    print("=" * 70)

    # 1. Load Settings
    print("\n[1] Loading Settings...")
    s = Settings.from_files(
        freqtrade_path=PROJECT_ROOT / "configs" / "freqtrade" / "config.json",
        experiment_path=PROJECT_ROOT / "configs" / "experiments" / "default.json",
        research_path=PROJECT_ROOT / "configs" / "research" / "default.json",
    )
    print(f"    OK — leverage={s.leverage}, mode={s.margin_mode}, dry_run={s.dry_run}")
    print(f"    OK — candidates/gen={s.candidates_per_gen}, parallel={s.parallel_workers}")
    print(f"    OK — buffer_pct={s.buffer_pct} (configurable), wall_time_budget={s.wall_time_budget_seconds}s")
    print(f"    OK — settings has {len(s)} keys, time.timeframe={s.timeframe}")

    # 2. Generate IDs
    print("\n[2] Generating IDs...")
    eid = make_experiment_id(generation=0, slug="stage1_smoke")
    assert is_valid_experiment_id(eid), f"experiment_id invalid: {eid}"
    print(f"    experiment_id: {eid}")

    cid = make_candidate_id(eid, candidate_index=0)
    assert is_valid_candidate_id(cid), f"candidate_id invalid: {cid}"
    print(f"    candidate_id:  {cid}")

    gid = make_genome_id()
    assert is_valid_genome_id(gid), f"genome_id invalid: {gid}"
    print(f"    genome_id:     {gid}")

    rid = make_run_metadata_id(eid, run_index=1, attempt_index=1)
    print(f"    run_id:        {rid}")

    # 3. Create + write run metadata
    print("\n[3] Creating + writing run metadata...")
    meta = make_run_metadata(
        experiment_id=eid,
        run_index=1,
        attempt_index=1,
        settings=s,
        candidate_id=cid,
        genome_id=gid,
    )
    meta.exit_reason = "success"
    meta.finished_at = "2026-06-20T19:50:00.000000Z"
    meta.safety_pass = {"passed": True, "reasons": [], "buffer_breach_count": 0}
    meta.extra["stage1_smoke_note"] = "Stage 1 acceptance smoke — passes if this script completes"

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "smoke_meta.json"
        written = write_run_metadata(meta, path=out_path)
        print(f"    OK — wrote {written}")
        # Verify it loads back
        loaded = json.loads(written.read_text())
        assert loaded["experiment_id"] == eid
        assert loaded["candidate_id"] == cid
        assert loaded["genome_id"] == gid
        assert loaded["exit_reason"] == "success"
        assert loaded["settings_snapshot_inline"]["leverage"] == 5.0
        print(f"    OK — round-trip verified, {len(loaded)} top-level keys")

    # 4. Verify Freqtrade loads our config
    print("\n[4] Verifying Freqtrade can read configs/freqtrade/config.json...")
    result = subprocess.run(
        [
            "/home/spatula/freqtrade/.venv/bin/freqtrade",
            "show-config",
            "--config", str(PROJECT_ROOT / "configs" / "freqtrade" / "config.json"),
            "--userdir", str(PROJECT_ROOT / "data"),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"    FAILED — freqtrade show-config exit {result.returncode}")
        print(f"    stderr: {result.stderr[-500:]}")
        return 1
    # Freqtrade prefixes output with "Your combined configuration is:" — strip it
    stdout = result.stdout.strip()
    if stdout.startswith("Your combined configuration is:"):
        stdout = stdout.split(":", 1)[1].strip()
    ft_loaded = json.loads(stdout)
    assert ft_loaded["timeframe"] == "1h"
    assert ft_loaded["dry_run"] is True
    assert ft_loaded["leverage"] == 5.0
    assert "BTC/USDT" in ft_loaded["exchange"]["pair_whitelist"]
    print(f"    OK — Freqtrade loaded config, leverage={ft_loaded['leverage']}, dry_run={ft_loaded['dry_run']}")

    # 5. Print summary
    print("\n" + "=" * 70)
    print("STAGE 1 SMOKE PASSED")
    print("=" * 70)
    print(f"  Settings loaded:    YES ({len(s)} keys)")
    print(f"  IDs generated:      experiment={eid[:40]}...")
    print("  Metadata roundtrip: YES")
    print(f"  Freqtrade compat:   YES (timeframe={ft_loaded['timeframe']}, leverage={ft_loaded['leverage']})")
    print()
    print("READY FOR GATE 1 REVIEW")
    return 0


if __name__ == "__main__":
    sys.exit(main())
