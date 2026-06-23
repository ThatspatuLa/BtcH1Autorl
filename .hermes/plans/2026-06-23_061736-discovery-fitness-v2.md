# Discovery Fitness v2 — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill (or implement yourself task-by-task) following test-driven-development strictly.

**Goal:** Replace the Stage 10 `discovery_fitness` formula with a recovery-aware, less-variance-punitive model. Stage 5 Base Score (Profit 0.55 / DD 0.15 / Sharpe 0.10 / PF 0.10 / TPM 0.10) remains the law — it now contributes 60% of discovery_fitness as `full_period_base_score`. The other 40% measures recovery, consistency, stability, and concentration. Deployment gates remain strict.

**Architecture:**
- Add a new module `fitness/recovery_metrics.py` (pure functions) — implements drawdown recovery speed, post-loss bounce rate, equity-high reclaim, cycle recovery health.
- Add `fitness/discovery_fitness.py` — pure aggregator: `discovery_fitness = 0.60·full_period_base_score + 0.20·recovery_score + 0.10·consistency_score + 0.05·stability_score + 0.05·concentration_score`.
- Modify `fitness/monthly_fitness.py`:
  - `MonthlyScore` gains per-month `recovery_subscores` (so reports can show month-level recovery health).
  - `MonthlyFitnessResult` gains new fields: `full_period_base_score`, `recovery_score`, `stability_score`, `concentration_score`, `recovery_breakdown` (the 4 sub-scores), `per_month_recovery` (list).
  - `aggregate_monthly_fitness()` now calls the new aggregator. Keeps the existing `discovery_fitness` field name (signature-compatible) and `base_aggregate_fitness` field for back-compat (now defined as the new weighted aggregate, NOT the old walk-forward blend).
- Stage 5 `reward/scoring.py` is **untouched** (Base Score is still the law).
- Deployment gates in `fitness/deployment_gates.py` are **untouched** (DD≤35%, TPM≥5, trades≥30, consistency≥50%, margin/DCA completion/equity remain hard gates).

**Tech Stack:** Python 3.11, pandas, numpy, dataclasses. No new dependencies.

---

## Locked decisions (Six's spec)

| Component | Weight | Source |
|---|---|---|
| `full_period_base_score` | 0.60 | Stage 5 `compute_score().breakdown.final_score` on full 5y equity |
| `recovery_score` | 0.20 | New — 4 sub-metrics (see below) |
| `consistency_score` | 0.10 | profitable_months / total_months (already exists, just renamed) |
| `stability_score` | 0.05 | Light stddev/CoV of monthly base scores (Sharpe already handles smoothness) |
| `concentration_score` | 0.05 | Penalty for one lucky month carrying the result |
| **Total** | **1.00** | |

### recovery_score sub-components
| Sub-metric | Weight | Definition |
|---|---|---|
| `drawdown_recovery_speed` | 0.40 | For each drawdown event (peak→trough→recovery), measure recovery_time / total_curve_length; average across events; map to [0,1] via `1 - mean(recovery_time_frac)`. Never-recovered events score 0. |
| `post_loss_month_bounce_rate` | 0.30 | For each month with `net_profit_pct < 0`, count whether any of the next 1–3 months is profitable. `rate = bounces / loss_months`. Map to [0,1] linearly (rate=1.0 → 1.0). |
| `equity_high_reclaim_rate` | 0.20 | At end of curve, fraction of historical peaks that were reclaimed. (Pure DCA "made new highs" measure.) |
| `cycle_recovery_health` | 0.10 | Fraction of DCA cycles opened within 30 days after a losing month that closed profitably. Uses `trades_df.cycle_id` + `close_time` + `pnl`. Fallback: 0.5 (neutral) if no losing months. |

### stability_score
`stability_score = 1 - clipped(stddev(monthly_base_scores) / 0.3)` — CoV-light. Clipped at 0..1. Stddev of 0 → 1.0; stddev ≥ 0.3 → 0.0. **Light** because Sharpe already rewards smoothness in the base score.

### concentration_score (penalty)
- Top month contribution to total net profit (best_month / sum_positive_profit). If `top_month_share <= 0.30` → 1.0; `>= 0.70` → 0.0; linear in between.
- We do NOT reward being concentrated — we only penalise it. (Top month = "one lucky month".)

### consistency_score
`consistency_score = consistency_ratio = profitable_months / total_months`. Same definition as current system, just lifted to its own field. Soft penalty at the deployment gate (>= 0.50) stays exactly as is.

---

## Files changed / created

**Create:**
- `fitness/recovery_metrics.py` — pure functions for the 4 recovery sub-metrics
- `fitness/discovery_fitness.py` — `compute_discovery_fitness(...)` aggregator
- `tests/test_recovery_metrics.py` — unit tests for recovery sub-metrics
- `tests/test_discovery_fitness_v2.py` — unit tests for the new aggregator
- `tests/test_monthly_fitness_v2_reporting.py` — verify all new fields surface in `MonthlyFitnessResult.to_dict()` and reports

**Modify:**
- `fitness/monthly_fitness.py`
  - `MonthlyScore` adds: `recovery_subscores: dict[str, float]` (per-month)
  - `MonthlyFitnessResult` adds: `full_period_base_score`, `recovery_score`, `stability_score`, `concentration_score`, `recovery_breakdown: dict[str, float]`, `per_month_recovery: list[dict]`. Existing `discovery_fitness`/`base_aggregate_fitness`/`consistency_ratio`/`consistency_multiplier` fields KEPT for back-compat.
  - `aggregate_monthly_fitness()` calls new aggregator. Signature-compatible (existing callers untouched).
  - `compute_monthly_fitness()` passes through `full_period_score` into the new aggregator.
- `evolution/evaluator.py` — verify `EvaluationResult.to_dict()` still emits `discovery_fitness` and adds the new fields (one-liner addition; no behavior change).
- `scripts/post_cycle_obsidian_update.py` — extend the per-cycle report to include the new metric breakdown (additive; no behavior change).

**DO NOT TOUCH:**
- `reward/scoring.py` (Stage 5 Base Score law is locked)
- `reward/weights.py` (weights are locked)
- `fitness/deployment_gates.py` (deployment gates are locked)
- `dca_engine/` (OrderManager, backtest — they produce the inputs we consume)
- `genome/`, `evolution/` beyond evaluator.py (this is a fitness change, not a GA change)

---

## Task structure

Each task is TDD: write failing test → verify failure → minimal impl → verify pass → commit.

### Phase A0: Stage 5 bug-fixes (BEFORE any v2 work)

**Six's Gate Check found 3 real bugs in `_compute_max_drawdown` + `compute_dd_quality_normalizer` + `_compute_metrics`. Bug fixes are allowed (Stage 5 is locked, but bugs are not "scoring-law changes"). Formula changes need separate approval.**

#### Bug 1: Final DD quality output not clamped to [0, 1]
**Where:** `reward/scoring.py:300-311` (`compute_dd_quality_normalizer`)
**Symptom (verified):** negative `max_dd_pct` → output exceeds 1.0 (e.g. -5% DD → 1.025)
**Fix:** wrap final return in `min(1.0, max(0.0, result))`.
**TDD test:** `test_dd_quality_normalizer_clamps_negative_max_dd` — passes in -0.05 → asserts output ≤ 1.0.

#### Bug 2: `dd_duration_candles` is always `len(equity_curve)`
**Where:** `reward/scoring.py:286` `_compute_metrics` line that returns `"dd_duration_candles": float(len(equity_curve))`.
**Symptom (verified):** The "recovery_ratio" is always `recovery_time / total_curve_length`, NOT the actual drawdown event duration. A 1000-candle curve with a 100-candle drawdown that recovered in 50 candles gets `recovery_ratio = 50/1000 = 0.05` even though recovery was fast *within the event*.
**Fix:** `_compute_max_drawdown` must return a 3-tuple: `(max_drawdown_pct, recovery_time_candles, dd_event_duration_candles)`. `dd_event_duration_candles = recovery_time + (trough_pos - peak_pos)` for recovered events, or `(len(curve) - peak_pos)` for unrecovered events (and mark `recovered=False`). `_compute_metrics` consumes the new tuple.
**TDD test:** `test_dd_duration_uses_event_not_total_curve` — synthetic 1000-candle curve with 100-candle drawdown event → assert `dd_duration_candles ≈ 100`, NOT `≈ 1000`.

#### Bug 3: Unrecovered drawdowns get only a weak penalty
**Where:** `reward/scoring.py:182-205` (`_compute_max_drawdown`) + `:300-311` (formula)
**Symptom (verified):** If drawdown never recovers, `recovery_time = len(curve) - 1`, so `recovery_ratio = (len-1)/dd_duration_candles ≈ 1.0` (because dd_duration_candles was len too — Bug 2 made this worse). A 10% DD that never recovers gets DD-quality 0.50 instead of near 0.0.
**Fix (per Six's spec):**
- `_compute_max_drawdown` returns `recovered: bool` as a fourth element.
- `_compute_metrics` exposes `recovered_drawdown: bool` and `unrecovered_drawdown: bool` (latter is `not recovered`).
- **Default behaviour:** if unrecovered, set `recovery_ratio = 0.0` (don't zero the entire `dd_score` — that would be a formula change requiring separate approval).
- DD-quality output for unrecovered: `0.7 * dd_score + 0.3 * 0.0 = 0.7 * dd_score`. For 10% unrecovered DD: 0.7 × 0.714 = **0.500** (currently 0.5003 — meaningful but not zeroed).
- **Report `unrecovered_drawdown: True`** in metrics, monthly_score, and MonthlyFitnessResult so downstream consumers (deployment report, Obsidian) can see it.
**TDD tests:**
- `test_recovered_drawdown_false_when_never_reclaimed` — synthetic curve that drops 10% and stays there → `recovered=False`.
- `test_dd_quality_unrecovered_uses_zero_recovery_ratio` — same curve → output = 0.7 × dd_score.
- `test_unrecovered_flag_surfaces_in_metrics` — `_compute_metrics` exposes `recovered_drawdown` field.

#### Files A0:
- `reward/scoring.py` — modified (Bug 1, 2, 3 fixes)
- `tests/test_scoring.py` (NEW, if not exists) — TDD tests for the 3 bugs
- `tests/test_monthly_fitness.py` — verify monthly aggregation still works with new tuple shape (signature-compatible update).

---

### Phase A0.5: Stage 5 docstring fixes + sigmoid pinning tests

**Six's Gate Check found docstring/comments that contradict actual sigmoid behaviour. Fix the words, pin the maths.**

#### Profit normaliser (`reward/scoring.py:294-297`)
**Current wrong docstring:** "Sigmoid scaling centred at +50% profit. Below 0 = 0 (hard rejected anyway)."
**Correct docstring:** "Sigmoid scaling centred at 0% profit. Below 0% is penalised (still returns a value, but <0.5); net_profit_pct <= 0 is hard-rejected at Stage 5 (see _check_hard_rejects). At 0% → 0.5; at +50% → 0.68; at +100% → 0.82; at +300% → 0.99."
**Inline comment:** replace "Gentle sigmoid: 0 at -100% loss, ~0.5 at 0% profit, ~0.73 at +100%, ~0.88 at +300%" with "Sigmoid centred at 0% (k=1.5): -50% → 0.32, 0% → 0.50, +50% → 0.68, +100% → 0.82, +300% → 0.99."

#### TPM normaliser (`reward/scoring.py:326-328`)
**Current wrong docstring:** "Saturating curve: TPM 5 → ~0.3, TPM 20 → ~0.7, TPM 40+ → ~0.88."
**Correct docstring:** "Sigmoid centred at TPM=5 (k=1/8): TPM 0 → 0.35, TPM 5 → 0.50 (centre), TPM 10 → 0.65, TPM 20 → 0.87, TPM 40+ → 0.99. The TPM<5 hard reject happens elsewhere (`_check_hard_rejects`); this normaliser only scores TPM."

#### TDD pinning tests
**Files:** `tests/test_scoring.py` (extend)
- `test_profit_normalizer_known_values` — assert exact outputs: -50%→0.3208, 0%→0.5000, +50%→0.6792, +100%→0.8176, +300%→0.9890 (with 4-decimal tolerance).
- `test_tpm_normalizer_known_values` — assert exact outputs: 0→0.3486, 5→0.5000, 10→0.6514, 20→0.8670, 40→0.9876.
- These tests pin the *intended* behaviour (matching current code output). They prevent silent regressions if the sigmoid is later "fixed" to match the docstring (which would be a formula change requiring approval).

---

### Phase A: Pure functions (smallest units)

#### Task A1: `drawdown_recovery_speed` function
**Files:** `fitness/recovery_metrics.py` (create), `tests/test_recovery_metrics.py` (create)
- Write failing test: synthetic equity curve with one drawdown event that recovers → score should be high; one that never recovers → score = 0.
- Implement `drawdown_recovery_speed(equity_curve: pd.Series) -> float`.
- Verify RED → GREEN.

#### Task A2: `post_loss_month_bounce_rate`
**Files:** same
- Write failing test: 5 losing months, 3 of which are followed by a profitable month within 3 months → rate = 0.6.
- Implement.
- Verify RED → GREEN.

#### Task A3: `equity_high_reclaim_rate`
- Write failing test: curve that makes a new high 4 of 5 times → rate = 0.8.
- Implement.

#### Task A4: `cycle_recovery_health`
- Write failing test: trades_df with 10 cycles opened within 30 days after a losing month; 7 closed profitably → 0.7. Empty trades → 0.5 (neutral).
- Implement.

#### Task A5: `recovery_score` aggregator
- Write failing test: weights sub-metrics by 0.40/0.30/0.20/0.10.
- Implement.

### Phase B: New `discovery_fitness` aggregator

#### Task B1: `compute_discovery_fitness` skeleton
**Files:** `fitness/discovery_fitness.py` (create), `tests/test_discovery_fitness_v2.py` (create)
- Write failing test: pass known inputs, verify weights sum to 1.0 and output is in [0,1].
- Implement with `0.60·full_period_base_score + 0.20·recovery_score + 0.10·consistency_score + 0.05·stability_score + 0.05·concentration_score`.

#### Task B2: `stability_score` and `concentration_score` helpers
- Write failing tests for each.
- Implement.

#### Task B3: Integration test — realistic scenario
- Build a synthetic equity curve with one 20% drawdown that recovers in 30 days, 8 of 10 profitable months, CoV=0.2, no single month carries >30% of profit.
- Verify all component scores land in expected ranges.

### Phase C: Wire into `MonthlyFitnessResult`

#### Task C1: Extend `MonthlyScore` with per-month recovery subscores
**Files:** `fitness/monthly_fitness.py`
- Write failing test: compute monthly fitness → result contains `per_month_recovery` list with one entry per month, each entry has 4 sub-scores in [0,1].
- Add field to `MonthlyScore` dataclass and `to_dict()`. Populate in `_score_one_month` (initially 0.0 placeholder).

#### Task C2: Extend `MonthlyFitnessResult` with new fields
- Add `full_period_base_score`, `recovery_score`, `stability_score`, `concentration_score`, `recovery_breakdown` fields. Initialize to 0.0.

#### Task C3: Wire new aggregator into `aggregate_monthly_fitness`
- Replace the current `discovery_fitness = base_aggregate_fitness * consistency_multiplier` formula with: call `compute_discovery_fitness(...)` using the new inputs.
- Update `to_dict()`.
- Verify all existing tests in `test_monthly_fitness.py` still pass (signature-compatible).

#### Task C3.5: EXPLICIT preservation of hard rejects (Six's Gate Check requirement)

The new `discovery_fitness_v2` aggregator MUST NOT replace or weaken the existing hard rejects. They run BEFORE `compute_discovery_fitness()` is called and short-circuit return `rejected=True` with the appropriate `reject_reason`. Specifically:

| Hard reject | Source file | Threshold | Behaviour in v2 |
|---|---|---|---|
| `worst_month_return < -0.50` | `fitness/monthly_fitness.py:485-487` (uses `WALK_FORWARD_V1["min_worst_month_score"] = -0.5`) | worst month score < -0.5 | Hard reject. `rejected=True`, `reject_reason="worst_month<-0.50"`. `discovery_fitness = 0.0`. **STAYS.** |
| `median_month_score < 0.10` | `fitness/monthly_fitness.py:488-490` (uses `WALK_FORWARD_V1["min_median_score"] = 0.10`) | median month score < 0.10 | Hard reject. `rejected=True`, `reject_reason="median<0.10"`. `discovery_fitness = 0.0`. **STAYS.** |
| `consistency_ratio < 0.50` at deployment | `fitness/deployment_gates.py:48, 143-144` | consistency < 0.50 | Deployment gate (file untouched). Hard reject at deployment. `deployment_fitness = 0.0`. **STAYS.** |
| `max_drawdown_pct > 0.35` | `fitness/deployment_gates.py:49, 158-159` | DD > 35% | Deployment gate (file untouched). Hard reject. **STAYS.** |
| `trades_per_month < 5` (deployment) | `fitness/deployment_gates.py:50, 163-164` | TPM < 5 | Deployment gate (file untouched). **STAYS.** |
| `total_trades < 30` | `fitness/deployment_gates.py:51, 165-166` | trades < 30 | Deployment gate (file untouched). **STAYS.** |
| `has_invalid_equity` | `fitness/deployment_gates.py:147-148` | equity NaN/inf | Deployment gate (file untouched). **STAYS.** |
| `has_margin_failure` | `fitness/deployment_gates.py:149-150` | margin < 0 | Deployment gate (file untouched). **STAYS.** |
| `has_dca_completion_failure` | `fitness/deployment_gates.py:151-152` | DCA never closed | Deployment gate (file untouched). **STAYS.** |

**TDD test:** `test_discovery_fitness_v2_preserves_hard_rejects` — pass a candidate whose worst month is -0.6 → assert `rejected=True, reject_reason="worst_month<-0.50", discovery_fitness=0.0`. Repeat for `median=0.05`. Repeat for deployment gates via `compute_deployment_gates(...)` (no direct `discovery_fitness` interaction, but verify file untouched via diff).

**The new aggregator NEVER sees rejected candidates** — they're short-circuited upstream. This is enforced by keeping the rejection check in `aggregate_monthly_fitness` BEFORE the call to `compute_discovery_fitness`.

#### Task C4: Run all existing tests, fix regressions
- `pytest tests/ -q` — full suite must stay green.
- If any test in `test_monthly_fitness.py` or `test_deployment_gates` breaks because expected `discovery_fitness` values changed, update the test (with comment explaining) — the new formula is the new ground truth.

### Phase D: Reporting + smoke

#### Task D1: Extend `EvaluationResult.to_dict()` to include new fields
**Files:** `evolution/evaluator.py`
- Add new fields to `to_dict()` so they flow into leaderboards and reports. No behavior change.

#### Task D2: Extend Obsidian post-cycle report
**Files:** `scripts/post_cycle_obsidian_update.py`
- Add a "Fitness v2 breakdown" section to the per-cycle Markdown showing the 5 component scores + 4 recovery sub-scores for the best candidate.

#### Task D3: Smoke run — run 1 quick cycle, verify metrics surface
- `python3 scripts/run_continuous_evolution.py --output-dir runs/smoke_v2_discofit --wall-time 90 --max-generations 1 --workers 2 --islands 0` (synthetic mode)
- Verify `runs/smoke_v2_discofit/final_status.json` shows new fields populated.

### Phase E: Final verification

#### Task E1: Full test suite green
- `pytest tests/ -q` — all green, no regressions.

#### Task E2: Lint clean
- `ruff check fitness/ tests/ scripts/post_cycle_obsidian_update.py evolution/evaluator.py` (if ruff is configured) OR `python3 -m py_compile` on changed files.

#### Task E3: Commit
- Single commit per phase. Conventional commits. NO PUSH (per Six's GH rule).

---

### Phase F: Island-lineage preservation + retirement wiring (Six's Gate Check requirement)

**This phase addresses the bug Six found: `get_island_id_for_genome()` reads `lineage.mutation_ops` looking for `island_assign` tag, but mutations/crossovers don't preserve it → `per_island_best_fitness` stays empty → retirement never fires. NO island has retired since 2026-06-22.**

#### Task F1: TDD test — seed genome has island_id
**Files:** `tests/test_island_lineage.py` (NEW)
- Build a genome via `build_island_population(island_id=3, ...)`.
- Assert `get_island_id_for_genome(genome) == 3`.
- Assert the genome's `lineage.mutation_ops` contains `{"op": "island_assign", "island_id": 3}`.
- Run RED → fails because either no such op exists OR it's not queryable.

#### Task F2: TDD test — mutation child inherits island_id
- Take a parent genome with island_id=5.
- Call `mutate(parent, rng)` → produce child.
- Assert `get_island_id_for_genome(child) == 5`.
- Assert child's `lineage.mutation_ops` contains both `{"op": "island_assign", "island_id": 5}` (inherited) AND `{"op": "mutate", ...}` (new).
- Run RED → fails because mutation strips the tag.

#### Task F3: TDD test — crossover child inherits primary/hybrid island_id
- Take two parents: A.island_id=2, B.island_id=4.
- Call `crossover(A, B, rng)` → produce child.
- Assert `get_island_id_for_genome(child)` is one of: 2 (primary), 4 (secondary), or a defined hybrid id (e.g. min(2,4) per existing rules).
- Assert child's lineage preserves `island_assign` ops for both parents in correct order.
- Run RED → fails because crossover strips tags.

#### Task F4: TDD test — fresh random genome gets island_id assigned
- Call `build_island_population(island_id=7, random_injection_count=10)`.
- Assert every genome in the output has `get_island_id_for_genome(g) == 7` OR `0` (the documented "random bag" sentinel).
- Document the contract: random bag has `island_id=0` and is explicitly skipped by per-island tracking (matches current `if iid == 0: continue` behaviour).
- Run RED → fails because freshly seeded random genomes don't get tagged.

#### Task F5: TDD test — per_island_best_fitness logs correctly per generation
**Files:** `tests/test_harness_per_island_logging.py` (NEW)
- Run a tiny 2-generation evolution with 8 islands.
- After gen 0, assert `record.per_island_best_fitness` contains entries for islands 1-8 (not 0).
- After gen 1, assert the dict is non-empty for at least one island.
- Run RED → fails (this is the bug Six found: dict is always empty).

#### Task F6: TDD test — retirement only fires when threshold met AND deployment_pass is true
**Files:** `tests/test_retirement_gate.py` (NEW)
- Set up a fake `GenerationRecord` with `per_island_best_fitness = {3: 0.85}` (above 0.80 threshold).
- BUT set deployment_pass=False for the top candidate (e.g. via failed consistency gate).
- Run `_check_retirement(record, candidates, rng)`.
- Assert: NO retirement happens (gate requires both fitness threshold AND deployment pass).
- Repeat with deployment_pass=True → assert retirement fires for island 3.
- Run RED → fails because current code only checks fitness threshold.

#### Task F7: Fix `evolution/population_builder.py` — preserve island tag through mutation/crossover/seeding
- Modify `mutate(...)` to propagate the parent's island_assign op to the child.
- Modify `crossover(...)` to propagate both parents' island_assign ops, then assign hybrid id.
- Modify random-genome seeding to inject `island_assign` op when island_id is provided.
- Verify `get_island_id_for_genome()` returns correct id for all test cases F1-F4.

#### Task F8: Fix `evolution/harness.py:_check_retirement` — require deployment_pass AND threshold
- Modify retirement check to require `top_candidate.deployment_pass == True` in addition to `per_island_best_fitness[iid] >= threshold`.
- This is a SAFETY gate: only archive + re-seed when the candidate is actually deployment-ready. A high-discovery-fitness candidate with broken consistency/TPM shouldn't trigger retirement — the island's "specialty" hasn't been validated.

#### Files F:
- `evolution/population_builder.py` — modify `mutate`, `crossover`, `_seed_island` to preserve `island_assign` lineage op.
- `evolution/harness.py` — modify `_check_retirement` to gate on deployment_pass.
- `tests/test_island_lineage.py` (NEW) — 4 tests (F1-F4).
- `tests/test_harness_per_island_logging.py` (NEW) — 1 test (F5).
- `tests/test_retirement_gate.py` (NEW) — 1 test (F6).
- `evolution/islands.py` — possibly add `hybrid_island_id(parent_a, parent_b)` helper if not already there.

#### Why this matters for v2
Without Phase F, the retirement system is dead code. The new fitness v2 might evolve good candidates that should retire islands — but if `per_island_best_fitness` stays empty, no retirement ever fires, and we waste compute re-running the same biases. Phase F unblocks the original island-retirement design AND keeps it consistent with the v2 deployment-gate philosophy (retire only on proven deployment readiness).

---

## Verification gates (Six will run these)

After execution, before deployment:

1. `pytest tests/ -q` → all green, total count ≥ current 474 + ~40 new tests.
2. Smoke cycle: `python3 scripts/run_continuous_evolution.py --output-dir runs/smoke_v2_discofit --wall-time 90 --max-generations 1 --workers 2 --islands 0` → new metrics present in `final_status.json`.
3. Verify ONE existing backtest (e.g. `tests/test_monthly_fitness.py::test_compute_monthly_fitness_synthetic_profitable`) still passes — confirms signature compatibility.
4. Verify ONE new recovery scenario (`tests/test_recovery_metrics.py::test_realistic_dca_curve`) shows recovery_score > base_score — confirms recovery is rewarded.

---

## Risks & tradeoffs

1. **Breaking signature changes** — `MonthlyFitnessResult` gains fields but keeps the old ones. Old code that only reads `discovery_fitness` still works; new code can read `recovery_score` etc. **Mitigation:** Phase C3 verifies signature compat.

2. **Stage 5 lock-in** — `full_period_base_score` is the existing `final_score` from `reward/scoring.py`. We do NOT replace Stage 5, we consume it. **Mitigation:** `reward/scoring.py` is in the "DO NOT TOUCH" list.

3. **Deployment gates untouched** — Per Six's spec, gates stay strict (DD≤35%, TPM≥5, trades≥30, consistency≥50%, safety). New `discovery_fitness` can only make breeding smarter; final acceptance unchanged. **Mitigation:** `fitness/deployment_gates.py` is in the "DO NOT TOUCH" list.

4. **Stage 5 (reward/scoring.py) — bug-fix vs. formula-change distinction:**
   - **Allowed without further approval (this plan, Phase A0):**
     - Bug 1: clamp final DD-quality output to [0,1] (defensive clamp, no semantic change for in-range inputs).
     - Bug 2: compute actual drawdown event duration instead of `len(equity_curve)`. This corrects a measurement error, not a scoring law. The "recovery_ratio" semantic is preserved (faster recovery = higher score); only the denominator is corrected.
     - Bug 3: expose `recovered_drawdown` flag in metrics + set `recovery_ratio=0` for unrecovered events (Six's spec, does NOT zero `dd_score`).
     - Docstring fixes (A0.5): clarifying comments and pinning tests for profit + TPM sigmoid outputs. The sigmoid code itself is NOT changed — only the words describing it.
   - **Requires separate Gate approval (NOT in this plan):**
     - Changing the `0.7·dd_score + 0.3·recovery_ratio` weights.
     - Zeroing `dd_score` entirely for unrecovered drawdowns (Six explicitly said don't do this yet).
     - Changing the profit or TPM sigmoid `k` parameter.
     - Changing the Stage 5 component weights (locked).
   - The old WALK_FORWARD_V1 walk-forward blend (`base_aggregate_fitness = 0.50·median + 0.20·consistency + 0.15·variance + 0.15·worst_floor`) is being **demoted**, not deleted. Per Six's spec, monthly consistency moves to its own 10% slot, and stability gets only 5%. The old variance_penalty and worst_floor_multiplier are no longer weighted into discovery_fitness. **This is intentional per the spec** ("monthly fitness should not just reward low standard deviation or being close to average"). The old `base_aggregate_fitness` field will be REMOVED from the dataclass; if any downstream code reads it, we update those tests with comments.

5. **Worst-month hard reject stays** — `worst_month < -0.50` still hard-rejects (from WALK_FORWARD_V1). This is the safety net for catastrophic months. NOT in the new formula because it's a hard reject, not a soft score. **Mitigation:** Phase C3 preserves `rejected`/`reject_reason` logic.

6. **`consistency_score` ≠ `consistency_ratio`?** — They are the same value. We keep `consistency_ratio` as the field name (existing) and add `consistency_score` as a back-compat alias (= consistency_ratio). Avoids surprise downstream breaks.

7. **Cron is running** — A live evolution process (PID 787901) is using the OLD `discovery_fitness` formula right now. **Mitigation:** We DO NOT touch the running process. After commit, Six decides when to restart cron with the new code (likely after Gate 4 review).

---

## Open questions for Six

None — the spec is precise and grounded. The "Open questions" above are not blockers, they are documented design decisions I'm flagging for explicit Gate 0 sign-off.

---

## Gate 0 sign-off requested

Six, before I touch any code, I need your explicit Gate 0 approval on:

1. **Phase A0 bug-fixes** (Six's Gate Check finding):
   - Bug 1: clamp final DD-quality output to [0,1] (defensive).
   - Bug 2: compute actual drawdown event duration instead of `len(equity_curve)` (measurement correction).
   - Bug 3: expose `recovered_drawdown` flag + set `recovery_ratio=0` for unrecovered events. **Do NOT zero `dd_score` for unrecovered drawdowns** — that requires separate approval.

2. **Phase A0.5 docstring fixes** (Six's Gate Check finding):
   - Fix profit normaliser docstring/comment to say "centred at 0% profit" (not +50%). Sigmoid code itself unchanged.
   - Fix TPM normaliser docstring to say "centred at TPM=5" (not "→ ~0.3"). Sigmoid code itself unchanged.
   - Add TDD tests pinning the actual intended sigmoid outputs.

3. **Phase A–D: Discovery Fitness v2 formula and integration:**
   - Formula (60/20/10/5/5 weights, sub-weights 0.40/0.30/0.20/0.10).
   - Files in scope (create 5 new, modify 3 existing, leave 4 untouched).
   - Back-compat policy (keep old field names, add new ones; old `base_aggregate_fitness` field is REMOVED because it's superseded).
   - Don't touch Stage 5 beyond A0/A0.5 bug-fixes — Base Score is consumed, not replaced.
   - Don't touch deployment gates — strict gates preserved exactly.
   - Phase C3.5 explicitly preserves all hard rejects (worst_month < -0.50, median < 0.10, deployment gates).

4. **Phase F: Island-lineage + retirement wiring** (Six's Gate Check finding):
   - Preserve `island_assign` mutation_op through seed→mutation→crossover.
   - Add TDD tests F1-F6 proving lineage survives.
   - Modify `_check_retirement` to require `deployment_pass=True` AND fitness threshold (safety gate).
   - This fixes the bug that has prevented retirement since 2026-06-22.

5. **Cron stays on old code until you decide to restart it.**

6. **No push to GitHub** — local commits only.

**Reply "Gate 0 approved" and I'll execute task-by-task.** Or push back on any of the 6 items above.
