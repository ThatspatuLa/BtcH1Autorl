"""Per-island intelligence — learning state for the smart mutator.

Six's directive (2026-06-25): replace blind Gaussian mutation with family-aware,
backtest-informed mutation. Each island tracks its own:
- Niche fingerprint (where this family lives in parameter space)
- Param-fitness correlations (which params matter for THIS family)
- Recent fitness trend (improving / stagnant / declining)
- Backtest patterns (regime performance, drawdown hotspots, recovery speed)

This state is persisted per-generation to disk (island_intelligence/I<N>.json)
so the smart mutator can read it before producing the next generation's children.

Architecture:
1. IslandIntelligenceTracker — collects eval results per gen, builds the state
2. NicheFingerprint — centroid + std of params across top elites
3. ParamCorrelations — Pearson r between each param and fitness
4. BacktestPatterns — regime performance, DD hotspots, recovery speed

The smart_mutator.py reads IslandIntelligence and applies:
- Per-param Gaussian std multipliers (boost promising, dampen saturated)
- Family-specific reasoning hints (from family_reasoning.py)
- Niche fingerprint bias (stay near centroid, with controlled drift)
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evolution.evaluator import EvaluationResult
from evolution.islands import get_island_spec
from evolution.population_builder import get_island_id_for_genome
from genome.schema import CandidateGenome


# ============================================================
# Niche fingerprint — centroid + std of params across top elites
# ============================================================

@dataclass
class NicheFingerprint:
    """Where this island's family lives in parameter space.

    centroid = mean of param values across top-20 elites
    std = std dev of param values across top-20 elites
    spread = max - min of param values (sanity check vs std)
    """
    centroid: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    # Categorical distributions (e.g., grid_method appears X% of the time)
    categorical_dist: dict[str, dict[str, float]] = field(default_factory=dict)
    sample_size: int = 0
    last_updated_gen: int = -1

    def niche_radius(self, param: str, sigma_multiplier: float = 2.5) -> float:
        """Distance from centroid considered 'in-niche' (default 2.5σ).

        Used by SmartMutator to decide whether a mutation stays in-niche
        or is niche-leaving.
        """
        s = self.std.get(param, 0.0)
        return max(s * sigma_multiplier, 1e-6)


# ============================================================
# Param correlations — which params matter for this family
# ============================================================

@dataclass
class ParamCorrelations:
    """Pearson r between each param and discovery_fitness across recent gens.

    saturated = param with low |r| AND low variance (mutation won't help)
    promising = param with high |r| (mutation likely moves fitness)
    """
    correlations: dict[str, float] = field(default_factory=dict)
    variances: dict[str, float] = field(default_factory=dict)
    saturated_params: set[str] = field(default_factory=set)
    promising_params: set[str] = field(default_factory=set)
    last_updated_gen: int = -1

    def classify(self, correlation_threshold: float = 0.15, variance_floor: float = 1e-6):
        """Re-classify saturated vs promising based on current correlations."""
        self.saturated_params = set()
        self.promising_params = set()
        for param, r in self.correlations.items():
            var = self.variances.get(param, 0.0)
            if var < variance_floor:
                # Zero variance = everyone has the same value, no signal
                self.saturated_params.add(param)
            elif abs(r) >= correlation_threshold:
                self.promising_params.add(param)


# ============================================================
# Backtest patterns — read from actual evaluation results
# ============================================================

@dataclass
class BacktestPatterns:
    """Patterns extracted from recent backtest results for this island.

    Used by SmartMutator to reason about regime performance and DD hotspots.
    """
    # Last N gens of best fitness
    recent_best_fitness: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_median_fitness: deque = field(default_factory=lambda: deque(maxlen=20))

    # Regime performance: bull/bear/chop average monthly score
    bull_avg_score: float = 0.0
    bear_avg_score: float = 0.0
    chop_avg_score: float = 0.0
    bull_n_months: int = 0
    bear_n_months: int = 0
    chop_n_months: int = 0

    # Drawdown hotspots — months with worst DD
    worst_dd_months: list[dict[str, Any]] = field(default_factory=list)
    avg_worst_month_dd: float = 0.0

    # Recovery speed — mean trades between drawdown and recovery
    avg_recovery_speed: float = 0.0

    # Trade frequency
    avg_trades_per_month: float = 0.0

    last_updated_gen: int = -1

    def trend(self) -> str:
        """Return 'improving' / 'stagnant' / 'declining' based on recent fitness."""
        if len(self.recent_best_fitness) < 3:
            return "unknown"
        # Use a simple linear regression slope over last N
        n = len(self.recent_best_fitness)
        xs = list(range(n))
        ys = list(self.recent_best_fitness)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den > 0 else 0.0
        if slope > 0.005:
            return "improving"
        elif slope < -0.005:
            return "declining"
        return "stagnant"


# ============================================================
# Main island intelligence tracker
# ============================================================

@dataclass
class IslandIntelligence:
    """All intelligence state for one island. Persisted to disk per gen.

    Used by SmartMutator to bias mutations toward promising params and
    away from saturated ones, with family-specific reasoning layered on top.
    """
    island_id: int
    bias_name: str  # "fixed_pct", "atr", "trend", etc.
    last_updated_gen: int = -1
    niche: NicheFingerprint = field(default_factory=NicheFingerprint)
    correlations: ParamCorrelations = field(default_factory=ParamCorrelations)
    backtest: BacktestPatterns = field(default_factory=BacktestPatterns)
    # Metadata
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # deques need to be converted to lists
        d["backtest"]["recent_best_fitness"] = list(self.backtest.recent_best_fitness)
        d["backtest"]["recent_median_fitness"] = list(self.backtest.recent_median_fitness)
        # sets → lists
        d["correlations"]["saturated_params"] = list(self.correlations.saturated_params)
        d["correlations"]["promising_params"] = list(self.correlations.promising_params)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IslandIntelligence:
        """Hydrate from disk. deque + set fields restored from lists."""
        intel = cls(
            island_id=d["island_id"],
            bias_name=d["bias_name"],
            last_updated_gen=d.get("last_updated_gen", -1),
            created_at=d.get("created_at", time.time()),
        )
        # Niche
        niche_d = d.get("niche", {})
        intel.niche = NicheFingerprint(
            centroid=niche_d.get("centroid", {}),
            std=niche_d.get("std", {}),
            spread=niche_d.get("spread", {}),
            categorical_dist=niche_d.get("categorical_dist", {}),
            sample_size=niche_d.get("sample_size", 0),
            last_updated_gen=niche_d.get("last_updated_gen", -1),
        )
        # Correlations
        corr_d = d.get("correlations", {})
        intel.correlations = ParamCorrelations(
            correlations=corr_d.get("correlations", {}),
            variances=corr_d.get("variances", {}),
            saturated_params=set(corr_d.get("saturated_params", [])),
            promising_params=set(corr_d.get("promising_params", [])),
            last_updated_gen=corr_d.get("last_updated_gen", -1),
        )
        # Backtest
        bt_d = d.get("backtest", {})
        bt = BacktestPatterns(
            recent_best_fitness=deque(bt_d.get("recent_best_fitness", []), maxlen=20),
            recent_median_fitness=deque(bt_d.get("recent_median_fitness", []), maxlen=20),
            bull_avg_score=bt_d.get("bull_avg_score", 0.0),
            bear_avg_score=bt_d.get("bear_avg_score", 0.0),
            chop_avg_score=bt_d.get("chop_avg_score", 0.0),
            bull_n_months=bt_d.get("bull_n_months", 0),
            bear_n_months=bt_d.get("bear_n_months", 0),
            chop_n_months=bt_d.get("chop_n_months", 0),
            worst_dd_months=bt_d.get("worst_dd_months", []),
            avg_worst_month_dd=bt_d.get("avg_worst_month_dd", 0.0),
            avg_recovery_speed=bt_d.get("avg_recovery_speed", 0.0),
            avg_trades_per_month=bt_d.get("avg_trades_per_month", 0.0),
            last_updated_gen=bt_d.get("last_updated_gen", -1),
        )
        intel.backtest = bt
        return intel


# ============================================================
# Intelligence tracker — updates state after each gen
# ============================================================

class IslandIntelligenceTracker:
    """Builds IslandIntelligence from a gen's EvaluationResults.

    Called by the harness after each gen with:
    - All eval results (filtered to this island)
    - The top-20 elites (for niche fingerprint)
    - The gen_index

    Persists state to disk so the smart mutator can read it next gen.
    """

    HISTORY_WINDOW = 10  # how many recent gens to track for correlations
    TOP_N_FOR_NICHE = 20  # top elites to use for niche fingerprint

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.intel_dir = self.output_dir / "island_intelligence"
        self.intel_dir.mkdir(parents=True, exist_ok=True)

    def load(self, island_id: int) -> IslandIntelligence | None:
        """Load most recent intelligence state for an island."""
        path = self.intel_dir / f"I{island_id:02d}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return IslandIntelligence.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def save(self, intel: IslandIntelligence) -> Path:
        """Persist island intelligence to disk."""
        path = self.intel_dir / f"I{intel.island_id:02d}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(intel.to_dict(), f, indent=2, default=str)
        tmp.replace(path)
        return path

    def update(
        self,
        island_id: int,
        gen_index: int,
        eval_results: list[EvaluationResult],
        elites: list[CandidateGenome],
    ) -> IslandIntelligence:
        """Build / update intelligence state for an island after one gen.

        Args:
            island_id: which island (1..N)
            gen_index: current generation index
            eval_results: ALL eval results for this gen (filter to this island internally)
            elites: top-20 candidates (CandidateGenome objects) for this island

        Returns:
            Updated IslandIntelligence object (also persisted to disk).
        """
        # Load prior state if exists, else start fresh
        intel = self.load(island_id) or IslandIntelligence(
            island_id=island_id,
            bias_name=get_island_spec(island_id).name,
        )

        # Filter eval results to this island
        island_results = [
            r for r in eval_results
            if get_island_id_for_genome_by_candidate_id(r.genome_id, elites) == island_id
        ]

        # 1) Update niche fingerprint from top elites
        intel.niche = self._build_niche(elites, gen_index)
        intel.niche.last_updated_gen = gen_index

        # 2) Update correlations from recent history
        intel.correlations = self._build_correlations(island_id, gen_index)
        intel.correlations.last_updated_gen = gen_index

        # 3) Update backtest patterns
        intel.backtest = self._build_backtest_patterns(
            intel.backtest, island_results, gen_index
        )
        intel.backtest.last_updated_gen = gen_index

        intel.last_updated_gen = gen_index
        self.save(intel)
        return intel

    # ----- Internal helpers -----

    def _build_niche(self, elites: list[CandidateGenome], gen_index: int) -> NicheFingerprint:
        """Compute centroid + std of params across top elites."""
        if not elites:
            return NicheFingerprint()

        # Collect all numeric param values
        all_keys: set[str] = set()
        for e in elites:
            all_keys.update(e.dca_genome.grid_params.keys())

        centroid: dict[str, float] = {}
        std: dict[str, float] = {}
        spread: dict[str, float] = {}

        for key in all_keys:
            values = [
                float(e.dca_genome.grid_params.get(key))
                for e in elites
                if e.dca_genome.grid_params.get(key) is not None
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
            centroid[key] = mean
            std[key] = math.sqrt(var)
            spread[key] = max(values) - min(values)

        # Categorical distributions
        categorical_dist: dict[str, dict[str, float]] = {}
        for cat_key in ["grid_method", "allocation_method"]:
            counts: dict[str, int] = {}
            for e in elites:
                val = getattr(e.dca_genome, cat_key)
                v = val.value if hasattr(val, "value") else str(val)
                counts[v] = counts.get(v, 0) + 1
            total = sum(counts.values())
            if total > 0:
                categorical_dist[cat_key] = {k: v / total for k, v in counts.items()}

        return NicheFingerprint(
            centroid=centroid,
            std=std,
            spread=spread,
            categorical_dist=categorical_dist,
            sample_size=len(elites),
            last_updated_gen=gen_index,
        )

    def _build_correlations(self, island_id: int, gen_index: int) -> ParamCorrelations:
        """Compute param-fitness correlations from last N gens.

        Reads island_intelligence history from generation_history.json
        (the per-island_best_fitness + per-island_best_genome files written
        by the harness each gen).
        """
        correlations: dict[str, float] = {}
        variances: dict[str, float] = {}

        # Read per-island best genomes for last N gens
        param_fitness_pairs: dict[str, list[tuple[float, float]]] = {}
        for gen_offset in range(self.HISTORY_WINDOW):
            target_gen = gen_index - gen_offset
            if target_gen < 0:
                break
            path = (
                self.output_dir
                / "best_genomes"
                / f"per_island_gen_{target_gen:04d}_island_{island_id:02d}.json"
            )
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    bg = json.load(f)
                fitness = bg.get("dca_genome", {}).get("fitness", 0.0)
                # Fitness isn't stored in the genome file; need to pair with gen_history
                # For now, use a placeholder: get fitness from generation_history.json
                gh_path = self.output_dir / "generation_history.json"
                if gh_path.exists():
                    with open(gh_path) as gf:
                        gh = json.load(gf)
                    # Find this gen's best fitness
                    for g in gh.get("generations", []):
                        if g["generation_index"] == target_gen:
                            pibf = g.get("per_island_best_fitness", {})
                            fitness = float(pibf.get(str(island_id), 0.0))
                            break
                params = bg.get("dca_genome", {}).get("grid_params", {})
                for key, val in params.items():
                    if not isinstance(val, (int, float)):
                        continue
                    param_fitness_pairs.setdefault(key, []).append((float(val), fitness))
            except (json.JSONDecodeError, OSError, KeyError):
                continue

        # Compute Pearson r for each param
        for key, pairs in param_fitness_pairs.items():
            if len(pairs) < 3:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
            den_x = sum((x - mean_x) ** 2 for x in xs)
            den_y = sum((y - mean_y) ** 2 for y in ys)
            den = math.sqrt(den_x * den_y) if den_x > 0 and den_y > 0 else 0.0
            if den > 0:
                correlations[key] = num / den
            # Variance of param values
            var = sum((x - mean_x) ** 2 for x in xs) / max(len(xs) - 1, 1)
            variances[key] = var

        pc = ParamCorrelations(
            correlations=correlations,
            variances=variances,
            last_updated_gen=gen_index,
        )
        pc.classify()
        return pc

    def _build_backtest_patterns(
        self, prior: BacktestPatterns, results: list[EvaluationResult], gen_index: int
    ) -> BacktestPatterns:
        """Extract regime performance, DD hotspots, recovery from this gen's results."""
        bt = BacktestPatterns(
            recent_best_fitness=deque(prior.recent_best_fitness, maxlen=20),
            recent_median_fitness=deque(prior.recent_median_fitness, maxlen=20),
        )

        if not results:
            bt.last_updated_gen = gen_index
            return bt

        # Find this gen's best + median fitness for this island
        fitnesses = sorted([r.discovery_fitness for r in results], reverse=True)
        bt.recent_best_fitness.append(fitnesses[0])
        if len(fitnesses) >= 2:
            median_idx = len(fitnesses) // 2
            bt.recent_median_fitness.append(fitnesses[median_idx])

        # Regime performance from monthly scores
        bull_scores: list[float] = []
        bear_scores: list[float] = []
        chop_scores: list[float] = []
        worst_dds: list[dict[str, Any]] = []
        recovery_speeds: list[float] = []
        trade_counts: list[float] = []

        for r in results:
            mf = r.monthly_fitness
            for ms in mf.monthly_scores:
                if ms.rejected:
                    continue
                trade_counts.append(ms.trades_per_month)
                worst_dds.append({
                    "month": ms.month_label,
                    "dd_pct": ms.max_drawdown_pct,
                    "score": ms.monthly_score,
                })
                # Classify regime by monthly return (simple heuristic)
                if ms.net_profit_pct > 5.0:
                    bull_scores.append(ms.monthly_score)
                elif ms.net_profit_pct < -2.0:
                    bear_scores.append(ms.monthly_score)
                else:
                    chop_scores.append(ms.monthly_score)
            # Recovery speed from breakdown
            rb = mf.recovery_breakdown
            if rb:
                recovery_speeds.append(rb.get("drawdown_recovery_speed", 0.0))

        if bull_scores:
            bt.bull_avg_score = sum(bull_scores) / len(bull_scores)
            bt.bull_n_months = len(bull_scores)
        if bear_scores:
            bt.bear_avg_score = sum(bear_scores) / len(bear_scores)
            bt.bear_n_months = len(bear_scores)
        if chop_scores:
            bt.chop_avg_score = sum(chop_scores) / len(chop_scores)
            bt.chop_n_months = len(chop_scores)

        if worst_dds:
            worst_dds.sort(key=lambda x: x["dd_pct"], reverse=True)
            bt.worst_dd_months = worst_dds[:5]
            bt.avg_worst_month_dd = sum(d["dd_pct"] for d in worst_dds) / len(worst_dds)
        if recovery_speeds:
            bt.avg_recovery_speed = sum(recovery_speeds) / len(recovery_speeds)
        if trade_counts:
            bt.avg_trades_per_month = sum(trade_counts) / len(trade_counts)

        bt.last_updated_gen = gen_index
        return bt


# ============================================================
# Helper — find island_id for a candidate by genome_id
# ============================================================

def get_island_id_for_genome_by_candidate_id(
    genome_id: str, elites: list[CandidateGenome]
) -> int:
    """Look up island_id from a CandidateGenome by matching genome_id."""
    for e in elites:
        if e.genome_id == genome_id:
            return get_island_id_for_genome(e)
    return 0  # random bag / unknown
