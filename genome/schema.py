"""Genome schema — CandidateGenome dataclass + serialisation + validation + Freqtrade export.

Schema design (per Kanban Stage 4):
- dca_genome: grid_method, allocation_method, combo_method, trigger_mode, params
- tp_genome: exit_method + params
- safety_genome: max_dca_layers, overlap_allowed, buffer_pct
- settings_overrides: optional per-candidate overrides
- lineage: parent_a_id, parent_b_id, mutation_ops, generation_index, mutation_seed

The schema is framework-aware (NOT Freqtrade-hardcoded): all internal types use enums and dataclasses,
but to_freqtrade_strategy_params() converts to an IStrategy-compatible dict for Stage 17 deployment.

Serialisation: JSON + msgpack both supported, deterministic.
Hashing: sha256 of canonical JSON (sorted keys, no whitespace) — stable genome identity.
Validation: strict checks at parse time + explicit validate_genome() helper.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# ============================================================
# Enums — locked for v1
# ============================================================

class GridMethod(StrEnum):
    """Grid spacing methods for DCA layers (Stage 8 blocks)."""
    FIXED_PCT = "fixed_pct"
    ATR = "atr"
    VOLATILITY = "volatility"
    DRAWDOWN_FROM_HIGH = "drawdown_from_high"
    MA_DISTANCE = "ma_distance"
    RSI_OVERSOLD = "rsi_oversold"
    Z_SCORE = "z_score"
    TREND_ADJUSTED = "trend_adjusted"


class AllocationMethod(StrEnum):
    """Position-sizing methods for each DCA layer."""
    EQUAL = "equal"
    LINEAR_INCREASING = "linear_increasing"
    CONTROLLED_EXP = "controlled_exp"
    DRAWDOWN_ADJUSTED = "drawdown_adjusted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


class ComboMethod(StrEnum):
    """How to combine multiple grid spacing / allocation blocks."""
    WEIGHTED_AVERAGE = "weighted_average"
    CONDITIONAL_SWITCH = "conditional_switch"


class TriggerMode(StrEnum):
    """When to fire the next DCA layer."""
    PRICE_ONLY = "price_only"
    PRICE_WITH_CONFIRMATION = "price_with_confirmation"


class ConfirmationIndicator(StrEnum):
    """Indicators that can confirm a price trigger."""
    RSI_BELOW = "rsi_below"
    RSI_ABOVE = "rsi_above"
    MA_ABOVE = "ma_above"
    MA_BELOW = "ma_below"
    VOLATILITY_HIGH = "volatility_high"
    VOLATILITY_LOW = "volatility_low"


class TpExitMethod(StrEnum):
    """TP / exit logic (Stage 11 blocks)."""
    FIXED = "fixed"
    ATR = "atr"
    VOL_ADJUSTED = "vol_adjusted"
    DCA_DEPTH_ADJUSTED = "dca_depth_adjusted"
    PARTIAL_LADDER = "partial_ladder"
    TRAILING = "trailing"
    BREAK_EVEN = "break_even"
    MOMENTUM_DECAY = "momentum_decay"
    EXHAUSTION = "exhaustion"
    TREND_HOLD = "trend_hold"
    FAILED_CONTINUATION = "failed_continuation"
    TIME_IN_POSITION = "time_in_position"
    HYBRID = "hybrid"


class MarginMode(StrEnum):
    ISOLATED = "isolated"
    CROSS = "cross"


# ============================================================
# Sections
# ============================================================

@dataclass
class GridZoneSpec:
    """One zone in a multi-zone (combo) DCA grid.

    A zone defines which grid method controls a contiguous range of layers.
    For example, in a 3-split combo over 10 layers:
      - zone1: layer 1-3 (shallow dip)
      - zone2: layer 4-6 (medium dip)
      - zone3: layer 7-10 (deep dip)
    Each zone carries its own (grid_method, grid_params) pair so the engine
    switches spacing behaviour when crossing a zone boundary.

    `layer_start` is 1-indexed (first DCA fill). The zone ends at
    `layer_start + layer_count - 1`. Zones must not overlap and together
    must cover layers 1..max_dca_layers (validated in validate_genome).
    """

    layer_start: int
    layer_count: int
    grid_method: GridMethod
    grid_params: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_start": self.layer_start,
            "layer_count": self.layer_count,
            "grid_method": self.grid_method.value
            if hasattr(self.grid_method, "value")
            else str(self.grid_method),
            "grid_params": dict(self.grid_params),
        }


@dataclass
class DcaGenome:
    grid_method: GridMethod
    grid_params: dict[str, float]
    allocation_method: AllocationMethod
    allocation_params: dict[str, float]
    combo_method: ComboMethod = ComboMethod.WEIGHTED_AVERAGE
    combo_params: dict[str, float] = field(default_factory=dict)
    trigger_mode: TriggerMode = TriggerMode.PRICE_ONLY
    confirmation_indicators: list[ConfirmationIndicator] = field(default_factory=list)
    indicator_params: dict[str, dict[str, float]] = field(default_factory=dict)
    max_dca_layers: int = 5
    # Optional per-layer zones (Stage 2 combos). None = single-zone (legacy behaviour).
    # When set, the OrderManager switches grid_method+grid_params based on
    # (position_layers + 1) — the index of the NEXT layer to be filled.
    zones: list[GridZoneSpec] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert enums to their string values for JSON safety
        # Handle both enum and string (when reconstructed from JSON)
        def _enum_str(val):
            return val.value if hasattr(val, 'value') else str(val)
        d["grid_method"] = _enum_str(self.grid_method)
        d["allocation_method"] = _enum_str(self.allocation_method)
        d["combo_method"] = _enum_str(self.combo_method)
        d["trigger_mode"] = _enum_str(self.trigger_mode)
        d["confirmation_indicators"] = [
            c.value if hasattr(c, 'value') else str(c)
            for c in self.confirmation_indicators
        ]
        if self.zones is not None:
            d["zones"] = [z.to_dict() for z in self.zones]
        return d


@dataclass
class TpGenome:
    exit_method: TpExitMethod
    exit_params: dict[str, float]
    # For partial_ladder / hybrid: list of sub-exit rules evaluated together
    sub_exits: list[TpGenome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["exit_method"] = self.exit_method.value if hasattr(self.exit_method, 'value') else str(self.exit_method)
        d["sub_exits"] = [s.to_dict() for s in self.sub_exits]
        return d


@dataclass
class SafetyGenome:
    max_dca_layers: int = 5
    overlap_allowed: bool = False
    min_break_even_for_overlap_pct: float = 0.0
    require_buffer_pct: float = 0.20  # AMENDMENT 2: configurable, default 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SettingsOverrides:
    """Optional per-candidate broker/cost overrides. Validated against Stage 1 schema at use-time."""
    starting_balance: float | None = None
    leverage: float | None = None
    margin_mode: MarginMode | None = None
    fee_pct: float | None = None
    spread_pips: float | None = None
    slippage_pct: float | None = None
    funding_rate_per_8h: float | None = None
    buffer_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.margin_mode is not None:
            d["margin_mode"] = self.margin_mode.value
        # Strip Nones for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class LineageMetadata:
    parent_a_id: str | None = None
    parent_b_id: str | None = None
    generation_index: int = 0
    mutation_seed: int | None = None
    mutation_ops: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Top-level
# ============================================================

@dataclass
class CandidateGenome:
    """Top-level genome — represents one candidate in evolution."""
    genome_id: str
    dca_genome: DcaGenome
    tp_genome: TpGenome
    safety_genome: SafetyGenome = field(default_factory=SafetyGenome)
    settings_overrides: SettingsOverrides = field(default_factory=SettingsOverrides)
    lineage: LineageMetadata = field(default_factory=LineageMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "dca_genome": self.dca_genome.to_dict(),
            "tp_genome": self.tp_genome.to_dict(),
            "safety_genome": self.safety_genome.to_dict(),
            "settings_overrides": self.settings_overrides.to_dict(),
            "lineage": self.lineage.to_dict(),
        }


# ============================================================
# Defaults
# ============================================================

DEFAULT_DCA_GENOME = DcaGenome(
    grid_method=GridMethod.FIXED_PCT,
    grid_params={"grid_pct": 1.5, "max_layers": 5},
    allocation_method=AllocationMethod.EQUAL,
    allocation_params={"base_notional": 100.0, "allocation_cap_pct": 0.10},
    combo_method=ComboMethod.WEIGHTED_AVERAGE,
    combo_params={},
    trigger_mode=TriggerMode.PRICE_ONLY,
    confirmation_indicators=[],
    indicator_params={},
    max_dca_layers=5,
)

DEFAULT_TP_GENOME = TpGenome(
    exit_method=TpExitMethod.FIXED,
    exit_params={"tp_pct": 2.0},
    sub_exits=[],
)

DEFAULT_SAFETY_GENOME = SafetyGenome(
    max_dca_layers=5,
    overlap_allowed=False,
    min_break_even_for_overlap_pct=0.0,
    require_buffer_pct=0.20,
)


# ============================================================
# Serialisation
# ============================================================

def _canonical_json(obj: Any) -> str:
    """Sort keys, no whitespace, deterministic."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def genome_to_json(genome: CandidateGenome, indent: int | None = 2) -> str:
    return json.dumps(genome.to_dict(), indent=indent, sort_keys=True)


def genome_from_json(s: str) -> CandidateGenome:
    d = json.loads(s)
    return _dict_to_genome(d)


def genome_to_msgpack(genome: CandidateGenome) -> bytes:
    import msgpack
    return msgpack.packb(genome.to_dict(), use_bin_type=True)


def genome_from_msgpack(blob: bytes) -> CandidateGenome:
    import msgpack
    d = msgpack.unpackb(blob, raw=False)
    return _dict_to_genome(d)


def _dict_to_zone(d: Mapping[str, Any]) -> GridZoneSpec:
    if not isinstance(d, dict):
        raise GenomeValidationError(f"Zone must be dict, got {type(d).__name__}")
    for k in ("layer_start", "layer_count", "grid_method"):
        if k not in d:
            raise GenomeValidationError(f"Zone missing required field: {k}")
    return GridZoneSpec(
        layer_start=int(d["layer_start"]),
        layer_count=int(d["layer_count"]),
        grid_method=GridMethod(d["grid_method"]),
        grid_params={k: float(v) for k, v in d.get("grid_params", {}).items()},
    )


def _dict_to_genome(d: Mapping[str, Any]) -> CandidateGenome:
    if not isinstance(d, dict):
        raise GenomeValidationError(f"Expected dict, got {type(d).__name__}")
    if "genome_id" not in d:
        raise GenomeValidationError("Missing required field: genome_id")
    dca = d.get("dca_genome", {})
    tp = d.get("tp_genome", {})
    safety = d.get("safety_genome", {})
    overrides = d.get("settings_overrides", {})
    lineage = d.get("lineage", {})
    raw_zones = dca.get("zones")
    zones = [_dict_to_zone(z) for z in raw_zones] if raw_zones is not None else None
    return CandidateGenome(
        genome_id=d["genome_id"],
        dca_genome=DcaGenome(
            grid_method=GridMethod(dca["grid_method"]),
            grid_params=dict(dca.get("grid_params", {})),
            allocation_method=AllocationMethod(dca["allocation_method"]),
            allocation_params=dict(dca.get("allocation_params", {})),
            combo_method=ComboMethod(dca.get("combo_method", ComboMethod.WEIGHTED_AVERAGE.value)),
            combo_params=dict(dca.get("combo_params", {})),
            trigger_mode=TriggerMode(dca.get("trigger_mode", TriggerMode.PRICE_ONLY.value)),
            confirmation_indicators=[ConfirmationIndicator(c) for c in dca.get("confirmation_indicators", [])],
            indicator_params=dict(dca.get("indicator_params", {})),
            max_dca_layers=int(dca.get("max_dca_layers", 5)),
            zones=zones,
        ),
        tp_genome=TpGenome(
            exit_method=TpExitMethod(tp["exit_method"]),
            exit_params=dict(tp.get("exit_params", {})),
            sub_exits=[_dict_to_tp(s) for s in tp.get("sub_exits", [])],
        ),
        safety_genome=SafetyGenome(
            max_dca_layers=int(safety.get("max_dca_layers", 5)),
            overlap_allowed=bool(safety.get("overlap_allowed", False)),
            min_break_even_for_overlap_pct=float(safety.get("min_break_even_for_overlap_pct", 0.0)),
            require_buffer_pct=float(safety.get("require_buffer_pct", 0.20)),
        ),
        settings_overrides=SettingsOverrides(
            starting_balance=_maybe_float(overrides.get("starting_balance")),
            leverage=_maybe_float(overrides.get("leverage")),
            margin_mode=MarginMode(overrides["margin_mode"]) if overrides.get("margin_mode") else None,
            fee_pct=_maybe_float(overrides.get("fee_pct")),
            spread_pips=_maybe_float(overrides.get("spread_pips")),
            slippage_pct=_maybe_float(overrides.get("slippage_pct")),
            funding_rate_per_8h=_maybe_float(overrides.get("funding_rate_per_8h")),
            buffer_pct=_maybe_float(overrides.get("buffer_pct")),
        ),
        lineage=LineageMetadata(
            parent_a_id=lineage.get("parent_a_id"),
            parent_b_id=lineage.get("parent_b_id"),
            generation_index=int(lineage.get("generation_index", 0)),
            mutation_seed=lineage.get("mutation_seed"),
            mutation_ops=list(lineage.get("mutation_ops", [])),
            created_at=lineage.get("created_at"),
        ),
    )


def _dict_to_tp(d: Mapping[str, Any]) -> TpGenome:
    return TpGenome(
        exit_method=TpExitMethod(d["exit_method"]),
        exit_params=dict(d.get("exit_params", {})),
        sub_exits=[_dict_to_tp(s) for s in d.get("sub_exits", [])],
    )


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise GenomeValidationError(f"Expected number, got {type(v).__name__}: {v!r}")


# ============================================================
# Hashing + Freqtrade export
# ============================================================

def genome_hash(genome: CandidateGenome) -> str:
    """Stable sha256 of canonical JSON. Used as genome_id when not user-supplied.

    Hash is over ALL fields except genome_id (so two genomes with same params hash identically,
    distinguishing via lineage only).
    """
    d = genome.to_dict()
    d.pop("genome_id", None)
    blob = _canonical_json(d).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def to_freqtrade_strategy_params(genome: CandidateGenome) -> dict[str, Any]:
    """Convert genome to IStrategy-compatible dict for Stage 17 deployment.

    NOT a real IStrategy yet — just the parameter block that would feed into a generated strategy file.
    """
    return {
        "strategy_name": f"BTCWeightedDCA_{genome.genome_id[:8]}",
        "dca": {
            "grid_method": genome.dca_genome.grid_method.value,
            "grid_params": genome.dca_genome.grid_params,
            "allocation_method": genome.dca_genome.allocation_method.value,
            "allocation_params": genome.dca_genome.allocation_params,
            "combo_method": genome.dca_genome.combo_method.value,
            "combo_params": genome.dca_genome.combo_params,
            "trigger_mode": genome.dca_genome.trigger_mode.value,
            "confirmation_indicators": [c.value for c in genome.dca_genome.confirmation_indicators],
            "indicator_params": genome.dca_genome.indicator_params,
            "max_dca_layers": genome.dca_genome.max_dca_layers,
            "zones": [z.to_dict() for z in genome.dca_genome.zones]
            if genome.dca_genome.zones is not None
            else None,
        },
        "tp": {
            "exit_method": genome.tp_genome.exit_method.value,
            "exit_params": genome.tp_genome.exit_params,
            "sub_exits": [
                {"exit_method": s.exit_method.value, "exit_params": s.exit_params}
                for s in genome.tp_genome.sub_exits
            ],
        },
        "safety": genome.safety_genome.to_dict(),
        "settings_overrides": genome.settings_overrides.to_dict(),
    }


# ============================================================
# Validation
# ============================================================

class GenomeValidationError(ValueError):
    """Raised when genome validation fails."""


def validate_genome(genome: CandidateGenome) -> None:
    """Strict validation. Raises GenomeValidationError on any problem."""
    if not genome.genome_id or not isinstance(genome.genome_id, str):
        raise GenomeValidationError("genome_id must be non-empty string")

    # DCA checks
    if genome.dca_genome.max_dca_layers < 1:
        raise GenomeValidationError(f"max_dca_layers must be >= 1, got {genome.dca_genome.max_dca_layers}")
    if genome.dca_genome.max_dca_layers > 50:
        raise GenomeValidationError(f"max_dca_layers must be <= 50 (sanity cap), got {genome.dca_genome.max_dca_layers}")

    # Confirmation indicators required if trigger mode is PRICE_WITH_CONFIRMATION
    if (
        genome.dca_genome.trigger_mode == TriggerMode.PRICE_WITH_CONFIRMATION
        and not genome.dca_genome.confirmation_indicators
    ):
        raise GenomeValidationError(
            "trigger_mode=price_with_confirmation requires at least one confirmation_indicator"
        )

    # Grid params sanity — allow primitives only (no nested lists/dicts in DCA params)
    for k, v in genome.dca_genome.grid_params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or math.isinf(v):
            raise GenomeValidationError(f"grid_params[{k!r}] must be finite number, got {v!r}")

    # Allocation params sanity — allow primitives only
    for k, v in genome.dca_genome.allocation_params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or math.isinf(v):
            raise GenomeValidationError(f"allocation_params[{k!r}] must be finite number, got {v!r}")

    # TP params — allow primitives AND lists (for partial_ladder levels like [[0.5, 0.3], [1.0, 0.3]])
    for k, v in genome.tp_genome.exit_params.items():
        if isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, list):
                    for j, sub in enumerate(item):
                        if not isinstance(sub, (int, float)) or isinstance(sub, bool) or math.isnan(sub) or math.isinf(sub):
                            raise GenomeValidationError(
                                f"exit_params[{k!r}][{i}][{j}] must be finite number, got {sub!r}"
                            )
                elif not isinstance(item, (int, float)) or isinstance(item, bool) or math.isnan(item) or math.isinf(item):
                    raise GenomeValidationError(f"exit_params[{k!r}][{i}] must be finite number, got {item!r}")
        elif not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or math.isinf(v):
            raise GenomeValidationError(f"exit_params[{k!r}] must be finite number, got {v!r}")

    # Safety buffer_pct must be in [0, 1) (AMENDMENT 2: configurable)
    if not (0.0 <= genome.safety_genome.require_buffer_pct < 1.0):
        raise GenomeValidationError(
            f"safety_genome.require_buffer_pct must be in [0, 1), got {genome.safety_genome.require_buffer_pct}"
        )

    # Settings overrides — if any, must be finite
    for k, v in genome.settings_overrides.to_dict().items():
        if k == "margin_mode":
            continue
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise GenomeValidationError(f"settings_overrides[{k!r}] must be finite number, got {v!r}")

    # Zones — when present, must be a contiguous, non-overlapping cover of layers 1..max_dca_layers
    if genome.dca_genome.zones is not None:
        zones = genome.dca_genome.zones
        if not zones:
            raise GenomeValidationError("zones list is empty (use None for single-zone)")
        # Sort by layer_start for validation
        sorted_zones = sorted(zones, key=lambda z: z.layer_start)
        cursor = 1
        for zone in sorted_zones:
            if zone.layer_start < 1:
                raise GenomeValidationError(
                    f"zone.layer_start must be >= 1, got {zone.layer_start}"
                )
            if zone.layer_count < 1:
                raise GenomeValidationError(
                    f"zone.layer_count must be >= 1, got {zone.layer_count}"
                )
            if zone.layer_start != cursor:
                raise GenomeValidationError(
                    f"zones must be contiguous starting at 1 — gap at layer {cursor}, "
                    f"next zone starts at {zone.layer_start}"
                )
            cursor += zone.layer_count
            # Validate zone grid_params are finite numbers
            for k, v in zone.grid_params.items():
                if (
                    not isinstance(v, (int, float))
                    or isinstance(v, bool)
                    or math.isnan(v)
                    or math.isinf(v)
                ):
                    raise GenomeValidationError(
                        f"zone.grid_params[{k!r}] must be finite number, got {v!r}"
                    )
        if cursor - 1 != genome.dca_genome.max_dca_layers:
            raise GenomeValidationError(
                f"zones cover layers 1..{cursor - 1} but max_dca_layers="
                f"{genome.dca_genome.max_dca_layers} (must match)"
            )
