"""Allocation calculators — pure functions returning layer sizes in USDT.

Each function takes a context (layer index, base stake, current state)
and returns the size (in USDT) for that layer.

Conventions:
- Size is always in stake currency (USDT)
- All methods respect the max_total_size cap (from safety_genome)
- The base_stake is the seed; methods scale from there
"""
from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# Context
# ============================================================

@dataclass
class AllocationContext:
    """State for computing a layer's allocation size."""
    base_stake: float            # base layer size in USDT
    layer_index: int             # 0 = initial, 1 = first DCA, etc.
    layers_filled: int           # how many layers already placed
    n_layers_total: int          # total layers planned
    current_price: float
    avg_entry: float
    current_dd_pct: float = 0.0          # current drawdown from avg_entry
    volatility: float | None = None      # for vol-adjusted
    max_total_size: float | None = None  # cap from safety_genome


# ============================================================
# Dispatcher
# ============================================================

def compute_layer_allocation(
    allocation_method: str,
    allocation_params: dict[str, float],
    ctx: AllocationContext,
) -> float:
    """Top-level dispatcher — returns the layer size in USDT."""
    if allocation_method == "equal":
        return allocation_equal(allocation_params, ctx)
    if allocation_method == "linear_increasing":
        return allocation_linear_increasing(allocation_params, ctx)
    if allocation_method == "controlled_exp":
        return allocation_controlled_exp(allocation_params, ctx)
    if allocation_method == "drawdown_adjusted":
        return allocation_drawdown_adjusted(allocation_params, ctx)
    if allocation_method == "volatility_adjusted":
        return allocation_volatility_adjusted(allocation_params, ctx)
    raise ValueError(f"Unknown allocation_method: {allocation_method!r}")


def compute_total_position_size(
    allocation_method: str,
    allocation_params: dict[str, float],
    base_stake: float,
    n_layers: int,
    current_price: float = 0.0,
    max_total_size: float | None = None,
) -> float:
    """Compute the total position size across all layers.

    Useful for pre-flight check: can we afford n_layers of size base_stake
    with this allocation method? Returns the sum of layer sizes.
    """
    total = 0.0
    for i in range(n_layers):
        ctx = AllocationContext(
            base_stake=base_stake,
            layer_index=i,
            layers_filled=i,
            n_layers_total=n_layers,
            current_price=current_price,
            avg_entry=current_price,
            max_total_size=max_total_size,
        )
        size = compute_layer_allocation(allocation_method, allocation_params, ctx)
        total += size
    return total


# ============================================================
# Implementations
# ============================================================

def allocation_equal(params: dict[str, float], ctx: AllocationContext) -> float:
    """Every layer is the same size = base_stake."""
    return _cap_size(ctx.base_stake, ctx)


def allocation_linear_increasing(params: dict[str, float], ctx: AllocationContext) -> float:
    """Size grows linearly with layer index.

    params: { "increment_pct": 0.20 } — each layer is 20% larger than the previous
    size_i = base_stake * (1 + i * increment_pct)
    """
    increment = float(params.get("increment_pct", 0.20))
    size = ctx.base_stake * (1.0 + ctx.layer_index * increment)
    return _cap_size(size, ctx)


def allocation_controlled_exp(params: dict[str, float], ctx: AllocationContext) -> float:
    """Size grows by controlled exponential (martingale-like, capped).

    params: { "multiplier": 1.5, "max_layer_size_pct": 5.0 }
    size_i = base_stake * multiplier^i
    Capped at base_stake * max_layer_size_pct.

    This is the classic martingale: layer 0 = 100, layer 1 = 150, layer 2 = 225...
    The cap prevents runaway sizing.
    """
    multiplier = float(params.get("multiplier", 1.5))
    max_size_pct = float(params.get("max_layer_size_pct", 5.0))
    if multiplier <= 1.0:
        raise ValueError(f"multiplier must be > 1.0, got {multiplier}")
    raw_size = ctx.base_stake * (multiplier ** ctx.layer_index)
    max_size = ctx.base_stake * max_size_pct
    size = min(raw_size, max_size)
    return _cap_size(size, ctx)


def allocation_drawdown_adjusted(params: dict[str, float], ctx: AllocationContext) -> float:
    """Size scales with current drawdown (deeper DD = bigger layer).

    params: { "sensitivity": 2.0, "min_size_pct": 0.5, "max_size_pct": 5.0 }
    size = base_stake * (1 + sensitivity * dd)
    Clamped to [min_size_pct, max_size_pct] * base_stake.
    """
    sensitivity = float(params.get("sensitivity", 2.0))
    min_size_pct = float(params.get("min_size_pct", 0.5))
    max_size_pct = float(params.get("max_size_pct", 5.0))
    # ctx.current_dd_pct is positive (0.10 = 10% below avg)
    dd = max(0.0, ctx.current_dd_pct)
    scale = 1.0 + sensitivity * dd
    scale = max(min_size_pct, min(scale, max_size_pct))
    size = ctx.base_stake * scale
    return _cap_size(size, ctx)


def allocation_volatility_adjusted(params: dict[str, float], ctx: AllocationContext) -> float:
    """Size inversely with volatility (calm markets = bigger positions).

    params: { "reference_vol": 0.02, "min_size_pct": 0.5, "max_size_pct": 3.0 }
    High vol → smaller layer. Low vol → bigger layer.
    """
    if ctx.volatility is None or ctx.volatility <= 0:
        return _cap_size(ctx.base_stake, ctx)
    ref_vol = float(params.get("reference_vol", 0.02))
    min_pct = float(params.get("min_size_pct", 0.5))
    max_pct = float(params.get("max_size_pct", 3.0))
    # If vol = ref_vol → scale = 1.0
    # If vol = 2 * ref_vol → scale = 0.5
    # If vol = 0.5 * ref_vol → scale = 2.0
    scale = ref_vol / ctx.volatility
    scale = max(min_pct, min(scale, max_pct))
    return _cap_size(ctx.base_stake * scale, ctx)


# ============================================================
# Helpers
# ============================================================

def _cap_size(size: float, ctx: AllocationContext) -> float:
    """Apply max_total_size cap if set."""
    if ctx.max_total_size is not None and ctx.max_total_size > 0:
        return min(size, ctx.max_total_size)
    return size
