"""Order manager — decides when to fire DCA layers / close on TP.

Stage 3 placeholder logic:
- If no open position: emit "open" decision when trigger price reached (initial entry)
- If open position: emit "add_layer" when price drops by `grid_pct` from average entry, up to `max_layers`
- Emit "close" when current price >= average_entry × (1 + tp_pct)
- Confirmations: checked when indicators + confirmation_indicators are provided

Stage 8 wiring: confirmation_indicators + indicators_df enable gating.
Stage 11 will swap close logic for real tp_genome.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from dca_engine.indicators import IndicatorSnapshot


class OrderAction(StrEnum):
    NONE = "none"
    OPEN_CYCLE = "open_cycle"
    ADD_LAYER = "add_layer"
    CLOSE_CYCLE = "close_cycle"


@dataclass
class OrderDecision:
    action: OrderAction
    cycle_id: str | None = None
    price: float = 0.0
    qty: float = 0.0
    reason: str = ""  # human-readable explanation


class OrderManager:
    """Order manager with optional confirmation gating, pluggable grid methods, and per-layer zones.

    Args:
        grid_pct: spacing between DCA layers as % of average_entry (e.g. 0.015 = 1.5%).
                  Used as fallback for fixed_pct and as default pct when grid_params omits it.
        tp_pct: take-profit as % of average_entry (e.g. 0.02 = 2%)
        max_layers: max number of DCA layers per cycle
        symbol: trading pair (default "BTC/USDT")
        min_trade_qty: minimum qty for a layer (default 0.001 BTC)
        confirmation_indicators: list of indicator names to gate on (empty = no gating)
        indicator_params: dict of {indicator_name: {param: value}} for threshold overrides
        cooldown_candles: candles to wait after cycle close before opening new cycle
        grid_method: grid spacing method (default "fixed_pct") — used when zones is None
        grid_params: extra params for the grid method (e.g. {"pct": 0.015, "atr_multiplier": 2.0})
                     — used when zones is None
        zones: optional list of GridZoneSpec for per-layer method switching (Stage 2 combos).
               When set, the OrderManager picks (grid_method, grid_params) from the zone whose
               layer_start..layer_start+layer_count-1 range contains the next layer index
               (position_layers + 1, 1-indexed). The flat grid_method/grid_params kwargs are
               ignored when zones is set.
    """

    def __init__(
        self,
        grid_pct: float = 0.015,
        tp_pct: float = 0.02,
        max_layers: int = 3,
        symbol: str = "BTC/USDT",
        min_trade_qty: float = 0.001,
        confirmation_indicators: list[str] | None = None,
        indicator_params: dict[str, dict[str, float]] | None = None,
        cooldown_candles: int = 0,
        grid_method: str = "fixed_pct",
        grid_params: dict[str, float] | None = None,
        zones: list | None = None,
    ) -> None:
        if grid_pct <= 0:
            raise ValueError(f"grid_pct must be > 0, got {grid_pct}")
        if tp_pct <= 0:
            raise ValueError(f"tp_pct must be > 0, got {tp_pct}")
        if max_layers < 1:
            raise ValueError(f"max_layers must be >= 1, got {max_layers}")
        if cooldown_candles < 0:
            raise ValueError(f"cooldown_candles must be >= 0, got {cooldown_candles}")
        self.grid_pct = grid_pct
        self.tp_pct = tp_pct
        self.max_layers = max_layers
        self.symbol = symbol
        self.min_trade_qty = min_trade_qty
        self.confirmation_indicators = confirmation_indicators or []
        self.indicator_params = indicator_params or {}
        self.cooldown_candles = cooldown_candles
        self.grid_method = grid_method
        self.grid_params = grid_params or {}
        # For grid methods that need the pct fallback (fixed_pct uses grid_pct)
        if "pct" not in self.grid_params:
            self.grid_params["pct"] = grid_pct
        # Optional per-layer zones (Stage 2 combos). Default None = single-zone legacy behaviour.
        self.zones = zones
        if self.zones is not None:
            # Sort by layer_start for fast lookup; validator guarantees non-overlap & contiguity
            self.zones = sorted(self.zones, key=lambda z: z.layer_start)

    def _active_zone(self, next_layer_index: int):
        """Return the zone that owns this layer (1-indexed).

        next_layer_index is the index of the NEXT layer to be filled (1-indexed).
        Falls back to the last zone if next_layer_index exceeds all zone bounds
        (shouldn't happen when max_layers matches zones coverage, but defends
        against off-by-one in callers).
        Returns None only when self.zones is empty.
        """
        if not self.zones:
            return None
        for zone in self.zones:
            if zone.layer_start <= next_layer_index <= zone.layer_start + zone.layer_count - 1:
                return zone
        return self.zones[-1]

    def _check_confirmations(
        self,
        indicators: IndicatorSnapshot,
        current_price: float,
    ) -> tuple[bool, list[str]]:
        """Check all confirmation indicators. Returns (all_passed, failed_list)."""
        from dca_calc.confirmation import ConfirmationContext, check_all_confirmations

        if not self.confirmation_indicators:
            return True, []

        ctx = ConfirmationContext(
            rsi_value=indicators.rsi,
            ma_value=indicators.ma,
            current_price=current_price,
            volatility=indicators.volatility,
            reference_vol=indicators.reference_vol,
        )
        return check_all_confirmations(
            indicators=self.confirmation_indicators,
            params_map=self.indicator_params,
            ctx=ctx,
        )

    def decide(
        self,
        cycle_id: str,
        current_price: float,
        current_time: str,
        position_layers: int,
        average_entry: float,
        has_open_position: bool,
        stake_amount: float,
        indicators: Optional[IndicatorSnapshot] = None,
    ) -> OrderDecision:
        """Decide what to do this candle.

        When confirmation_indicators are configured, they act as an AND gate:
        all must pass before a new cycle can open or a layer can be added.
        TP close is NEVER gated by confirmations (always closes when TP hit).

        Args:
            indicators: pre-computed indicator snapshot for this candle.
                        If None and confirmations are configured, the check
                        is skipped (fail-open for safety during transition).

        Returns an OrderDecision; caller (state_machine) executes it.
        """
        # 1) No open position → check if we should OPEN
        if not has_open_position:
            # Check confirmations before opening
            if self.confirmation_indicators and indicators is not None:
                passed, failed = self._check_confirmations(indicators, current_price)
                if not passed:
                    return OrderDecision(
                        action=OrderAction.NONE,
                        reason=f"confirmation_failed:{','.join(failed)}",
                    )
            qty = stake_amount / current_price
            if qty < self.min_trade_qty:
                return OrderDecision(action=OrderAction.NONE, reason="qty_below_min")
            return OrderDecision(
                action=OrderAction.OPEN_CYCLE,
                cycle_id=cycle_id,
                price=current_price,
                qty=qty,
                reason="no_open_position_open_new",
            )

        # 2) Open position → check if we should CLOSE (TP hit)
        # TP close is NEVER gated by confirmations
        tp_target = average_entry * (1.0 + self.tp_pct)
        if current_price >= tp_target:
            return OrderDecision(
                action=OrderAction.CLOSE_CYCLE,
                cycle_id=cycle_id,
                price=current_price,
                qty=0.0,  # caller resolves from position
                reason="tp_hit",
            )

        # 3) Open position → check if we should ADD LAYER
        if position_layers >= self.max_layers:
            return OrderDecision(action=OrderAction.NONE, reason="max_layers_reached")

        # Check confirmations before adding layer
        if self.confirmation_indicators and indicators is not None:
            passed, failed = self._check_confirmations(indicators, current_price)
            if not passed:
                return OrderDecision(
                    action=OrderAction.NONE,
                    reason=f"confirmation_failed:{','.join(failed)}",
                )

        # Trigger if price dropped grid_pct × layers_filled from average_entry
        # For pluggable grid methods, compute the trigger price via dca_calc.
        from dca_calc.grid_spacing import GridContext, compute_next_layer_price

        # Zone-aware dispatch: pick (grid_method, grid_params) from the zone that owns
        # the next layer index. next_layer_index is 1-indexed (1 = first DCA fill).
        next_layer_index = position_layers + 1
        zone = self._active_zone(next_layer_index) if self.zones else None
        if zone is not None:
            active_grid_method = zone.grid_method.value
            active_grid_params = dict(zone.grid_params)
            # Fallback: ensure pct is set for fixed_pct-style methods
            if "pct" not in active_grid_params:
                active_grid_params["pct"] = self.grid_pct
        else:
            active_grid_method = self.grid_method
            active_grid_params = self.grid_params

        ctx = GridContext(
            current_price=current_price,
            avg_entry=average_entry,
            cycle_high=current_price,  # best-effort; state_machine tracks actual high
            layers_filled=position_layers,
            n_layers_total=self.max_layers,
            atr=indicators.atr if indicators else None,
            volatility=indicators.volatility if indicators else None,
            ma_value=indicators.ma if indicators else None,
            rsi_value=indicators.rsi if indicators else None,
            z_score=indicators.z_score if indicators else None,
            trend_strength=indicators.trend_strength if indicators else None,
            reference_high=average_entry,
        )
        next_layer_target = compute_next_layer_price(
            grid_method=active_grid_method,
            grid_params=active_grid_params,
            ctx=ctx,
        )
        if next_layer_target is None:
            # Grid method couldn't compute (missing indicators) — skip
            return OrderDecision(action=OrderAction.NONE, reason="grid_method_no_price")
        if current_price <= next_layer_target:
            qty = stake_amount / current_price
            if qty < self.min_trade_qty:
                return OrderDecision(action=OrderAction.NONE, reason="qty_below_min")
            return OrderDecision(
                action=OrderAction.ADD_LAYER,
                cycle_id=cycle_id,
                price=current_price,
                qty=qty,
                reason=f"layer_{position_layers + 1}_triggered",
            )

        return OrderDecision(action=OrderAction.NONE, reason="no_trigger")
