"""Order manager — decides when to fire DCA layers / close on TP.

Stage 3 placeholder logic:
- If no open position: emit "open" decision when trigger price reached (initial entry)
- If open position: emit "add_layer" when price drops by `grid_pct` from average entry, up to `max_layers`
- Emit "close" when current price >= average_entry × (1 + tp_pct)
- Confirmations: Stage 3 ignores confirmation indicators (price-only trigger)

Stage 8 will swap this for real grid_method / allocation_method / combo_method logic.
Stage 11 will swap close logic for real tp_genome.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
    """Stage 3 placeholder — real logic comes in Stages 8 + 11.

    Args:
        grid_pct: spacing between DCA layers as % of average_entry (e.g. 0.015 = 1.5%)
        tp_pct: take-profit as % of average_entry (e.g. 0.02 = 2%)
        max_layers: max number of DCA layers per cycle (Stage 3 hardcoded to 3)
        symbol: trading pair (default "BTC/USDT")
        min_trade_qty: minimum qty for a layer (Stage 3 = 0.001 BTC)
    """

    def __init__(
        self,
        grid_pct: float = 0.015,
        tp_pct: float = 0.02,
        max_layers: int = 3,
        symbol: str = "BTC/USDT",
        min_trade_qty: float = 0.001,
    ) -> None:
        if grid_pct <= 0:
            raise ValueError(f"grid_pct must be > 0, got {grid_pct}")
        if tp_pct <= 0:
            raise ValueError(f"tp_pct must be > 0, got {tp_pct}")
        if max_layers < 1:
            raise ValueError(f"max_layers must be >= 1, got {max_layers}")
        self.grid_pct = grid_pct
        self.tp_pct = tp_pct
        self.max_layers = max_layers
        self.symbol = symbol
        self.min_trade_qty = min_trade_qty

    def decide(
        self,
        cycle_id: str,
        current_price: float,
        current_time: str,
        position_layers: int,
        average_entry: float,
        has_open_position: bool,
        stake_amount: float,
    ) -> OrderDecision:
        """Decide what to do this candle.

        Returns an OrderDecision; caller (state_machine) executes it.
        """
        # 1) No open position → check if we should OPEN
        if not has_open_position:
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
        tp_target = average_entry * (1.0 + self.tp_pct)
        if current_price >= tp_target:
            # Close entire position
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

        # Trigger if price dropped grid_pct × layers_filled from average_entry
        # Layer 1 (initial) is filled. For layer 2, trigger = -grid_pct from layer 1 price.
        # Stage 3 uses a simpler model: next layer trigger = avg_entry × (1 - grid_pct × position_layers)
        next_layer_target = average_entry * (1.0 - self.grid_pct * position_layers)
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
