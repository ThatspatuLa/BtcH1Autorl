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
    """Order manager with optional confirmation gating.

    Args:
        grid_pct: spacing between DCA layers as % of average_entry (e.g. 0.015 = 1.5%)
        tp_pct: take-profit as % of average_entry (e.g. 0.02 = 2%)
        max_layers: max number of DCA layers per cycle
        symbol: trading pair (default "BTC/USDT")
        min_trade_qty: minimum qty for a layer (default 0.001 BTC)
        confirmation_indicators: list of indicator names to gate on (empty = no gating)
        indicator_params: dict of {indicator_name: {param: value}} for threshold overrides
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
