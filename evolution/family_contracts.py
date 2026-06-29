"""Mutation contracts for Stage 1 spacing-family hyperopt runs.

These contracts keep a family focused on its spacing identity while allowing
allocation, depth, and fixed TP to continue mutating inside that identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genome.schema import ConfirmationIndicator, GridMethod


@dataclass(frozen=True)
class FamilyMutationContract:
    """Rules that a Stage 1 family must obey during generation and mutation."""

    name: str
    forced_grid_methods: tuple[GridMethod, ...]
    forced_confirmations: tuple[ConfirmationIndicator, ...] | None = ()
    allow_grid_method_switch: bool = False
    allow_allocation_mutation: bool = True
    allow_depth_mutation: bool = True
    allow_fixed_tp_mutation: bool = True
    max_dca_layers_cap: int | None = None
    stage: int = 1
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "forced_grid_methods": [method.value for method in self.forced_grid_methods],
            "forced_confirmations": (
                [indicator.value for indicator in self.forced_confirmations]
                if self.forced_confirmations is not None
                else None
            ),
            "allow_grid_method_switch": self.allow_grid_method_switch,
            "allow_allocation_mutation": self.allow_allocation_mutation,
            "allow_depth_mutation": self.allow_depth_mutation,
            "allow_fixed_tp_mutation": self.allow_fixed_tp_mutation,
            "max_dca_layers_cap": self.max_dca_layers_cap,
            "notes": self.notes,
        }


_active_family_contract: FamilyMutationContract | None = None


def set_active_family_contract(contract: FamilyMutationContract | None) -> None:
    global _active_family_contract
    _active_family_contract = contract


def clear_active_family_contract() -> None:
    global _active_family_contract
    _active_family_contract = None


def active_family_contract() -> FamilyMutationContract | None:
    return _active_family_contract


def allowed_grid_methods() -> tuple[GridMethod, ...] | None:
    contract = active_family_contract()
    if contract and contract.forced_grid_methods:
        return contract.forced_grid_methods
    return None


def normalize_grid_method_for_contract(grid_method: GridMethod) -> GridMethod:
    """Coerce an out-of-contract method back into the active family."""
    allowed = allowed_grid_methods()
    if allowed and grid_method not in allowed:
        return allowed[0]
    return grid_method
