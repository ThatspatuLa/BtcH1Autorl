"""Mutation contracts for Stage 1 spacing-family hyperopt runs.

These contracts keep a family focused on its spacing identity while allowing
allocation, depth, and fixed TP to continue mutating inside that identity.

Stage 2 extends the pattern with `ComboMutationContract` for combo runs — the
combo contract carries the per-layer zones and per-family allowed grid methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genome.schema import ConfirmationIndicator, GridMethod, GridZoneSpec


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


@dataclass(frozen=True)
class ComboMutationContract:
    """Rules for a Stage 2 combo run.

    A combo binds multiple families to per-layer zones. Each zone's grid_method
    is constrained to that family's allowed methods. Allocation/depth/cooldown
    remain free to mutate across all candidates in the combo.

    Zones are immutable across mutation/crossover (the combo contract is fixed).
    """

    name: str
    families: tuple[str, ...]
    split_strategy: str
    zones: tuple[GridZoneSpec, ...]
    family_grid_methods: tuple[tuple[GridMethod, ...], ...]
    max_dca_layers: int
    stage: int = 2
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "families": list(self.families),
            "split_strategy": self.split_strategy,
            "zones": [z.to_dict() for z in self.zones],
            "family_grid_methods": [
                [m.value for m in methods] for methods in self.family_grid_methods
            ],
            "max_dca_layers": self.max_dca_layers,
            "notes": self.notes,
        }

    @property
    def n_families(self) -> int:
        return len(self.families)


_active_family_contract: FamilyMutationContract | None = None
_active_combo_contract: ComboMutationContract | None = None


def set_active_family_contract(contract: FamilyMutationContract | None) -> None:
    global _active_family_contract
    _active_family_contract = contract


def clear_active_family_contract() -> None:
    global _active_family_contract
    _active_family_contract = None


def active_family_contract() -> FamilyMutationContract | None:
    return _active_family_contract


def set_active_combo_contract(contract: ComboMutationContract | None) -> None:
    global _active_combo_contract
    _active_combo_contract = contract


def clear_active_combo_contract() -> None:
    global _active_combo_contract
    _active_combo_contract = None


def active_combo_contract() -> ComboMutationContract | None:
    return _active_combo_contract


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


def combo_contract_from_spec(spec) -> ComboMutationContract:
    """Build a ComboMutationContract from a ComboSpec (combo_specs.py).

    Keeps the family_contract ↔ combo_spec wiring in one place. The contract
    captures: families (immutable names), zones (immutable per-layer mapping),
    family_grid_methods (per-family allowed GridMethods), and the split strategy.
    """
    # Import here to avoid circular import (combo_specs imports FamilySpec from
    # hyperopt_config which imports from family_contracts via FamilyMutationContract).
    from evolution.combo_specs import ComboSpec as _ComboSpec

    if not isinstance(spec, _ComboSpec):
        raise TypeError(f"expected ComboSpec, got {type(spec).__name__}")

    # Each family's allowed methods come from its FamilySpec. We pack them into
    # a tuple of tuples — parallel to spec.families.
    from evolution.hyperopt_config import build_family_specs

    family_specs = {f.name: f for f in build_family_specs()}
    family_grid_methods: list[tuple[GridMethod, ...]] = []
    for fam_name in spec.families:
        fs = family_specs.get(fam_name)
        if fs is None:
            raise KeyError(f"Unknown family in combo: {fam_name!r}")
        if fs.forced_grid_methods:
            family_grid_methods.append(tuple(fs.forced_grid_methods))
        else:
            # No forced methods — allow anything in this zone (fallback)
            from evolution.operators import ALL_GRID_METHODS

            family_grid_methods.append(tuple(ALL_GRID_METHODS))

    return ComboMutationContract(
        name=spec.name,
        families=tuple(spec.families),
        split_strategy=spec.split_strategy,
        zones=tuple(spec.zones),
        family_grid_methods=tuple(family_grid_methods),
        max_dca_layers=spec.max_dca_layers,
        notes=spec.description,
    )
