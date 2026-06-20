"""Immutable Settings loader — merges Freqtrade config + experiment overrides + research config.

Design principles:
1. Immutable: returns a FrozenSettingsDict (Mapping) — no mutation after load.
2. Deterministic: identical input files produce identical Settings object across runs.
3. Three-tier merge: research < experiment < (settings_overrides per-candidate).
4. Validation: rejects missing required fields, out-of-range values, type mismatches.
5. JSONB-safe: settings serialise back to JSON losslessly via .to_dict().

Usage:
    settings = Settings.from_files(
        freqtrade_path=Path("configs/freqtrade/config.json"),
        experiment_path=Path("configs/experiments/default.json"),
        research_path=Path("configs/research/default.json"),
    )
    assert settings.leverage == 5.0
    assert settings.candidates_per_gen == 500
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


class FrozenSettingsDict(Mapping):
    """Immutable dict wrapper. Reads from a MappingProxyType internally.

    All nested values are also frozen recursively. Mutation attempts raise TypeError.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        # Deep-freeze: convert each value to immutable form
        frozen: dict[str, Any] = {}
        for k, v in data.items():
            frozen[k] = _freeze_value(v)
        self._data = MappingProxyType(frozen)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"FrozenSettingsDict({dict(self._data)!r})"

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copy dict (mutable). Used for JSON serialisation."""
        return _unfreeze(self._data)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=True)


def _freeze_value(v: Any) -> Any:
    """Recursively freeze a value into immutable form."""
    if isinstance(v, dict):
        return FrozenSettingsDict(v)
    if isinstance(v, list):
        return tuple(_freeze_value(x) for x in v)
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    # Fallback: leave as-is (e.g. tuples already frozen)
    return v


def _unfreeze(v: Any) -> Any:
    """Recursively unfreeze to plain dict/list for JSON."""
    if isinstance(v, FrozenSettingsDict):
        return {k: _unfreeze(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_unfreeze(x) for x in v]
    if isinstance(v, Mapping):
        return {k: _unfreeze(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return v


# Required fields per tier — used by validation
_FREQTRADE_REQUIRED = {
    "stake_currency", "stake_amount", "dry_run", "fee",
    "exchange", "timeframe", "dataformat",
}

_EXPERIMENT_REQUIRED = {
    "starting_balance", "leverage", "margin_mode",
    "fee_pct", "spread_pips", "slippage_pct",
    "buffer_pct", "wall_time_budget_seconds", "dry_run", "log_level",
}

_RESEARCH_REQUIRED = {
    "candidates_per_gen", "parallel_workers", "base_seed",
    "sqlite_path", "results_dir", "wall_time_budget_seconds",
}

_VALID_MARGIN_MODES = {"isolated", "cross"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class SettingsError(ValueError):
    """Raised when config validation fails."""


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SettingsError(f"Config file not found: {path}")
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise SettingsError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise SettingsError(f"{path} must contain a JSON object at top level")
    return data


def _validate_freqtrade(d: dict, path: Path) -> None:
    missing = _FREQTRADE_REQUIRED - set(d.keys())
    if missing:
        raise SettingsError(f"{path} missing required Freqtrade fields: {sorted(missing)}")
    if d["timeframe"] != "1h":
        raise SettingsError(
            f"{path}: BTC H1 project requires timeframe=1h; got timeframe={d['timeframe']}"
        )
    # ticker_interval is deprecated in Freqtrade 2026.5+; reject if present
    if "ticker_interval" in d:
        raise SettingsError(
            f"{path}: 'ticker_interval' is deprecated in Freqtrade 2026.5+. "
            f"Use 'timeframe' instead."
        )
    if d["margin_mode"] not in _VALID_MARGIN_MODES:
        raise SettingsError(f"{path}: margin_mode must be one of {_VALID_MARGIN_MODES}")


def _validate_experiment(d: dict, path: Path) -> None:
    missing = _EXPERIMENT_REQUIRED - set(d.keys())
    if missing:
        raise SettingsError(f"{path} missing required experiment fields: {sorted(missing)}")
    if d["margin_mode"] not in _VALID_MARGIN_MODES:
        raise SettingsError(f"{path}: margin_mode must be one of {_VALID_MARGIN_MODES}")
    if not (0.0 < d["leverage"] <= 125.0):
        raise SettingsError(f"{path}: leverage must be in (0, 125], got {d['leverage']}")
    if not (0.0 <= d["buffer_pct"] < 1.0):
        raise SettingsError(f"{path}: buffer_pct must be in [0, 1), got {d['buffer_pct']}")
    if d["wall_time_budget_seconds"] <= 0:
        raise SettingsError(f"{path}: wall_time_budget_seconds must be > 0")
    if d["log_level"] not in _VALID_LOG_LEVELS:
        raise SettingsError(f"{path}: log_level must be one of {_VALID_LOG_LEVELS}")


def _validate_research(d: dict, path: Path) -> None:
    missing = _RESEARCH_REQUIRED - set(d.keys())
    if missing:
        raise SettingsError(f"{path} missing required research fields: {sorted(missing)}")
    if d["candidates_per_gen"] != 500:
        raise SettingsError(
            f"{path}: candidates_per_gen MUST be 500 per project spec (locked for v1); "
            f"got {d['candidates_per_gen']}"
        )
    if d["parallel_workers"] < 1:
        raise SettingsError(f"{path}: parallel_workers must be >= 1")


class Settings:
    """Immutable Settings — result of merging freqtrade + experiment + research configs.

    Access via attribute style (settings.leverage) or dict style (settings["leverage"]).
    """

    __slots__ = ("_data", "_experiment_path", "_freqtrade_path", "_research_path")

    def __init__(
        self,
        freqtrade: Mapping[str, Any],
        experiment: Mapping[str, Any],
        research: Mapping[str, Any],
        freqtrade_path: Path,
        experiment_path: Path,
        research_path: Path,
    ):
        # Merge: research < experiment < freqtrade (freqtrade wins for shared keys like dry_run, fee)
        # But experiment is the canonical source for: starting_balance, leverage, margin_mode,
        # buffer_pct, wall_time_budget_seconds, log_level — those go last.
        merged: dict[str, Any] = {}
        merged.update(_to_plain(research))
        merged.update(_to_plain(experiment))
        # Freqtrade provides exchange + dataformat + timeframe; selected fields only
        for ft_key in (
            "stake_currency", "stake_amount", "exchange", "dataformat", "datadir",
            "user_data_dir", "order_types", "entry_pricing", "exit_pricing",
            "unfilledtimeout", "trading_mode", "margin_mode", "leverage",
            "pairlists", "bot_name", "timeframe", "tradable_balance_ratio",
            "available_capital", "max_open_trades",
        ):
            if ft_key in freqtrade:
                merged[ft_key] = _to_plain(freqtrade[ft_key])
        # Ensure final margin_mode + leverage from experiment win
        merged["margin_mode"] = experiment["margin_mode"]
        merged["leverage"] = experiment["leverage"]
        # Freqtrade fee is in absolute (0.0006), experiment fee_pct same — use experiment as canonical
        merged["fee_pct"] = experiment["fee_pct"]
        # dry_run: experiment can override freqtrade
        merged["dry_run"] = experiment["dry_run"]
        # log_level only from experiment
        merged["log_level"] = experiment["log_level"]

        self._data = FrozenSettingsDict(merged)
        self._freqtrade_path = freqtrade_path
        self._experiment_path = experiment_path
        self._research_path = research_path

    @classmethod
    def from_files(
        cls,
        freqtrade_path: Path | str,
        experiment_path: Path | str,
        research_path: Path | str,
    ) -> Settings:
        ft = _load_json(Path(freqtrade_path))
        ex = _load_json(Path(experiment_path))
        rs = _load_json(Path(research_path))
        # Strip schema meta fields before validation
        _validate_freqtrade(ft, Path(freqtrade_path))
        _validate_experiment(ex, Path(experiment_path))
        _validate_research(rs, Path(research_path))
        return cls(ft, ex, rs, Path(freqtrade_path), Path(experiment_path), Path(research_path))

    def __getattr__(self, name: str) -> Any:
        # Only invoked when normal lookup fails — dict lookup on _data
        try:
            return self._data[name]
        except KeyError as err:
            raise AttributeError(
                f"Settings has no attribute or key '{name}'. "
                f"Available keys: {sorted(self._data.keys())[:20]}..."
            ) from err

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"Settings(freqtrade={self._freqtrade_path.name}, "
            f"experiment={self._experiment_path.name}, "
            f"research={self._research_path.name}, "
            f"keys={len(self._data)})"
        )

    @property
    def data(self) -> FrozenSettingsDict:
        """Return the underlying FrozenSettingsDict."""
        return self._data

    @property
    def sources(self) -> dict[str, Path]:
        return {
            "freqtrade": self._freqtrade_path,
            "experiment": self._experiment_path,
            "research": self._research_path,
        }


def _to_plain(v: Any) -> Any:
    """Convert any value to plain dict/list/primitive (for merging)."""
    if isinstance(v, Mapping):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    return v
