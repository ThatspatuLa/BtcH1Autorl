"""Stage 1 acceptance tests — config loader.

Verifies:
- Settings loads from committed config files without errors
- All required fields present and correctly typed
- FrozenSettingsDict is immutable
- Loader is deterministic (same input → same Settings)
- Validation catches missing fields, out-of-range values, type errors
"""
from __future__ import annotations

import pytest

from configs.loader import FrozenSettingsDict, Settings, SettingsError

pytestmark = pytest.mark.stage1


def test_settings_loads_from_default_configs(settings):
    """The three default config files must load without errors."""
    assert isinstance(settings, Settings)
    assert isinstance(settings.data, FrozenSettingsDict)


def test_settings_contains_required_broker_fields(settings):
    """Locked broker/account fields must be present and correctly typed."""
    assert settings.starting_balance == 10000.0
    assert isinstance(settings.starting_balance, float)
    assert settings.leverage == 5.0
    assert settings.margin_mode == "isolated"
    assert settings.dry_run is True
    assert settings.fee_pct == 0.0006
    assert settings.spread_pips == 5.0
    assert settings.slippage_pct == 0.0005
    assert settings.funding_rate_per_8h == 0.0001
    assert settings.maintenance_margin_pct == 0.005
    assert settings.liquidation_buffer_pct == 0.05
    assert settings.contract_size == 1.0
    assert settings.min_trade_size == 0.0001
    assert settings.stake_precision == 2
    assert settings.amount_precision == 6


def test_settings_contains_required_research_fields(settings):
    """Evolution-engine config fields must be locked to spec values."""
    assert settings.candidates_per_gen == 500
    assert settings.parallel_workers == 8
    assert settings.base_seed == 42
    assert settings.wall_time_budget_seconds == 28800  # 8h cap (LOCKED)


def test_buffer_pct_is_configurable_not_hardcoded(settings):
    """Buffer must be read from settings, not hardcoded."""
    assert settings.buffer_pct == 0.20  # default


def test_freqtrade_compatibility(settings):
    """Settings must include Freqtrade-required fields for config compatibility."""
    assert settings.stake_currency == "USDT"
    assert settings.timeframe == "1h"
    assert settings.dataformat == "feather"
    assert "name" in settings.exchange
    assert settings.exchange["name"] == "kraken"
    assert "BTC/USDT" in settings.exchange["pair_whitelist"]
    # ticker_interval deprecated and must NOT appear
    assert "ticker_interval" not in settings


def test_settings_is_immutable(settings):
    """FrozenSettingsDict must reject mutation attempts."""
    with pytest.raises(TypeError):
        settings.data["leverage"] = 999.0  # type: ignore[index]


def test_settings_attribute_access(settings):
    """Settings must support both attribute and item access."""
    assert settings["leverage"] == 5.0
    assert settings.leverage == 5.0
    assert "leverage" in settings
    assert len(settings) > 10  # lots of keys


def test_settings_is_deterministic(project_root):
    """Two loads of the same files must produce identical Settings."""
    s1 = Settings.from_files(
        freqtrade_path=project_root / "configs" / "freqtrade" / "config.json",
        experiment_path=project_root / "configs" / "experiments" / "default.json",
        research_path=project_root / "configs" / "research" / "default.json",
    )
    s2 = Settings.from_files(
        freqtrade_path=project_root / "configs" / "freqtrade" / "config.json",
        experiment_path=project_root / "configs" / "experiments" / "default.json",
        research_path=project_root / "configs" / "research" / "default.json",
    )
    # Same keys, same values, same order
    assert list(s1.data.keys()) == list(s2.data.keys())
    for k in s1.data:
        assert s1[k] == s2[k], f"mismatch on key {k}"


def test_to_json_roundtrip(settings):
    """Settings must serialise to JSON losslessly."""
    j = settings.data.to_json()
    import json
    parsed = json.loads(j)
    assert parsed["leverage"] == 5.0
    assert parsed["candidates_per_gen"] == 500
    assert parsed["buffer_pct"] == 0.20
    assert parsed["margin_mode"] == "isolated"


def test_settings_rejects_missing_file(project_root):
    """Loader must raise SettingsError if a config file is missing."""
    with pytest.raises(SettingsError, match="not found"):
        Settings.from_files(
            freqtrade_path=project_root / "configs" / "freqtrade" / "config.json",
            experiment_path=project_root / "nonexistent.json",
            research_path=project_root / "configs" / "research" / "default.json",
        )


def test_settings_rejects_wrong_timeframe(project_root, tmp_path):
    """Loader must reject Freqtrade config with wrong timeframe (project is H1)."""
    import json
    bad_ft = tmp_path / "bad_ft.json"
    cfg = json.loads((project_root / "configs" / "freqtrade" / "config.json").read_text())
    cfg["timeframe"] = "5m"
    bad_ft.write_text(json.dumps(cfg))
    with pytest.raises(SettingsError, match="timeframe=1h"):
        Settings.from_files(
            freqtrade_path=bad_ft,
            experiment_path=project_root / "configs" / "experiments" / "default.json",
            research_path=project_root / "configs" / "research" / "default.json",
        )


def test_settings_rejects_deprecated_ticker_interval(project_root, tmp_path):
    """ticker_interval is deprecated in Freqtrade 2026.5+ and must be rejected."""
    import json
    bad_ft = tmp_path / "bad_ft.json"
    cfg = json.loads((project_root / "configs" / "freqtrade" / "config.json").read_text())
    cfg["ticker_interval"] = "1h"  # old key reintroduced
    bad_ft.write_text(json.dumps(cfg))
    with pytest.raises(SettingsError, match=r"ticker_interval.*deprecated"):
        Settings.from_files(
            freqtrade_path=bad_ft,
            experiment_path=project_root / "configs" / "experiments" / "default.json",
            research_path=project_root / "configs" / "research" / "default.json",
        )


def test_settings_rejects_wrong_candidates_per_gen(project_root, tmp_path):
    """LOCKED for v1: candidates_per_gen must be exactly 500."""
    import json
    bad_rs = tmp_path / "bad_rs.json"
    cfg = json.loads((project_root / "configs" / "research" / "default.json").read_text())
    cfg["candidates_per_gen"] = 250
    bad_rs.write_text(json.dumps(cfg))
    with pytest.raises(SettingsError, match="candidates_per_gen MUST be 500"):
        Settings.from_files(
            freqtrade_path=project_root / "configs" / "freqtrade" / "config.json",
            experiment_path=project_root / "configs" / "experiments" / "default.json",
            research_path=bad_rs,
        )


def test_settings_rejects_bad_buffer_pct(project_root, tmp_path):
    """Buffer_pct must be in [0, 1)."""
    import json
    bad_ex = tmp_path / "bad_ex.json"
    cfg = json.loads((project_root / "configs" / "experiments" / "default.json").read_text())
    cfg["buffer_pct"] = 1.5
    bad_ex.write_text(json.dumps(cfg))
    with pytest.raises(SettingsError, match="buffer_pct must be in"):
        Settings.from_files(
            freqtrade_path=project_root / "configs" / "freqtrade" / "config.json",
            experiment_path=bad_ex,
            research_path=project_root / "configs" / "research" / "default.json",
        )


def test_settings_rejects_bad_leverage(project_root, tmp_path):
    """Leverage must be in (0, 125]."""
    import json
    bad_ex = tmp_path / "bad_ex.json"
    cfg = json.loads((project_root / "configs" / "experiments" / "default.json").read_text())
    cfg["leverage"] = 200.0
    bad_ex.write_text(json.dumps(cfg))
    with pytest.raises(SettingsError, match="leverage must be in"):
        Settings.from_files(
            freqtrade_path=project_root / "configs" / "freqtrade" / "config.json",
            experiment_path=bad_ex,
            research_path=project_root / "configs" / "research" / "default.json",
        )
