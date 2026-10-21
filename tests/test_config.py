from pathlib import Path

import pytest

from src.core.config import (
    ConfigError, comparability_hash, config_from_dict, config_hash, load_config, unresolved_fields,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
ARMS = ("solexs", "helios", "combined")


def test_all_configs_load():
    for name in ("base", *ARMS):
        assert load_config(CONFIGS / f"{name}.yaml") is not None


def test_arm_input_configurations():
    got = {a: load_config(CONFIGS / f"{a}.yaml").data.input_configuration for a in ARMS}
    assert got == {"solexs": "solexs_only", "helios": "helios_only", "combined": "combined"}


def test_arms_are_comparable_by_construction():
    cfgs = [load_config(CONFIGS / f"{a}.yaml") for a in ARMS]
    assert len({comparability_hash(c) for c in cfgs}) == 1      # identical apart from name/input
    assert len({config_hash(c) for c in cfgs}) == 3             # but they ARE different experiments


def test_comparability_hash_detects_a_drifting_setting():
    a = load_config(CONFIGS / "solexs.yaml")
    b = load_config(CONFIGS / "helios.yaml")
    b.labels.horizons_minutes = [15, 30]
    assert comparability_hash(a) != comparability_hash(b)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"data": {"soruce_path": "x"}})


def test_bad_enum_rejected():
    with pytest.raises(ConfigError):
        config_from_dict({"data": {"input_configuration": "everything"}})


def test_random_split_not_allowed():
    with pytest.raises(ConfigError):
        config_from_dict({"split": {"method": "random"}})


def test_threshold_cannot_be_tuned_on_test():
    with pytest.raises(ConfigError, match="fixed_before_test"):
        config_from_dict({"threshold": {"fixed_before_test": False}})


def test_split_fractions_must_sum_to_one_or_be_absent():
    with pytest.raises(ConfigError):
        config_from_dict({"split": {"train_fraction": 0.5, "val_fraction": 0.2, "test_fraction": 0.2}})
    with pytest.raises(ConfigError):
        config_from_dict({"split": {"train_fraction": 0.7}})
    config_from_dict({"split": {"train_fraction": 0.7, "val_fraction": 0.15, "test_fraction": 0.15}})


def test_horizons_validated():
    for bad in ([], [0], [15, 15], [-5], ["15"]):
        with pytest.raises(ConfigError):
            config_from_dict({"labels": {"horizons_minutes": bad}})


def test_nothing_about_the_dataset_is_assumed_by_default():
    cfg = load_config(CONFIGS / "base.yaml")
    d = cfg.data
    assert d.source_path is None and d.time_column is None and d.timestamp_format is None
    assert d.measurement_columns is None and d.quality_flag_columns is None
    assert cfg.synchronization.method == "asof_backward"
    assert cfg.synchronization.max_staleness_seconds == 1.0
    assert cfg.labels.onset_detection_rule is not None and cfg.features.history_window_minutes is None


def test_unresolved_fields_report():
    combined = {p for p, _, _ in unresolved_fields(load_config(CONFIGS / "combined.yaml"))}
    single = {p for p, _, _ in unresolved_fields(load_config(CONFIGS / "solexs.yaml"))}
    assert "synchronization.method" not in combined
    assert "synchronization.method" not in single       # not needed for a single instrument
    assert "data.source_path" in combined
    assert "features.history_window_minutes" not in combined


def test_extends_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_lstm_disabled_and_xgboost_primary_by_default():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.models.xgboost.enabled and not cfg.models.lstm.enabled
