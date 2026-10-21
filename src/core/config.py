"""Experiment configuration: schema, loading (with `extends`), validation, hashing.

Design rules
------------
* No dataset structure is assumed. Anything that depends on the real data
  (paths, column names, cadence, window lengths, split boundaries, ...) defaults
  to ``None`` = *unresolved*. ``unresolved_fields`` lists what a phase still needs.
* Unknown keys are rejected so a typo can never silently change an experiment.
* The three arms (SoLEXS-only / HEL1OS-only / combined) inherit from one base
  config and may differ ONLY in ``experiment.name`` and ``data.input_configuration``.
  ``comparability_hash`` proves this: it is computed with those two fields removed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

from src.core.errors import ConfigError

# ----------------------------------------------------------------------------
# Allowed values (policy choices, not data assumptions)
# ----------------------------------------------------------------------------
INPUT_CONFIGURATIONS = ("solexs_only", "helios_only", "combined")
DUPLICATE_POLICIES = ("report_only", "keep_first", "keep_last", "error")
MISSING_POLICIES = ("report_only", "error")  # imputation options are added only after inspection
SYNC_METHODS = ("asof_backward", "causal_bin_aggregate")  # both use past observations only
FEATURE_GROUPS = ("levels", "rates", "rolling_stats", "cross_instrument")
SPLIT_METHODS = ("chronological_fractions", "chronological_dates")
IMBALANCE_STRATEGIES = ("none", "class_weight", "controlled_sampling")
THRESHOLD_CRITERIA = ("max_tss_validation", "max_hss_validation")
METRICS = ("tss", "hss", "precision", "recall", "auc", "confusion_matrix", "accuracy")
ABLATION_STEPS = ("levels_only", "levels_rates", "levels_rolling", "all_causal", "combined_instruments")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------
@dataclass
class ExperimentSection:
    name: str = "unnamed"
    seed: int = 42
    description: str = ""


@dataclass
class DataSection:
    input_configuration: str = "combined"
    source_path: Optional[str] = None          # unresolved until the real dataset exists
    # Optional overrides for the adapter. None => adapter must detect and REPORT,
    # never guess. Filled in after schema inspection (Phase 2).
    solexs_source: Optional[str] = None
    helios_source: Optional[str] = None
    time_column: Optional[str] = None
    timestamp_format: Optional[str] = None
    measurement_columns: Optional[list] = None
    quality_flag_columns: Optional[list] = None


@dataclass
class PreprocessingSection:
    sort_chronologically: bool = True
    duplicate_timestamp_policy: str = "report_only"
    missing_value_policy: str = "report_only"
    use_quality_flags: Optional[bool] = None   # cannot decide until flags are known to exist


@dataclass
class SynchronizationSection:
    method: Optional[str] = None               # must be chosen + documented (Phase 4)
    max_staleness_seconds: Optional[float] = None


@dataclass
class LabelsSection:
    horizons_minutes: list = field(default_factory=lambda: [15, 30, 45, 60])
    onset_detection_rule: Optional[str] = "reference_catalogue_onset_utc; positive iff onset is in (t, t+horizon]"
    reference_catalogue_path: Optional[str] = None   # external/reference flare catalogue
    association_tolerance_minutes: Optional[float] = None


@dataclass
class FeaturesSection:
    # Candidate groups. A group is built only if the verified data support it.
    groups: list = field(default_factory=lambda: list(FEATURE_GROUPS))
    history_window_minutes: Optional[float] = None
    rolling_window_minutes: Optional[list] = None


@dataclass
class SplitSection:
    method: str = "chronological_fractions"
    train_fraction: Optional[float] = None
    val_fraction: Optional[float] = None
    test_fraction: Optional[float] = None
    train_end: Optional[str] = None
    val_end: Optional[str] = None
    purge_gap_minutes: Optional[float] = None        # separation between splits (leakage control)


@dataclass
class ImbalanceSection:
    strategy: str = "class_weight"                   # always fitted on the training split only


@dataclass
class XGBoostSection:
    enabled: bool = True
    params: dict = field(default_factory=dict)       # tuned on validation only (Phase 8)
    early_stopping_rounds: Optional[int] = None


@dataclass
class LSTMSection:
    enabled: bool = False                            # Phase 12, only after XGBoost works
    sequence_length: Optional[int] = None
    hidden_units: Optional[int] = None
    layers: Optional[int] = None
    dropout: Optional[float] = None
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    early_stopping_patience: Optional[int] = None


@dataclass
class ModelsSection:
    xgboost: XGBoostSection = field(default_factory=XGBoostSection)
    lstm: LSTMSection = field(default_factory=LSTMSection)


@dataclass
class ThresholdSection:
    criterion: str = "max_tss_validation"
    fixed_before_test: bool = True                   # must stay True (validated)


@dataclass
class EvaluationSection:
    metrics: list = field(default_factory=lambda: ["tss", "hss", "precision", "recall", "auc", "confusion_matrix"])
    lead_time: bool = True
    report_accuracy_supplementary: bool = True


@dataclass
class AblationSection:
    enabled: bool = False
    steps: list = field(default_factory=lambda: list(ABLATION_STEPS))


@dataclass
class InterpretabilitySection:
    feature_importance: bool = True
    shap: bool = False                               # only if computationally practical


@dataclass
class OutputSection:
    runs_dir: str = "experiments/runs"
    log_level: str = "INFO"


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    data: DataSection = field(default_factory=DataSection)
    preprocessing: PreprocessingSection = field(default_factory=PreprocessingSection)
    synchronization: SynchronizationSection = field(default_factory=SynchronizationSection)
    labels: LabelsSection = field(default_factory=LabelsSection)
    features: FeaturesSection = field(default_factory=FeaturesSection)
    split: SplitSection = field(default_factory=SplitSection)
    imbalance: ImbalanceSection = field(default_factory=ImbalanceSection)
    models: ModelsSection = field(default_factory=ModelsSection)
    threshold: ThresholdSection = field(default_factory=ThresholdSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    ablation: AblationSection = field(default_factory=AblationSection)
    interpretability: InterpretabilitySection = field(default_factory=InterpretabilitySection)
    output: OutputSection = field(default_factory=OutputSection)


_SECTION_TYPES = {
    cls.__name__: cls
    for cls in (
        ExperimentSection, DataSection, PreprocessingSection, SynchronizationSection,
        LabelsSection, FeaturesSection, SplitSection, ImbalanceSection, XGBoostSection,
        LSTMSection, ModelsSection, ThresholdSection, EvaluationSection, AblationSection,
        InterpretabilitySection, OutputSection,
    )
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read_yaml(path: Path, _seen: frozenset = frozenset()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"Circular 'extends' involving {path}")
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent is not None:
        base = _read_yaml((path.parent / str(parent)), _seen | {path})
        data = _deep_merge(base, data)
    return data


def _build(cls, data: Any, where: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        nested = _SECTION_TYPES.get(f.type) if isinstance(f.type, str) else None
        if nested is not None:
            value = _build(nested, value, f"{where}.{name}" if where else name)
        kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict) -> ExperimentConfig:
    cfg = _build(ExperimentConfig, data, "")
    errors = validate(cfg)
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return cfg


def load_config(path) -> ExperimentConfig:
    """Load a YAML config (supports `extends: other.yaml`) and validate it."""
    return config_from_dict(_read_yaml(Path(path)))


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _in(errors, value, allowed, label):
    if value not in allowed:
        errors.append(f"{label}={value!r} not in {list(allowed)}")


def _subset(errors, values, allowed, label):
    if not isinstance(values, list) or not values:
        errors.append(f"{label} must be a non-empty list")
        return
    bad = [v for v in values if v not in allowed]
    if bad:
        errors.append(f"{label} contains unsupported value(s) {bad}; allowed: {list(allowed)}")


def validate(cfg: ExperimentConfig) -> list:
    """Return a list of human-readable problems (empty list == valid)."""
    e: list = []
    if not isinstance(cfg.experiment.name, str) or not cfg.experiment.name.strip():
        e.append("experiment.name must be a non-empty string")
    if isinstance(cfg.experiment.seed, bool) or not isinstance(cfg.experiment.seed, int):
        e.append("experiment.seed must be an integer")

    _in(e, cfg.data.input_configuration, INPUT_CONFIGURATIONS, "data.input_configuration")
    _in(e, cfg.preprocessing.duplicate_timestamp_policy, DUPLICATE_POLICIES, "preprocessing.duplicate_timestamp_policy")
    _in(e, cfg.preprocessing.missing_value_policy, MISSING_POLICIES, "preprocessing.missing_value_policy")
    if cfg.synchronization.method is not None:
        _in(e, cfg.synchronization.method, SYNC_METHODS, "synchronization.method")

    h = cfg.labels.horizons_minutes
    if (not isinstance(h, list) or not h
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in h)
            or len(set(h)) != len(h)):
        e.append("labels.horizons_minutes must be a list of unique positive integers")

    _subset(e, cfg.features.groups, FEATURE_GROUPS, "features.groups")
    _in(e, cfg.split.method, SPLIT_METHODS, "split.method")
    fr = [cfg.split.train_fraction, cfg.split.val_fraction, cfg.split.test_fraction]
    if any(x is not None for x in fr):
        if any(x is None for x in fr):
            e.append("split fractions: give all of train/val/test or none")
        elif any(not (0 < x < 1) for x in fr) or abs(sum(fr) - 1.0) > 1e-9:
            e.append("split fractions must each be in (0,1) and sum to 1")

    _in(e, cfg.imbalance.strategy, IMBALANCE_STRATEGIES, "imbalance.strategy")
    _in(e, cfg.threshold.criterion, THRESHOLD_CRITERIA, "threshold.criterion")
    if cfg.threshold.fixed_before_test is not True:
        e.append("threshold.fixed_before_test must be true (never optimise the threshold on test data)")
    _subset(e, cfg.evaluation.metrics, METRICS, "evaluation.metrics")
    _subset(e, cfg.ablation.steps, ABLATION_STEPS, "ablation.steps")
    _in(e, cfg.output.log_level, LOG_LEVELS, "output.log_level")
    if not (cfg.models.xgboost.enabled or cfg.models.lstm.enabled):
        e.append("at least one model must be enabled")
    return e


# ----------------------------------------------------------------------------
# Unresolved-field reporting  ("Cannot determine until the dataset is inspected")
# ----------------------------------------------------------------------------
# path -> (phase that needs it, why)
_REQUIRED_BY_PHASE = {
    "data.source_path": (2, "location of the real dataset"),
    "labels.onset_detection_rule": (5, "documented rule for validated flare onset"),
    "features.history_window_minutes": (6, "causal history window length"),
    "split.train_fraction": (7, "chronological split (or dates) - depends on event counts"),
    "split.purge_gap_minutes": (7, "gap that stops overlapping windows straddling splits"),
}
_COMBINED_ONLY = {
    "synchronization.method": (4, "documented, non-look-ahead synchronization policy"),
}


def _get(cfg, dotted: str):
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def unresolved_fields(cfg: ExperimentConfig) -> list:
    """Fields that later phases need but that are still unset. Sorted by phase."""
    needed = dict(_REQUIRED_BY_PHASE)
    if cfg.data.input_configuration == "combined":
        needed.update(_COMBINED_ONLY)
    # A concrete per-instrument source satisfies the Phase-2 source requirement.
    if cfg.data.input_configuration == "solexs_only" and cfg.data.solexs_source:
        needed.pop("data.source_path", None)
    elif cfg.data.input_configuration == "helios_only" and cfg.data.helios_source:
        needed.pop("data.source_path", None)
    elif cfg.data.input_configuration == "combined" and cfg.data.solexs_source and cfg.data.helios_source:
        needed.pop("data.source_path", None)
    if cfg.split.method == "chronological_dates":
        needed.pop("split.train_fraction", None)
        needed["split.train_end"] = (7, "end of the training period")
        needed["split.val_end"] = (7, "end of the validation period")
    out = [(path, phase, why) for path, (phase, why) in needed.items() if _get(cfg, path) is None]
    return sorted(out, key=lambda t: (t[1], t[0]))


# ----------------------------------------------------------------------------
# Hashing / serialisation
# ----------------------------------------------------------------------------
def to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)


def _hash(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_hash(cfg: ExperimentConfig) -> str:
    """Hash of the full resolved config (description excluded: prose only)."""
    d = to_dict(cfg)
    d["experiment"].pop("description", None)
    return _hash(d)


def comparability_hash(cfg: ExperimentConfig) -> str:
    """Hash with the ONLY two fields allowed to differ between arms removed.

    Equal comparability hashes => SoLEXS-only, HEL1OS-only and combined runs share
    target, horizons, window, preprocessing, split, imbalance, models, threshold,
    metrics, seed - i.e. the comparison is controlled.
    """
    d = to_dict(cfg)
    d["experiment"].pop("name", None)
    d["experiment"].pop("description", None)
    d["data"].pop("input_configuration", None)
    # Source locations are operational paths, not experimental settings. They
    # may legitimately differ between local machines/arms without changing the
    # scientific comparison protocol.
    for key in ("source_path", "solexs_source", "helios_source"):
        d["data"].pop(key, None)
    return _hash(d)


def dump_yaml(cfg: ExperimentConfig, path) -> None:
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False), encoding="utf-8")
