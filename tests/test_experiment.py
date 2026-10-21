import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.config import load_config
from src.core.experiment import (
    create_run_dir, make_experiment_id, seed_everything, slugify, write_run_metadata,
)
from src.core.logging_setup import TransformationLog, setup_logging

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
FIXED_NOW = datetime(2026, 9, 24, 12, 30, 5, tzinfo=timezone.utc)


def test_slugify():
    assert slugify("SoLEXS only!") == "solexs-only"
    assert slugify("###") == "experiment"


def test_experiment_id_format_and_determinism():
    cfg = load_config(CONFIGS / "combined.yaml")
    a = make_experiment_id(cfg, now=FIXED_NOW)
    assert a == make_experiment_id(cfg, now=FIXED_NOW)
    name, stamp, h = a.rsplit("-", 2)
    assert (name, stamp) == ("combined", "20260924T123005Z") and len(h) == 8


def test_experiment_id_changes_with_config():
    cfg = load_config(CONFIGS / "combined.yaml")
    other = load_config(CONFIGS / "combined.yaml")
    other.experiment.seed = 7
    assert make_experiment_id(cfg, FIXED_NOW) != make_experiment_id(other, FIXED_NOW)


def test_run_dir_never_overwritten(tmp_path):
    create_run_dir(tmp_path, "x")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "x")


def test_seed_is_reproducible():
    import random
    seed_everything(123); a = random.random()
    seed_everything(123); b = random.random()
    assert a == b


def test_manifest_contents(tmp_path):
    cfg = load_config(CONFIGS / "combined.yaml")
    run_dir = create_run_dir(tmp_path, "run1")
    seeded = seed_everything(cfg.experiment.seed)
    write_run_metadata(cfg, run_dir, "run1", tmp_path, seeded)
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["experiment_id"] == "run1" and m["framework_phase"] == 5
    assert m["seed"] == 42 and m["input_configuration"] == "combined"
    assert m["git_commit"] is None                        # tmp_path is not a git repo: reported, not invented
    assert any(u["field"] == "data.source_path" for u in m["unresolved_fields"])
    assert (run_dir / "config.resolved.yaml").exists()


def test_logging_writes_file_and_is_idempotent(tmp_path):
    setup_logging(tmp_path, "INFO")
    logger = setup_logging(tmp_path, "INFO")               # second call must not duplicate handlers
    assert len(logger.handlers) == 2
    logging.getLogger("solar_flare.test").info("hello")
    for h in logger.handlers:
        h.flush()
    assert "hello" in (tmp_path / "run.log").read_text()


def test_transformation_log_records_row_counts(tmp_path):
    setup_logging(tmp_path)
    tl = TransformationLog(tmp_path)
    entry = tl.record("preprocessing", "drop_duplicates", rows_before=100, rows_after=97, policy="keep_first")
    assert entry["rows_removed"] == 3
    assert tl.read()[0]["details"]["policy"] == "keep_first"


def test_run_experiment_entrypoint(tmp_path, capsys):
    import time
    import run_experiment
    args = ["--config", str(CONFIGS / "solexs.yaml"), "--runs-dir", str(tmp_path)]
    assert run_experiment.main(args + ["--scaffold-check"]) == 0
    assert run_experiment.main(["--config", str(tmp_path / "missing.yaml")]) == 1
    time.sleep(1.1)                                        # distinct experiment id (second resolution)
    assert run_experiment.main(args) == 2                  # honest: stops when source data are absent
    runs = sorted(p.name for p in tmp_path.iterdir())
    assert len(runs) == 2 and all(r.startswith("solexs-only-") for r in runs)


def test_same_second_collision_is_a_clean_error(tmp_path, monkeypatch, capsys):
    import run_experiment
    monkeypatch.setattr(run_experiment, "make_experiment_id", lambda cfg: "fixed-id")
    args = ["--config", str(CONFIGS / "solexs.yaml"), "--runs-dir", str(tmp_path), "--scaffold-check"]
    assert run_experiment.main(args) == 0
    assert run_experiment.main(args) == 1
    assert "never overwritten" in capsys.readouterr().err
