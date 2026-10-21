"""Experiment identity, run directories, manifests and seeding (reproducibility)."""
from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional

from src import FRAMEWORK_PHASE
from src.core.config import (
    ExperimentConfig, comparability_hash, config_hash, dump_yaml, unresolved_fields,
)

_TRACKED_PACKAGES = ("PyYAML", "numpy", "pandas", "scikit-learn", "xgboost", "shap",
                     "torch", "tensorflow", "matplotlib")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "experiment"


def make_experiment_id(cfg: ExperimentConfig, now: Optional[datetime] = None) -> str:
    """<name>-<UTC timestamp>-<8-char config hash>.

    The hash part means two runs with identical configs are recognisably
    the same experiment definition; the timestamp keeps run directories unique.
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slugify(cfg.experiment.name)}-{stamp}-{config_hash(cfg)[:8]}"


def create_run_dir(runs_dir, experiment_id: str) -> Path:
    run_dir = Path(runs_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)   # never overwrite an earlier run
    for sub in ("figures", "tables", "models"):
        (run_dir / sub).mkdir()
    return run_dir


def seed_everything(seed: int) -> dict:
    """Seed the RNGs available now. Returns what was actually seeded (for the log).

    Deep-learning frameworks are seeded when they are introduced (Phase 12).
    """
    seeded = {"python.random": seed}
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)   # affects child processes only
    seeded["PYTHONHASHSEED(env, child processes)"] = seed
    try:
        import numpy as np
        np.random.seed(seed)
        seeded["numpy.random"] = seed
    except ImportError:
        pass
    return seeded


def git_commit(repo_root) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None      # not installed (yet)
    return versions


def write_run_metadata(cfg: ExperimentConfig, run_dir: Path, experiment_id: str,
                       repo_root, seeded: dict) -> dict:
    """Write config.resolved.yaml and manifest.json into the run directory."""
    dump_yaml(cfg, run_dir / "config.resolved.yaml")
    manifest = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework_phase": FRAMEWORK_PHASE,
        "config_hash": config_hash(cfg),
        "comparability_hash": comparability_hash(cfg),
        "input_configuration": cfg.data.input_configuration,
        "seed": cfg.experiment.seed,
        "seeded": seeded,
        "git_commit": git_commit(repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "unresolved_fields": [
            {"field": p, "needed_in_phase": ph, "reason": why}
            for p, ph, why in unresolved_fields(cfg)
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
