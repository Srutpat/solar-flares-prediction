#!/usr/bin/env python3
"""Entry point:  python run_experiment.py --config configs/combined.yaml

Phase 1: creates the experiment identity, run directory, resolved config,
manifest and logs, reports unresolved settings and per-stage implementation
status, then stops at the first placeholder stage (honestly: nothing is run yet).
"""
from __future__ import annotations

import argparse
import sys
import warnings, pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src import FRAMEWORK_PHASE  # noqa: E402
from src.core.config import load_config, unresolved_fields  # noqa: E402
from src.core.errors import ConfigError  # noqa: E402
from src.core.experiment import (  # noqa: E402
    create_run_dir, make_experiment_id, seed_everything, write_run_metadata,
)
from src.core.logging_setup import TransformationLog, get_logger, setup_logging  # noqa: E402
from src.experiments.pipeline import format_status_table, run_pipeline  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one configured experiment.")
    ap.add_argument("--config", required=True, help="YAML experiment config")
    ap.add_argument("--runs-dir", default=None, help="override output.runs_dir")
    ap.add_argument("--data-source", default=None, help="override data.source_path")
    ap.add_argument("--solexs-source", default=None, help="override data.solexs_source")
    ap.add_argument("--helios-source", nargs="+", default=None, help="one or more HEL1OS FITS/ZIP sources (supports split daily archives)")
    ap.add_argument("--reference-catalogue", default=None, help="CSV of externally validated flare onsets; requires onset_utc and source columns")
    ap.add_argument("--scaffold-check", action="store_true",
                    help="initialise the run and report status; exit 0 even though stages are placeholders")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    # Dataset paths are operational overrides. They are intentionally not part
    # of the three-arm comparability hash.
    if args.data_source:
        cfg.data.source_path = args.data_source
    if args.solexs_source:
        cfg.data.solexs_source = args.solexs_source
    if args.helios_source:
        cfg.data.helios_source = args.helios_source
    if args.reference_catalogue:
        cfg.labels.reference_catalogue_path = args.reference_catalogue

    runs_dir = Path(args.runs_dir or cfg.output.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir

    exp_id = make_experiment_id(cfg)
    try:
        run_dir = create_run_dir(runs_dir, exp_id)
    except FileExistsError:
        print(f"RUN ERROR: run directory for '{exp_id}' already exists (same config started twice in "
              "the same second). Runs are never overwritten; wait a second and retry.", file=sys.stderr)
        return 1
    setup_logging(run_dir, cfg.output.log_level)
    log = get_logger("run")
    TransformationLog(run_dir)  # creates the handle; entries appear once stages transform data

    seeded = seed_everything(cfg.experiment.seed)
    manifest = write_run_metadata(cfg, run_dir, exp_id, REPO_ROOT, seeded)

    log.info("experiment_id=%s framework_phase=%d input=%s", exp_id, FRAMEWORK_PHASE, cfg.data.input_configuration)
    log.info("config_hash=%s comparability_hash=%s", manifest["config_hash"][:12], manifest["comparability_hash"][:12])
    log.info("seeded: %s", seeded)
    for path, phase, why in unresolved_fields(cfg):
        log.info("UNRESOLVED (needed in Phase %d): %s - %s. Cannot determine until the dataset is inspected.", phase, path, why)

    print("\n" + format_status_table() + "\n")
    stopped = run_pipeline(cfg, run_dir)
    print(f"Run directory: {run_dir}")
    if stopped is not None:
        print(f"Stopped at stage '{stopped.name}' (not completed, Phase {stopped.phase}). Check the run log for the reason.")
        return 0 if args.scaffold_check else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
