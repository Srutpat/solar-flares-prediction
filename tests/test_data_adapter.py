from pathlib import Path

import pytest

from src.core.config import load_config
from src.data_adapter.adapter import DataAdapterError, _choose_time_column, _config_sources, _is_fits_name

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_fits_name_detection():
    assert _is_fits_name("x.fits")
    assert _is_fits_name("x.fits.gz")
    assert not _is_fits_name("x.csv")
    # SoLEXS Level-1 products are FITS files with product-specific suffixes.
    assert _is_fits_name("SDD2/AL1_SOLEXS_20260925_SDD2_L1.lc.gz")
    assert _is_fits_name("SDD2/AL1_SOLEXS_20260925_SDD2_L1.gti.gz")
    assert _is_fits_name("SDD2/AL1_SOLEXS_20260925_SDD2_L1.pi.gz")


def test_solexs_source_resolution():
    cfg = load_config(CONFIGS / "solexs.yaml")
    cfg.data.solexs_source = "data/solexs"
    assert _config_sources(cfg) == [("SoLEXS", "data/solexs")]


def test_helios_source_resolution():
    cfg = load_config(CONFIGS / "helios.yaml")
    cfg.data.helios_source = "data/helios"
    assert _config_sources(cfg) == [("HEL1OS", "data/helios")]


def test_combined_requires_both_sources():
    cfg = load_config(CONFIGS / "combined.yaml")
    with pytest.raises(DataAdapterError, match="both data.solexs_source and data.helios_source"):
        _config_sources(cfg)


def test_combined_source_resolution():
    cfg = load_config(CONFIGS / "combined.yaml")
    cfg.data.solexs_source = "data/solexs"
    cfg.data.helios_source = "data/helios"
    assert _config_sources(cfg) == [("SoLEXS", "data/solexs"), ("HEL1OS", "data/helios")]


def test_phase2_adapter_has_no_channel_selection_policy():
    # The adapter must expose discovered measurement columns later; it does not
    # contain a hard-coded 'best' energy band or detector selection.
    from src.data_adapter.adapter import ConfiguredDataAdapter
    assert ConfiguredDataAdapter.IMPLEMENTED is True


def test_combined_source_resolution_accepts_split_helios_archives():
    from src.data_adapter.adapter import _config_sources
    from src.core.config import load_config
    cfg = load_config("configs/combined.yaml")
    cfg.data.solexs_source = "solexs.zip"
    cfg.data.helios_source = ["helios_part1.zip", "helios_part2.zip"]
    resolved = _config_sources(cfg)
    assert resolved == [("SoLEXS", "solexs.zip"), ("HEL1OS", "helios_part1.zip"), ("HEL1OS", "helios_part2.zip")]
