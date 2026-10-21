from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.errors import DataUndetermined
from src.feature_engineering.causal_features import CausalFeatureBuilder


class FakeStream:
    def __init__(self, stream_id, instrument, timestamps, values, columns=None, metadata=None):
        self.stream_id = stream_id
        self.instrument = instrument
        self.detector = "test"
        self.source_file = "test.fits"
        self.measurement_columns = columns or ["CTR"]
        self.dataframe = pd.DataFrame({"timestamp": timestamps, **values})
        self.metadata = metadata or {}


def _cfg(history=3, windows=None, groups=None):
    return SimpleNamespace(features=SimpleNamespace(
        history_window_minutes=history,
        rolling_window_minutes=windows or [1],
        groups=groups or ["levels", "rates", "rolling_stats"],
    ))


def test_phase6_builds_causal_features_and_excludes_error_columns():
    timestamps = pd.date_range("2026-10-17T00:00:00Z", periods=6, freq="s")
    stream = FakeStream(
        "SoLEXS__SDD2__LC__sample", "SoLEXS", timestamps,
        {"COUNTS": [1., 2., 4., 8., 16., 32.], "STAT_ERR": [1.] * 6},
        columns=["COUNTS", "STAT_ERR"],
        metadata={"phase4_role": "reference_timeline"},
    )
    features, report = CausalFeatureBuilder(_cfg()).build({stream.stream_id: stream})
    assert len(features) == 6
    assert "level__solexs_sdd2_lc_sample__counts" in features
    assert any(c.startswith("rate_per_second__") for c in features)
    assert any(c.startswith("rolling_mean_1_0m__") for c in features)
    assert not any("stat_err" in c.lower() for c in features.columns)
    assert report["feature_count"] > 0
    assert report["source_streams"][0]["measurement_columns_excluded"] == ["STAT_ERR"]


def test_phase6_uses_exact_timestamp_joins_without_forward_filling():
    timestamps = pd.date_range("2026-10-17T00:00:00Z", periods=4, freq="s")
    reference = FakeStream("SoLEXS_ref", "SoLEXS", timestamps,
                           {"COUNTS": [10., 20., 30., 40.]},
                           metadata={"phase4_role": "reference_timeline"})
    other = FakeStream("HEL1OS_band", "HEL1OS", timestamps[[0, 2, 3]],
                       {"CTR": [1., 3., 4.]})
    features, _ = CausalFeatureBuilder(_cfg()).build({reference.stream_id: reference, other.stream_id: other})
    other_level = next(c for c in features.columns if c.startswith("level__hel1os_band__ctr"))
    assert features[other_level].iloc[0] == 1.
    assert np.isnan(features[other_level].iloc[1])
    assert features[other_level].iloc[2] == 3.


def test_phase6_rejects_unresolved_history_window():
    timestamps = pd.date_range("2026-10-17T00:00:00Z", periods=3, freq="s")
    stream = FakeStream("s", "SoLEXS", timestamps, {"CTR": [1., 2., 3.]})
    with pytest.raises(DataUndetermined, match="history_window_minutes"):
        CausalFeatureBuilder(_cfg(history=None)).build({"s": stream})


def test_phase6_generates_causal_cross_instrument_rolling_correlation():
    timestamps = pd.date_range("2026-10-17T00:00:00Z", periods=4, freq="s")
    solexs = FakeStream("s", "SoLEXS", timestamps, {"CTR": [1., 2., 3., 4.]})
    helios = FakeStream("h", "HEL1OS", timestamps, {"CTR": [2., 4., 6., 8.]})
    features, report = CausalFeatureBuilder(_cfg(groups=["levels", "cross_instrument"])).build(
        {solexs.stream_id: solexs, helios.stream_id: helios}
    )
    corr_cols = [c for c in features.columns if c.startswith("rolling_corr_")]
    assert corr_cols
    assert "rolling Pearson correlation" in report["cross_instrument_features"]
    assert report["feature_count"] == len([c for c in features.columns if c not in {"timestamp", "segment_id"}])
