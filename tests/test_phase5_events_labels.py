import pandas as pd
import pytest

from src.core.errors import DataUndetermined
from src.event_detection.catalogue import ReferenceEventCatalogueBuilder
from src.event_detection.labels import OnsetLabelBuilder


def test_labels_use_open_left_closed_right_horizon_and_mask_tail():
    timeline = pd.date_range("2026-09-25T00:00:00Z", periods=61, freq="min")
    catalogue = pd.DataFrame({
        "onset_utc": pd.to_datetime(["2026-09-25T00:15:00Z"], utc=True),
        "within_observation_overlap": [True],
    })
    labels = OnsetLabelBuilder().build_labels(timeline, catalogue, [15])
    assert labels.loc[0, "y_15m"] == 1  # onset at t+15 is included
    assert labels.loc[15, "y_15m"] == 0  # onset at t is excluded
    assert labels.loc[46, "y_15m"] is pd.NA or pd.isna(labels.loc[46, "y_15m"])
    assert bool(labels.loc[45, "valid_15m"]) is True
    assert bool(labels.loc[46, "valid_15m"]) is False


def test_automatic_catalogue_requires_observation_timeline_when_no_csv_is_given(tmp_path):
    class Labels:
        reference_catalogue_path = None
    class Cfg:
        labels = Labels()
    with pytest.raises(DataUndetermined, match="No synchronized timeline"):
        ReferenceEventCatalogueBuilder(Cfg()).build({}, None)


def test_catalogue_validates_required_columns(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({"onset_utc": ["2026-09-25T00:10:00Z"]}).to_csv(path, index=False)
    class Labels:
        reference_catalogue_path = str(path)
    class Cfg:
        labels = Labels()
    with pytest.raises(DataUndetermined, match="missing required column"):
        ReferenceEventCatalogueBuilder(Cfg()).build({}, None)


def test_labels_do_not_cross_observation_gaps():
    # A 10-minute data gap splits two valid observation segments. Labels near
    # the end of the first segment must be invalid, not negative/positive.
    first = pd.date_range("2026-10-17T00:00:00Z", periods=5, freq="min")
    second = pd.date_range("2026-10-17T00:15:00Z", periods=20, freq="min")
    timeline = first.append(second)
    catalogue = pd.DataFrame({
        "onset_utc": pd.to_datetime(["2026-10-17T00:16:00Z"], utc=True),
        "within_observation_overlap": [True],
    })
    labels = OnsetLabelBuilder().build_labels(timeline, catalogue, [3])
    assert pd.isna(labels.loc[2, "y_3m"])  # horizon would cross the data gap
    assert labels.loc[5, "y_3m"] == 1
    assert labels.loc[5, "valid_3m"]
    assert labels["segment_id"].nunique() == 2


def test_reference_catalogue_does_not_mark_event_in_gap_as_observed(tmp_path):
    first = pd.date_range("2026-10-17T00:00:00Z", periods=5, freq="min")
    second = pd.date_range("2026-10-17T00:15:00Z", periods=20, freq="min")
    timeline = first.append(second)
    path = tmp_path / "events.csv"
    pd.DataFrame({
        "onset_utc": ["2026-10-17T00:10:00Z", "2026-10-17T00:16:00Z"],
        "source": ["validated catalogue", "validated catalogue"],
    }).to_csv(path, index=False)
    class Labels:
        reference_catalogue_path = str(path)
    class Cfg:
        labels = Labels()
    class Stream:
        def __init__(self, stamps):
            self.dataframe = pd.DataFrame({"timestamp": stamps})
    result = ReferenceEventCatalogueBuilder(Cfg()).build({"reference": Stream(timeline)}, None)
    assert result["within_observation_overlap"].tolist() == [False, True]
