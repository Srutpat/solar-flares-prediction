import pytest

from src.core.errors import NotImplementedPhase
from src.experiments.pipeline import STAGES, run_pipeline, stage_status


def test_phases_2_to_5_catalogue_and_labels_are_implemented():
    status = dict((st.name, ok) for st, ok in stage_status())
    assert status["data_adapter"] is True
    assert status["validation"] is True
    assert status["preprocessing"] is True
    assert status["synchronization"] is True
    assert status["event_catalogue"] is True
    assert status["labels"] is True
    assert status["causal_features"] is True
    assert all(not ok for name, ok in status.items() if name not in {"data_adapter", "validation", "preprocessing", "synchronization", "event_catalogue", "labels", "causal_features"})


def test_run_stops_at_data_adapter_without_a_configuration():
    assert run_pipeline(cfg=None, run_dir=None).name == "data_adapter"


def test_placeholders_raise_with_their_phase():
    import importlib
    for st in STAGES:
        cls = getattr(importlib.import_module(st.module), st.cls)
        assert cls.PHASE == st.phase
        if st.name in {"data_adapter", "validation", "preprocessing", "synchronization", "event_catalogue", "labels", "causal_features"}:
            continue
        obj = cls()
        abstract_names = [n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n))]
        method = next(n for n in abstract_names if n not in ("mark_undetermined",))
        with pytest.raises(NotImplementedPhase) as exc:
            getattr(obj, method)()
        assert exc.value.phase == st.phase


def test_schema_report_never_guesses():
    from src.core.interfaces import SchemaReport
    r = SchemaReport()
    assert r.n_observations is None and r.cadence is None and r.undetermined == []
    r.mark_undetermined("timestamp_column")
    r.mark_undetermined("timestamp_column")
    assert r.undetermined == ["timestamp_column"]
