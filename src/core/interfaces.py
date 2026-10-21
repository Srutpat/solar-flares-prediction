"""Stage contracts (abstract interfaces) for every pipeline component.

Each stage depends only on these contracts, never on raw file structure. The
concrete payload types below are deliberately `Any`: their structure is defined
in the phase named in the comment, and only from VERIFIED fields of the real
dataset. Nothing here assumes columns, cadence, timestamp format or labels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Payload types - structure finalised in the named phase, once real fields are known.
RawData = Any                 # Phase 2  (whatever the file format yields)
StandardizedTimeSeries = Any  # Phase 2  (only verified fields; timestamps preserved)
QualityReport = Any           # Phase 3
EventCatalogue = Any          # Phase 5
LabelSeries = Any             # Phase 5
FeatureTable = Any            # Phase 6
FeatureProvenance = Any       # Phase 6
ForecastDataset = Any         # Phase 7
SplitResult = Any             # Phase 7


@dataclass
class SchemaReport:
    """What the adapter found in the data. `None` / listed in `undetermined` means
    "Cannot determine until the dataset is inspected" - never a silent guess."""
    file_format: Optional[str] = None
    columns: Optional[list] = None
    dtypes: Optional[dict] = None
    timestamp_available: Optional[bool] = None
    timestamp_column: Optional[str] = None
    cadence: Optional[dict] = None              # detected, incl. irregularity
    missing_values: Optional[dict] = None
    duplicate_timestamps: Optional[int] = None
    numeric_ranges: Optional[dict] = None
    n_observations: Optional[int] = None
    temporal_coverage: Optional[dict] = None
    channel_identifiers: Optional[list] = None
    products: Optional[dict] = None
    files: Optional[list] = None
    extensions: Optional[list] = None
    measurement_candidates: Optional[list] = None
    measurement_columns: Optional[list] = None
    time_axes: Optional[list] = None
    warnings: list = field(default_factory=list)
    undetermined: list = field(default_factory=list)

    def mark_undetermined(self, item: str) -> None:
        if item not in self.undetermined:
            self.undetermined.append(item)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- data layer
class DataAdapter(ABC):
    """Phase 2. Isolates the framework from the raw file structure."""

    @abstractmethod
    def load_data(self, source, **options) -> RawData: ...

    @abstractmethod
    def validate_schema(self, raw: RawData) -> SchemaReport: ...

    @abstractmethod
    def identify_time_axis(self, raw: RawData, report: SchemaReport): ...

    @abstractmethod
    def identify_measurement_columns(self, raw: RawData, report: SchemaReport): ...

    @abstractmethod
    def return_standardized_timeseries(self, raw: RawData, report: SchemaReport) -> StandardizedTimeSeries: ...


class QualityValidator(ABC):
    """Phase 3. Inspect only; reports gaps, duplicates, invalid values, cadence."""

    @abstractmethod
    def check(self, ts: StandardizedTimeSeries, log) -> QualityReport: ...


class Preprocessor(ABC):
    """Phase 3. Every transformation is written to the TransformationLog."""

    @abstractmethod
    def apply(self, ts: StandardizedTimeSeries, log) -> StandardizedTimeSeries: ...


class Synchronizer(ABC):
    """Phase 4. Aligns instruments WITHOUT using observations later than the
    prediction timestamp; the method is documented, never chosen for score."""

    @abstractmethod
    def synchronize(self, streams: dict, log) -> StandardizedTimeSeries: ...


# ---------------------------------------------------------------- events/labels
class EventCatalogueBuilder(ABC):
    """Phase 5. Event detection is NOT the claimed novelty; prefer association
    with a reference catalogue over inventing flare classes from instrument signal."""

    @abstractmethod
    def build(self, ts: StandardizedTimeSeries, log) -> EventCatalogue: ...


class LabelBuilder(ABC):
    """Phase 5. y(t, dt) = 1 iff a validated flare onset occurs in (t, t + dt]."""

    @abstractmethod
    def build_labels(self, timeline: Any, catalogue: EventCatalogue, horizons_minutes: list) -> LabelSeries: ...


# ---------------------------------------------------------------- features/datasets
class FeatureBuilder(ABC):
    """Phase 6. Uses ONLY information at or before t; builds only features the
    verified data can support; emits a provenance report."""

    @abstractmethod
    def build(self, ts: StandardizedTimeSeries, log) -> tuple: ...   # (FeatureTable, FeatureProvenance)


class DatasetBuilder(ABC):
    """Phase 7."""

    @abstractmethod
    def build(self, features: FeatureTable, labels: LabelSeries) -> ForecastDataset: ...


class ChronologicalSplitter(ABC):
    """Phase 7. TRAIN -> VALIDATION -> TEST, never random; event separation and a
    purge gap prevent overlapping windows from straddling splits."""

    @abstractmethod
    def split(self, dataset: ForecastDataset) -> SplitResult: ...


# ---------------------------------------------------------------- models
class ForecastModel(ABC):
    """Phases 8 (XGBoost) and 12 (LSTM). Same target, horizons, inputs, leakage
    controls for both. Hyper-parameters are tuned on validation only."""

    @abstractmethod
    def fit(self, train: Any, validation: Any) -> None: ...

    @abstractmethod
    def predict_proba(self, features: Any): ...

    @abstractmethod
    def save(self, path) -> None: ...

    @abstractmethod
    def load(self, path) -> None: ...


# ---------------------------------------------------------------- evaluation
class ThresholdSelector(ABC):
    """Phase 10. Chosen on validation data by a documented criterion, then FIXED
    and stored before the test set is touched."""

    @abstractmethod
    def select(self, y_val: Any, p_val: Any) -> float: ...


class Evaluator(ABC):
    """Phase 10. TSS, HSS, precision, recall, AUC, confusion matrix; accuracy
    only as supplementary."""

    @abstractmethod
    def evaluate(self, y_true: Any, p: Any, threshold: float) -> dict: ...


class LeadTimeAnalyzer(ABC):
    """Phase 10. Lead = flare onset - first qualifying trigger. A trigger at or
    after onset is NOT early warning."""

    @abstractmethod
    def analyze(self, events: Any, triggers: Any) -> dict: ...


class ComparisonRunner(ABC):
    """Phase 9. Runs the three arms and builds the comparison table."""

    @abstractmethod
    def run(self, config_paths: list) -> Any: ...


class AblationRunner(ABC):
    """Phase 11. Feature-group and instrument ablations."""

    @abstractmethod
    def run(self, cfg: Any) -> Any: ...


class Explainer(ABC):
    """Phase 11. Descriptive only: 'contributed to the prediction', never 'caused'."""

    @abstractmethod
    def explain(self, model: Any, features: Any) -> Any: ...


class Visualizer(ABC):
    """Phase 13 (figures are added progressively from Phase 2)."""

    @abstractmethod
    def render(self, results: Any, run_dir) -> list: ...


class ReportWriter(ABC):
    """Phase 13. Separates FACT / METHOD / RESULT / INTERPRETATION / LIMITATION."""

    @abstractmethod
    def write(self, results: Any, run_dir) -> Any: ...
