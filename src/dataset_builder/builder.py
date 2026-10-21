"""Phase 7 placeholder - Join features and labels into the forecast dataset."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import DatasetBuilder


class ForecastDatasetBuilder(DatasetBuilder):
    """Join features and labels into the forecast dataset."""

    PHASE = 7
    IMPLEMENTED = False

    def build(self, *args, **kwargs):
        raise NotImplementedPhase(7, "ForecastDatasetBuilder.build")
