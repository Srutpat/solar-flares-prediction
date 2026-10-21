"""Phase 10 placeholder - Select and freeze the decision threshold on validation data."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import ThresholdSelector


class ValidationThresholdSelector(ThresholdSelector):
    """Select and freeze the decision threshold on validation data."""

    PHASE = 10
    IMPLEMENTED = False

    def select(self, *args, **kwargs):
        raise NotImplementedPhase(10, "ValidationThresholdSelector.select")
