"""Phase 11 placeholder - Feature-group and instrument ablations."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import AblationRunner


class FeatureAblationRunner(AblationRunner):
    """Feature-group and instrument ablations."""

    PHASE = 11
    IMPLEMENTED = False

    def run(self, *args, **kwargs):
        raise NotImplementedPhase(11, "FeatureAblationRunner.run")
