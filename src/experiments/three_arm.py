"""Phase 9 placeholder - Run SoLEXS-only / HEL1OS-only / combined and tabulate."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import ComparisonRunner


class ThreeArmComparison(ComparisonRunner):
    """Run SoLEXS-only / HEL1OS-only / combined and tabulate."""

    PHASE = 9
    IMPLEMENTED = False

    def run(self, *args, **kwargs):
        raise NotImplementedPhase(9, "ThreeArmComparison.run")
