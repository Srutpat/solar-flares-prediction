"""Phase 7 placeholder - Chronological train/validation/test split with event separation and purge gap."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import ChronologicalSplitter


class EventAwareChronologicalSplitter(ChronologicalSplitter):
    """Chronological train/validation/test split with event separation and purge gap."""

    PHASE = 7
    IMPLEMENTED = False

    def split(self, *args, **kwargs):
        raise NotImplementedPhase(7, "EventAwareChronologicalSplitter.split")
