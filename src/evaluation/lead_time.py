"""Phase 10 placeholder - Event-level warning lead time from first qualifying trigger."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import LeadTimeAnalyzer


class EventLeadTimeAnalyzer(LeadTimeAnalyzer):
    """Event-level warning lead time from first qualifying trigger."""

    PHASE = 10
    IMPLEMENTED = False

    def analyze(self, *args, **kwargs):
        raise NotImplementedPhase(10, "EventLeadTimeAnalyzer.analyze")
