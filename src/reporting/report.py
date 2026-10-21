"""Phase 13 placeholder - Findings report separating fact / method / result / interpretation / limitation."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import ReportWriter


class ResearchReportWriter(ReportWriter):
    """Findings report separating fact / method / result / interpretation / limitation."""

    PHASE = 13
    IMPLEMENTED = False

    def write(self, *args, **kwargs):
        raise NotImplementedPhase(13, "ResearchReportWriter.write")
