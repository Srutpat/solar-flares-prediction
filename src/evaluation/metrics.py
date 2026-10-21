"""Phase 10 placeholder - TSS, HSS, precision, recall, AUC, confusion matrix."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import Evaluator


class ForecastEvaluator(Evaluator):
    """TSS, HSS, precision, recall, AUC, confusion matrix."""

    PHASE = 10
    IMPLEMENTED = False

    def evaluate(self, *args, **kwargs):
        raise NotImplementedPhase(10, "ForecastEvaluator.evaluate")
