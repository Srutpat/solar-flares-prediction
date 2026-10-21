"""Phase 11 placeholder - Feature importance and SHAP (if practical); descriptive wording only."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import Explainer


class XGBoostExplainer(Explainer):
    """Feature importance and SHAP (if practical); descriptive wording only."""

    PHASE = 11
    IMPLEMENTED = False

    def explain(self, *args, **kwargs):
        raise NotImplementedPhase(11, "XGBoostExplainer.explain")
