"""Phase 8 placeholder - Primary structured baseline on engineered causal features."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import ForecastModel


class XGBoostForecaster(ForecastModel):
    """Primary structured baseline on engineered causal features."""

    PHASE = 8
    IMPLEMENTED = False

    def fit(self, *args, **kwargs):
        raise NotImplementedPhase(8, "XGBoostForecaster.fit")

    def predict_proba(self, *args, **kwargs):
        raise NotImplementedPhase(8, "XGBoostForecaster.predict_proba")

    def save(self, *args, **kwargs):
        raise NotImplementedPhase(8, "XGBoostForecaster.save")

    def load(self, *args, **kwargs):
        raise NotImplementedPhase(8, "XGBoostForecaster.load")
