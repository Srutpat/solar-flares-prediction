"""Phase 12 placeholder - Experimental sequence model, added only after XGBoost works."""
from src.core.errors import NotImplementedPhase
from src.core.interfaces import ForecastModel


class LSTMForecaster(ForecastModel):
    """Experimental sequence model, added only after XGBoost works."""

    PHASE = 12
    IMPLEMENTED = False

    def fit(self, *args, **kwargs):
        raise NotImplementedPhase(12, "LSTMForecaster.fit")

    def predict_proba(self, *args, **kwargs):
        raise NotImplementedPhase(12, "LSTMForecaster.predict_proba")

    def save(self, *args, **kwargs):
        raise NotImplementedPhase(12, "LSTMForecaster.save")

    def load(self, *args, **kwargs):
        raise NotImplementedPhase(12, "LSTMForecaster.load")
