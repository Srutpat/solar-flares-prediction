"""Framework-specific exceptions."""


class ConfigError(ValueError):
    """The experiment configuration is malformed or self-inconsistent."""


class NotImplementedPhase(NotImplementedError):
    """A placeholder component that is scheduled for a later development phase."""

    def __init__(self, phase: int, what: str):
        self.phase = phase
        self.what = what
        super().__init__(f"Not implemented yet: {what} (scheduled for Phase {phase})")


class DataUndetermined(RuntimeError):
    """Required information cannot be determined from the supplied data.

    Raised instead of guessing. Message convention:
    "Cannot determine until the dataset is inspected: <what>".
    """
