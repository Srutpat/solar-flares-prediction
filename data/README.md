# data/

Place the real SoLEXS / HEL1OS dataset here (or point `data.source_path` elsewhere).
Data files are git-ignored. Nothing in this framework assumes the file format,
columns, timestamp format, or cadence - the Phase 2 adapter inspects and reports them.

Synthetic data may exist only inside `tests/` for unit testing and must never be
presented as research results.
