"""backend: thin internal ops API + minimal admin UI.

Reads the artifacts produced by training/eval (``.report.json``,
``.comparison.json``, ``.keras``) from the models directory and renders a
minimal admin console: model list, train report (period, per-appliance labeled
hours), and HA-vs-NILM accuracy vs the acceptance gate.

Intentionally minimal (no React build yet, per CLAUDE.md "trimmed, later"):
FastAPI + server-rendered HTML. The model registry logic is pure and tested.
"""
