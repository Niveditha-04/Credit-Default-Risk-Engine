"""Thin re-export: the actual labels live in src/explain/labels.py, the
single source of truth shared with the Phase 6 SHAP report. That module
has no heavy imports (no matplotlib, unlike shap_analysis.py itself), so
importing it here does not add anything to the serving dependency list.
"""

from src.explain.labels import PLAIN_ENGLISH, describe

__all__ = ["PLAIN_ENGLISH", "describe"]
