"""Evaluation primitives: fixed-class metrics, bootstrap CIs, gates."""
from rsi.eval.metrics import fixed_confusion, macro_f1_fixed, support_report

__all__ = ["fixed_confusion", "macro_f1_fixed", "support_report"]
