"""Evaluation primitives: fixed-class metrics, bootstrap CIs, evidence gates."""
from rsi.eval.gate import GateDecision, evaluate_evidence, select_arm
from rsi.eval.metrics import fixed_confusion, macro_f1_fixed, support_report
from rsi.eval.stats import Interval, block_bootstrap_ci, holm_adjust, paired_block_bootstrap

__all__ = [
    "GateDecision", "evaluate_evidence", "select_arm",
    "fixed_confusion", "macro_f1_fixed", "support_report",
    "Interval", "block_bootstrap_ci", "holm_adjust", "paired_block_bootstrap",
]
