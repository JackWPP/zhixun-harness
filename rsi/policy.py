"""Declarative data-use policy for the competition.

Competition rule: validation/hidden labels must never be fitted.  The harness
also supports bundling an external open dataset for *pretraining* (A: UCI gas
batches 3-7, which are not part of the official package).  Because that is an
extra-data choice rather than a rule requirement, it is strictly opt-in:
``RSI_ALLOW_EXTENDED_DATA=1``.  When disabled (the default, and the mode used
for scored runs) proposals that request ``data="extended"`` are rejected.
"""
from __future__ import annotations

import os

EXTENDED_FLAG = "RSI_ALLOW_EXTENDED_DATA"
TRAIN_ONLY = "train"
EXTENDED = "extended"
DENIED_DATA = ("train+calibration", "train+all-public", "validation", "hidden")


def extended_enabled() -> bool:
    return os.environ.get(EXTENDED_FLAG) == "1"


def allowed_training_data(task) -> set[str]:
    allowed = {TRAIN_ONLY}
    if extended_enabled() and getattr(task, "extended", None):
        allowed.add(EXTENDED)
    return allowed


def deny_reason(data: str, task) -> str | None:
    if data in allowed_training_data(task):
        return None
    if data == EXTENDED and getattr(task, "extended", None) and not extended_enabled():
        return (f"data='extended' requires {EXTENDED_FLAG}=1 (external pretraining is "
                "opt-in and off by default for compliance)")
    return (f"data={data!r} is not allowed: validation/calibration splits must never "
            "be fitted; only 'train' (and opt-in 'extended') may train")
