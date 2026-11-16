from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rsi.data import Dataset
from rsi.runner import Runner


def test_prelabel_drift_report_uses_only_ordered_covariates(tmp_path):
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    train = Dataset("train", [str(i) for i in range(12)], x,
                    np.zeros(12), [], "train-sha", [""] * 12)
    selection = Dataset("selection", [str(i) for i in range(12)], x + 50,
                        np.arange(12, dtype=np.float64), [], "selection-sha", [""] * 12)
    runner = Runner.__new__(Runner)
    runner.rounds = 3
    state = SimpleNamespace(train=train, selection=selection)

    first = runner.drift_report(state, 0)
    selection.y = selection.y[::-1].copy()
    replay = runner.drift_report(state, 0)

    assert first == replay
    assert (first["window_start"], first["window_end"], first["window_rows"]) == (0, 4, 4)
    assert first["monitor_type"] == "prequential_covariate_only"
    assert first["performance_before"] is None
    assert first["performance_target"] is None
