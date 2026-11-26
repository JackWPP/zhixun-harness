import json

import numpy as np
import pytest

from rsi.model import ExtendedModel
from rsi.platform.c_safety import conservative_candidates, group_guard, interpolate_checkpoint


def checkpoints():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(24, 2))
    y = 30 + 4 * x[:, 0] + x[:, 1]
    parent = ExtendedModel("regression", ['[["key","x"]]', '[["key","z"]]'], [], x, y,
                           hidden=3, seed=17, batch_size=8)
    parent.fit(x, y, 2)
    trained = parent.clone()
    trained.fit(x, y, 2)
    return parent, trained, x, y


def test_interpolation_preserves_lineage_and_restarts_adam_without_mutation():
    parent, trained, x, _ = checkpoints()
    old_parent, old_trained = parent.predict(x), trained.predict(x)
    for alpha, model in conservative_candidates(parent, trained):
        for key in model.params:
            np.testing.assert_array_equal(model.params[key], parent.params[key] + alpha * (trained.params[key] - parent.params[key]))
            assert not model.first[key].any()
            assert not model.second[key].any()
        assert model.meta["steps"] == 0
        assert model.meta["epochs"] == trained.meta["epochs"] == 4
        assert model.meta["c_conservative"]["trained_steps"] == 12
    np.testing.assert_array_equal(parent.predict(x), old_parent)
    np.testing.assert_array_equal(trained.predict(x), old_trained)


def test_saved_interpolated_checkpoint_resumes_exactly(tmp_path):
    parent, trained, x, y = checkpoints()
    continuous = interpolate_checkpoint(parent, trained, .25)
    split = continuous.clone()
    continuous.fit(x, y, 2)
    split.fit(x, y, 1)
    destination = tmp_path / "checkpoint.npz"
    split.save(destination)
    split = ExtendedModel.load(destination)
    split.fit(x, y, 1)
    assert split.meta["epochs"] == continuous.meta["epochs"] == 6
    assert split.meta["steps"] == continuous.meta["steps"] == 6
    for state in ("params", "first", "second"):
        for key in continuous.params:
            np.testing.assert_array_equal(getattr(split, state)[key], getattr(continuous, state)[key])


@pytest.mark.parametrize("field", ["mean", "scale", "target_mean", "target_scale", "paths"])
def test_interpolation_rejects_incompatible_feature_or_target_space(field):
    parent, trained, _, _ = checkpoints()
    if field in ("mean", "scale"):
        getattr(trained, field)[0] += 0.1
    elif field == "paths":
        trained.meta[field][0] = '[["key","different"]]'
    else:
        trained.meta[field] += .1
    with pytest.raises(ValueError, match="incompatible"):
        interpolate_checkpoint(parent, trained, .25)


def ids():
    return [f"{group}_train_u{unit:04d}_c{cycle:04d}"
            for group in ("fd001", "fd002") for unit in (1, 2) for cycle in range(4)]


def test_group_gate_rejects_regression_hidden_by_global_improvement():
    y = np.full(16, 100.)
    parent = y + np.r_[np.ones(8), np.full(8, 20.)]
    candidate = y + np.r_[np.full(8, 2.), np.full(8, 10.)]
    assert np.mean((candidate-y)**2) < np.mean((parent-y)**2)
    result = group_guard(parent, candidate, y, ids())
    assert not result["passed"]
    assert result["status"] == "group_regression"
    assert result["groups"]["fd001"]["failed_metrics"] == ["rmse", "mae", "nasa_score"]


def test_group_gate_accepts_equal_or_improving_supported_groups():
    y = np.full(16, 100.)
    for candidate in (y+2, y+1):
        assert group_guard(y+2, candidate, y, ids())["passed"]


def test_group_gate_reports_unknown_or_sparse_groups_instead_of_claiming_safety():
    y = np.zeros(16)
    unknown = ids()
    unknown[0] = "private-renamed-id"
    result = group_guard(y+2, y+1, y, unknown)
    assert result["passed"] and not result["coverage"]["fd_complete"]
    assert result["unknown_rows"] == 1
    result = group_guard(y[:4]+2, y[:4]+1, y[:4], ids()[:4])
    assert not result["passed"] and result["insufficient"] == ["fd001"]


def test_unknown_identifiers_use_rul_fallback_without_claiming_fd_protection():
    y = np.full(16, 200.)
    result = group_guard(y+2, y+1, y, [str(i) for i in range(16)])
    assert result["passed"] and result["protection"] == "rul_fallback"
    assert not result["coverage"]["fd_complete"]
    assert result["coverage"]["fd_supported_groups"] == []
    assert result["coverage"]["rul_supported_groups"] == ["RUL_100_249"]
    assert not group_guard(y+1, y+2, y, [str(i) for i in range(16)])["passed"]


def test_rul_tail_audit_is_explicit_and_opt_in_with_supported_fd_groups():
    y = np.tile(np.r_[np.full(4, 20.), np.full(4, 300.)], 2)
    parent = y + np.tile(np.r_[np.full(4, 20.), np.full(4, 1.)], 2)
    candidate = y + np.tile(np.r_[np.full(4, 10.), np.full(4, 2.)], 2)
    audit = group_guard(parent, candidate, y, ids())
    assert audit["passed"] and audit["rul_guard_passed"] is False
    assert audit["rul_groups"]["RUL_250_plus"]["status"] == "regression"
    assert not group_guard(parent, candidate, y, ids(), enforce_rul=True)["passed"]


def test_group_gate_handles_overflow_and_invalid_prediction_shape():
    y = np.zeros(16)
    result = group_guard(y+2, y+1e5, y, ids())
    assert not result["passed"]
    assert "nasa_score" in result["groups"]["fd001"]["failed_metrics"]
    json.dumps(result, allow_nan=False)
    assert group_guard(y, y[:2], y, ids())["status"] == "invalid_evidence"
    extreme = group_guard(y, y+1e308, y, ids())
    assert not extreme["passed"]
    assert extreme["groups"]["fd001"]["candidate"]["rmse"] is None
    json.dumps(extreme, allow_nan=False)


@pytest.mark.parametrize("alpha", [0., 1., .3, True, float("nan")])
def test_interpolation_refuses_undeclared_search_values(alpha):
    parent, trained, _, _ = checkpoints()
    with pytest.raises(ValueError, match="predeclared"):
        interpolate_checkpoint(parent, trained, alpha)
