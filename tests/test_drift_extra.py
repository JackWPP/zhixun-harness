"""Incremental drift-report tests: enrichment fields, compatibility, guard schema.

The v0.5.1 ``DriftReport`` contract (13 positional fields, cause rules, severity
formula/thresholds) must stay byte-compatible for existing consumers; the new
fields are appended with defaults.  These tests pin both halves of that promise
and check that every recommended action is already admissible by the executor.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsi.drift import DriftReport, analyse
from rsi.executor import validate_params, validate_postprocess
from rsi.strategies import STRATEGIES

DRIFT_TYPES = {"stable", "covariate_shift", "sensor_gain_drift",
               "label_prior_shift", "concept_drift", "mixed_drift"}


class _Classifier:
    """Deterministic 3-class probability model over the first two features."""

    meta = {"task": "classification"}

    def predict(self, x):
        index = ((x[:, 0] > 0).astype(int) + (x[:, 1] > 0).astype(int)) % 3
        probabilities = np.full((len(x), 3), 0.05)
        probabilities[np.arange(len(x)), index] = 0.9
        return probabilities


class _Regressor:
    """Model whose prediction is a fixed affine function of feature 0."""

    meta = {"task": "regression"}

    def predict(self, x):
        return 50.0 + 5.0 * x[:, 0]


def _classification_case(seed=0, rows=400, gain=1.8, offset=1.2):
    rng = np.random.default_rng(seed)
    reference_x = rng.normal(size=(rows, 2))
    target_x = reference_x * gain + offset
    reference = SimpleNamespace(x=reference_x,
                                y=(reference_x[:, 0] > 0).astype(int) * 2)
    target = SimpleNamespace(x=target_x, y=(target_x[:, 0] > 0).astype(int) * 2)
    return _Classifier(), reference, target


def _regression_case(seed=1, rows=300, gain=2.5, offset=1.0):
    rng = np.random.default_rng(seed)
    reference_x = rng.normal(size=(rows, 2))
    reference_y = 50.0 + 5.0 * reference_x[:, 0] + rng.normal(scale=0.5, size=rows)
    target_x = reference_x * gain + offset
    # label drift only (same conditional target law); feature gain drives the shift
    target_y = 50.0 + 5.0 * reference_x[:, 0] + rng.normal(scale=0.5, size=rows)
    reference = SimpleNamespace(x=reference_x, y=reference_y)
    target = SimpleNamespace(x=target_x, y=target_y)
    return _Regressor(), reference, target


def test_classification_enrichment_is_populated():
    report = analyse(*_classification_case(), ("Normal", "Leak", "SevereFault"))
    assert report.drift_type in DRIFT_TYPES
    assert report.drift_type == "sensor_gain_drift"
    assert "covariate_shift" in report.causes
    assert "sensor_or_environment_gain_drift" in report.causes
    assert report.score > 0.3
    assert report.score == pytest.approx(report.evidence["severity_score"])
    assert math.isfinite(report.score)

    # attribution ranked descending, plain python, bounded
    assert 1 <= len(report.attribution) <= 10
    scores = [row["score"] for row in report.attribution]
    assert scores == sorted(scores, reverse=True)
    top = report.attribution[0]
    assert set(top) >= {"feature", "score", "mean_shift_sd", "ks", "js",
                        "std_ratio", "direction"}
    assert isinstance(top["feature"], int)
    assert top["direction"] in {"up", "down", "flat"}

    evidence_attribution = report.evidence["attribution_top"]
    assert 1 <= len(evidence_attribution) <= 5
    assert evidence_attribution[0]["feature"] == top["feature"]

    assert report.knowledge_hints == report.evidence["knowledge_hints"]
    assert report.knowledge_hints
    assert report.next_checks
    assert any("selection" in check for check in report.next_checks)

    # first recommended action is the calibration play for gain drift
    first = report.recommended_actions[0]
    assert first["strategy"] == "retain"
    assert first["postprocess"]["type"] == "recenter"
    assert first["postprocess"]["mode"] in {"batch_mean", "batch_std"}


def test_stable_split_reports_mild_and_retain():
    rng = np.random.default_rng(7)
    reference_x = rng.normal(size=(300, 2))
    reference_y = (reference_x[:, 0] > 0).astype(int)
    reference = SimpleNamespace(x=reference_x, y=reference_y)
    target = SimpleNamespace(x=reference_x.copy(), y=reference_y.copy())
    report = analyse(_Classifier(), reference, target, ("a", "b", "c"))
    assert report.drift_type == "stable"
    assert report.causes == ["stable"]
    assert report.severity == "mild"
    assert report.score < 0.32
    assert report.recommended_actions[0]["strategy"] == "retain"
    assert report.recommended_actions[0]["postprocess"] == {"type": "none"}
    assert "taxonomy" in report.knowledge_hints


def test_regression_shift_offers_shrink_and_rul_hints():
    regressor, reference, target = _regression_case()
    report = analyse(regressor, reference, target, ())
    assert report.drift_type in DRIFT_TYPES
    assert report.drift_type in {"sensor_gain_drift", "concept_drift", "mixed_drift"}
    postprocess_types = {action["postprocess"]["type"] for action in report.recommended_actions}
    assert "none" in postprocess_types
    assert "recenter" in postprocess_types or "shrink" in postprocess_types
    assert {"rul", "regression"} & set(report.knowledge_hints)
    assert report.evidence["rows_reference"] == len(reference.x)
    assert report.evidence["rows_target"] == len(target.x)


def test_recommended_actions_are_guard_admissible():
    for task, case in (("classification", _classification_case()),
                       ("regression", _regression_case())):
        report = analyse(case[0], case[1], case[2], ())
        assert report.recommended_actions, f"{task}: no action proposed"
        assert len(report.recommended_actions) <= 4
        for action in report.recommended_actions:
            assert action["strategy"] in STRATEGIES
            assert validate_params(action["strategy"], action["params"]) == action["params"]
            validated = validate_postprocess(action["postprocess"], task)
            assert validated["type"] == action["postprocess"]["type"]
            assert action["risk"] in {"none", "low", "medium", "high"}
            assert action["rationale"]
            assert action["expected_effect"]


def test_severity_thresholds_match_legacy_formula():
    cases = [_classification_case(), _classification_case(seed=3, gain=1.05, offset=0.02),
             _regression_case()]
    labels = set()
    for model, reference, target in cases:
        report = analyse(model, reference, target, ())
        expected = ("severe" if report.score >= 0.6
                    else "moderate" if report.score >= 0.32 else "mild")
        assert report.severity == expected
        labels.add(report.severity)
        # the evidence block must carry a compact copy for whitelist consumers
        assert report.evidence["drift_type"] == report.drift_type
        assert report.evidence["causes"] == list(report.causes)
    assert len(labels) >= 2


def test_legacy_positional_construction_and_to_dict():
    report = DriftReport(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, None, None, None, None,
                         "mild", ["stable"], {})
    assert report.score == 0.0
    assert report.drift_type == "stable"
    assert report.attribution == []
    assert report.recommended_actions == []
    payload = report.to_dict()
    assert len(payload) == len(DriftReport.__dataclass_fields__) + 1  # + ``type`` alias
    assert payload["type"] == "stable"
    assert payload["score"] == 0.0
    json.dumps(payload)

    explicit = DriftReport(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, None, None, None, None,
                           "mild", ["stable"], {}, score=0.42,
                           drift_type="covariate_shift")
    assert explicit.to_dict()["type"] == "covariate_shift"
    assert explicit.score == 0.42


def test_tiny_and_constant_inputs_do_not_crash():
    rng = np.random.default_rng(11)
    constant = np.ones((3, 4))
    reference = SimpleNamespace(x=constant, y=np.array([0, 1, 2]))
    target = SimpleNamespace(x=constant + 1e-9, y=np.array([0, 1, 2]))
    report = analyse(_Classifier(), reference, target, ("a", "b", "c"))
    for value in (report.feature_shift, report.max_feature_shift, report.ks_max,
                  report.ks_p95, report.score):
        assert math.isfinite(value)
    assert report.drift_type in DRIFT_TYPES
    json.dumps(report.to_dict())

    # single-row regression split: all statistics must stay finite
    regressor = _Regressor()
    ref = SimpleNamespace(x=rng.normal(size=(1, 2)), y=np.array([50.0]))
    tgt = SimpleNamespace(x=rng.normal(size=(1, 2)), y=np.array([51.0]))
    single = analyse(regressor, ref, tgt, ())
    assert math.isfinite(single.score)
    assert single.drift_type in DRIFT_TYPES


def test_evidence_has_severity_components_and_top_features():
    model, reference, target = _classification_case(seed=5, gain=2.0, offset=0.5)
    report = analyse(model, reference, target, ("a", "b", "c"))
    components = report.evidence["severity_components"]
    assert set(components) == {"feature_shift", "ks", "performance"}
    assert sum(components.values()) == pytest.approx(report.score, abs=1e-3)
    tops = report.evidence["top_shifted_features"]
    assert len(set(tops)) == len(tops)
    assert set(tops) <= set(range(reference.x.shape[1]))
    assert len(tops) == min(8, reference.x.shape[1])
    assert report.max_feature_shift >= report.feature_shift
