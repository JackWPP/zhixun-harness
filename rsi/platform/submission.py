"""Workspace v2 participant entry (``contest.harness.workspace.v2``).

The platform runs ``/app/harness.py train`` for every round and later
``/app/harness.py predict`` for the selected artifact.  This module is the
participant implementation used by both commands:

``train(rnd)``
    Reads the public training/validation projection, evaluates the parent
    checkpoint, runs one model call through the host gateway, builds at most
    two candidate post-processing/continuation variants on the *official
    train split only*, gates them on validation, uploads the winner as the
    round candidate, persists state and finishes the round with a real
    ``model_call_id`` and ``history_ids``.

``predict(sample, model_dir)``
    Loads the selected checkpoint plus its ``submission_meta.json`` and
    answers one JSONL line at a time.  Post-processing is applied per sample
    (causal streams), so the module stays compatible with the batch-scored
    hidden stream while never reading an answer column.

Design notes
------------
* The checkpoint stays a byte-compatible continuation of the official
  baseline: continuation candidates only call ``Model.fit`` with the official
  ``public/train.jsonl`` rows; inference-time calibration lives in the
  artifact metadata (``submission_meta.json``).
* Everything is bounded: one gateway call per round, at most two candidates,
  no writes outside ``/workspace``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from rsi.inference import (
    POSTPROCESS_KINDS,
    DamageShrinkStream,
    PostprocessStream,
    RecenterStream,
    apply_postprocess,
    apply_postprocess_stream,
)
from rsi.executor import ExperimentProposal
from rsi.metrics import from_predictions, primary as primary_metric
from rsi.model import ExtendedModel
from rsi.platform import agent_host as _agent_host
from rsi.platform import plans as _plans
from rsi.platform.c_safety import conservative_candidates, group_guard
from rsi.platform.b_safety import bearing_guard
from rsi.platform.experiment_cache import ExperimentCache
from rsi import __version__

TASKS = ("classification", "regression")

TASK_DEFAULTS = {
    "A": {"task": "classification", "metric": "macro_f1", "direction": "higher", "margin": 0.004},
    "B": {"task": "regression", "metric": "rmse", "direction": "lower", "margin": 5.0},
    "C": {"task": "regression", "metric": "rmse", "direction": "lower", "margin": 0.3},
}
TASK_KEYS = "ABC"
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
META_NAME = "submission_meta.json"
CHECKPOINT_NAME = "checkpoint.npz"
BOUNDARY_NAME = "data_boundary.json"


def _seed() -> int:
    try:
        return int(os.environ.get("RSI_V2_SEED", "20261117"))
    except ValueError:
        return 20261117


def _warmup_rounds() -> int:
    """Rounds that must keep the official parent artifact.

    Formal #2 (v0.14.9) disabled this after the warmup experiment showed the
    opposite of the original hypothesis: freezing stage-0 at the static baseline
    drove ``worst_stage`` from 0.3715 to 0 and lowered recovery (4.206 -> 3.730)
    without improving eligibility.  The capability is kept behind the env flag
    only for controlled experiments; production default is 0.
    """
    try:
        value = int(os.environ.get("RSI_V2_WARMUP_ROUNDS", "0"))
    except ValueError:
        return 0
    return max(0, min(5, value))


def _probe_retain_tasks() -> set[str]:
    """Tasks forced to keep the parent artifact in every round (probe runs).

    ``RSI_V2_PROBE_RETAIN=A,C`` turns a trial into a hidden-baseline probe: the
    listed datasets never promote a candidate, so the trial's hidden metrics are
    those of the selected parent for those datasets.  The final v0.14.12 image
    also enables this for C to preserve its baseline.  A retained round still
    evaluates the parent and requests a real gateway diagnosis.
    """
    value = os.environ.get("RSI_V2_PROBE_RETAIN", "").strip().lower()
    if not value or value in {"0", "off", "false", "no", "none"}:
        return set()
    if value in {"all", "*", "*.*"}:
        return {"A", "B", "C"}
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def _file_sha256(path) -> str:
    import hashlib
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _receipt_usage(receipt) -> dict:
    """Token usage of a single gateway receipt (best effort, never fails a round)."""
    try:
        usage = getattr(receipt, "usage", None)
        if usage is None:
            return {}
        if isinstance(usage, dict):
            data = usage
        else:
            data = {name: getattr(usage, name, 0) for name in
                    ("prompt_tokens", "completion_tokens", "total_tokens")}
        return {key: int(data.get(key) or 0) for key in
                ("prompt_tokens", "completion_tokens", "total_tokens")}
    except Exception:
        return {}


def _work_dir() -> Path:
    configured = os.environ.get("CONTEST_WORK")
    if configured:
        path = Path(configured)
    elif Path("/workspace").is_dir():
        path = Path("/workspace")
    else:
        path = Path(tempfile.mkdtemp(prefix="contest-workspace-"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "id" not in row or "inputs" not in row:
                raise ValueError(f"invalid public projection row in {path}")
            rows.append(row)
    return rows


def _load_specs(model: ExtendedModel) -> list[list[list[str]]]:
    raw = model.meta.get("paths") or []
    specs = []
    for item in raw:
        spec = json.loads(item) if isinstance(item, str) else item
        specs.append([list(entry) for entry in spec])
    return specs


def _flatten_row(specs: list[list[list[str]]], inputs: dict) -> np.ndarray:
    """Strict checkpoint paths with a positional fallback for private schemas."""
    return _plans.flatten_row(specs, inputs)


def _xy_rows(rows: list[dict], model: ExtendedModel, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    specs = _load_specs(model)
    features = np.empty((len(rows), len(specs)), dtype=np.float64)
    task = model.meta.get("task")
    labels = np.empty(len(rows), dtype=np.float64)
    index = {name: position for position, name in enumerate(classes)}
    for row, position in zip(rows, range(len(rows))):
        features[position] = _flatten_row(specs, row["inputs"])
        answer = row.get("answer")
        if task == "classification":
            if answer not in index:
                raise ValueError(f"unknown class label: {answer!r}")
            labels[position] = index[answer]
        else:
            labels[position] = float(answer)
    return features, labels, np.arange(len(rows))


def _evaluate(model: ExtendedModel, features: np.ndarray, labels: np.ndarray, task: str,
              classes: list[str], postprocess: dict | None,
              profile: str | None = None) -> tuple[dict, float, np.ndarray]:
    predicted = apply_postprocess(postprocess or {"type": "none"}, model, features)
    values = from_predictions(task, predicted, labels, classes, profile=profile)
    primary = values.get("primary")
    if primary is None:
        primary = primary_metric(task, values)
    return values, float(primary), np.asarray(predicted)


def _c_candidate_acceptable(candidate: dict, parent: dict, margin: float) -> bool:
    """Require a real RMSE gain and no MAE/NASA regression on the same rows."""
    if candidate.get("group_guard", {}).get("passed") is False:
        return False
    values, baseline = candidate.get("values", {}), parent.get("values", {})
    for metric in ("rmse", "mae", "nasa_score"):
        new, old = values.get(metric), baseline.get(metric)
        if new is None or old is None or not np.isfinite(new) or not np.isfinite(old):
            return False
        if new > old:
            return False
    return values["rmse"] < baseline["rmse"] - margin


def _classification_guard(ctx, chosen: dict) -> dict:
    """Final, route-independent public non-inferiority check (not a CI claim)."""
    values, primary, predicted = _plans.evaluate(ctx, chosen.get("model") or ctx.model,
                                                chosen.get("postprocess"))
    chosen.update(values=values, primary=primary, predicted=predicted)
    baseline = ctx.parent_values
    support = np.bincount(np.asarray(ctx.y_val, dtype=int), minlength=len(ctx.classes))
    threshold = max(10, int(np.ceil(len(ctx.y_val) * 0.05)))
    supported = [str(i) for i, count in enumerate(support) if count >= threshold]
    old_f1 = baseline.get("per_class_f1", {})
    new_f1 = values.get("per_class_f1", {})
    deltas = {key: float(new_f1.get(key, 0.0) - old_f1.get(key, 0.0))
              for key in supported}
    overall = float(values.get("macro_f1", float("nan"))) - float(baseline["macro_f1"])
    class_mean = float(np.mean(list(deltas.values()))) if deltas else float("nan")
    passed = bool(np.isfinite(overall) and np.isfinite(class_mean)
                  and overall >= -ctx.margin and class_mean >= -ctx.margin
                  and min(deltas.values(), default=-float("inf")) >= -2 * ctx.margin)
    return {"passed": passed, "overall_delta": overall if np.isfinite(overall) else None,
            "supported_class_mean_delta": class_mean if np.isfinite(class_mean) else None,
            "per_class_delta": deltas, "minimum_class_support": threshold,
            "margin": ctx.margin, "scope": "public point-estimate guard, not hidden-stage evidence"}


def _build_identity() -> dict:
    """Only publish the manifest's bounded, non-sensitive identity fields."""
    path = Path(os.environ.get("RSI_BUILD_MANIFEST", "/opt/zhixun/app/build-manifest.json"))
    try:
        if path.stat().st_size > 65536:
            return {"status": "oversized"}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("sdk", {}), dict):
            return {"status": "invalid_manifest", "version": __version__}
        identity = {key: data[key] for key in ("version", "git_commit", "git_dirty", "source_sha256")
                    if key in data}
        sdk = data.get("sdk") or {}
        identity.update(sdk_version=sdk.get("version"), sdk_archive_sha256=sdk.get("archive_sha256"))
        return identity
    except (OSError, ValueError, TypeError):
        return {"status": "not_available", "version": __version__}


def _choose(rows: list[dict], direction: str, margin: float, *, task_key=None) -> int:
    """Deterministic fallback: best candidate that beats the parent margin."""
    parent = next((row for row in rows if row["kind"] == "retain"), rows[0])
    best_position = 0
    best_value = float(parent["primary"])
    for position, row in enumerate(rows):
        if position == 0:
            continue
        if task_key == "C" and not _c_candidate_acceptable(row, parent, margin):
            continue
        value = float(row["primary"])
        better = (value < best_value) if direction == "lower" else (value > best_value)
        if better:
            best_position, best_value = position, value
    if best_position == 0:
        return 0
    parent_value = float(parent["primary"])
    if direction == "lower":
        return best_position if best_value < parent_value - margin else 0
    return best_position if best_value > parent_value + margin else 0


def _block_scores(task: str, predicted, labels, classes: list[str],
                  block_size: int = 64) -> list[float]:
    """Per-block primary metric, used as paired evidence for the selection gate.

    ``predicted`` may be a probability matrix (pipeline) or a 1-D label vector
    (tests / kernel arms); both are normalised to the official metric shape.
    """
    n = len(labels)
    if n == 0:
        return []
    predicted = np.asarray(predicted)
    if task == "classification" and predicted.ndim == 1:
        count = max(len(classes), int(predicted.max()) + 1 if predicted.size else 1)
        matrix = np.zeros((predicted.size, count), dtype=np.float64)
        matrix[np.arange(predicted.size), predicted.astype(int)] = 1.0
        predicted = matrix
    size = max(1, int(block_size))
    scores: list[float] = []
    for start in range(0, n, size):
        stop = min(n, start + size)
        if stop - start < 8:
            continue
        if task == "classification":
            values = from_predictions(
                task, predicted[start:stop], labels[start:stop], classes)
        else:
            values = from_predictions(
                task, predicted[start:stop], labels[start:stop], classes)
        primary = values.get("primary")
        if primary is None:
            primary = primary_metric(task, values)
        scores.append(float(primary))
    return scores


def _gate_choose(plans: list[dict], task: str, direction: str, margin: float,
                 labels, classes: list[str]):
    """Pick a candidate only when its paired block bootstrap clears zero+margin.

    Returns ``(position, decisions)`` where decisions aligns with ``plans``.
    """
    from rsi.eval.gate import evaluate_evidence

    n = len(labels)
    block_size = max(16, n // 8) if n else 64
    parent_scores = _block_scores(task, plans[0].get("predicted"), labels, classes, block_size)
    blocks = list(range(len(parent_scores)))
    lower_is_better = str(direction).strip().lower() == "lower"

    def _improvement(series: list[float]) -> list[float]:
        # Normalise every metric to "larger = better improvement" *before* the
        # gate.  For RMSE/MAE style metrics improvement is parent - candidate;
        # the gate itself then always operates in the "higher" convention.  This
        # removes any ambiguity around direction handling in the shared stats
        # helpers (compliance review flagged the previous implicit coupling).
        return [-float(value) for value in series] if lower_is_better else list(series)

    parent_improvement = _improvement(parent_scores)
    decisions: list = [None]
    for plan in plans[1:]:
        predicted = plan.get("predicted")
        if predicted is None:
            decisions.append(None)
            continue
        scores = _block_scores(task, predicted, labels, classes, block_size)
        decisions.append(evaluate_evidence(
            parent_improvement, _improvement(scores), blocks,
            margin=float(margin), direction="higher",
            arm=str(plan.get("kind") or ""), family="pipeline",
            reps=1000, min_blocks=3))
    promotable = [(position, decision) for position, decision in enumerate(decisions)
                  if decision is not None and decision.decision == "promote"]
    if not promotable:
        return 0, decisions
    position, _ = max(promotable, key=lambda item: (item[1].delta, -item[0]))
    return position, decisions




def _safety_net_plans(ctx) -> list[dict]:
    """Deterministic candidates always worth evaluating before trusting the kernel.

    The platform reveals historical labels in ``/data/train`` across rounds; the
    agent may spend its turn budget on diagnosis and never evaluate a cumulative
    retrain arm (observed in trials).  These arms are cheap and are gated by the
    same paired block bootstrap, so they can only replace the agent's choice when
    the evidence is positive.
    """
    plans = [_plans.retain_plan(ctx)]
    if ctx.task == "regression" and ctx.task_key == "B" and _b_retain_mode():
        print("train: B retain mode: safety-net regression arms disabled",
              file=sys.stderr, flush=True)
        return plans
    modes = _adapt_arm_mode()
    if modes != "off":
        try:
            plan = _plans.plan_for_proposal(ctx, ExperimentProposal(
                strategy="continue_train", params={"learning_rate": 1e-3, "epochs": 20},
                postprocess={"type": "none"}, hypothesis="cumulative-label retrain"))
            if plan.get("ok"):
                plans.append(plan)
            else:
                print("train: safety_net continue_train unavailable: %s"
                      % plan.get("error"), file=sys.stderr, flush=True)
        except Exception as exc:
            print("train: safety_net continue_train error: %s: %s"
                  % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        if modes in ("auto", "pair", "retrain_pair"):
            try:
                plan = _plans.plan_for_proposal(ctx, ExperimentProposal(
                    strategy="continue_train", params={"learning_rate": 1e-3, "epochs": 20},
                    postprocess={"type": "pair_threshold", "pair": [0, 3],
                                 "delta": _plans.PAIR_HISTORY_DELTA},
                    hypothesis="cumulative-label retrain + delayed-label pair"))
                if plan.get("ok"):
                    plans.append(plan)
                else:
                    print("train: safety_net retrain_pair unavailable: %s"
                          % plan.get("error"), file=sys.stderr, flush=True)
            except Exception as exc:
                print("train: safety_net retrain_pair error: %s: %s"
                      % (type(exc).__name__, exc), file=sys.stderr, flush=True)
    if ctx.task == "regression":
        # Guardrail arms for B/C: the agent has repeatedly picked validation-
        # improving retrain arms that over-fit the public ranking; evaluate the
        # target-mean shrink / damage-shrink arms as deterministic candidates.
        try:
            shrink = _plans.plan_for_proposal(ctx, ExperimentProposal(
                strategy="retain", params={},
                postprocess={"type": "shrink", "alpha": 0.3,
                             "center": float(ctx.target_mean)},
                hypothesis="target-mean shrink guardrail"))
            if shrink.get("ok"):
                plans.append(shrink)
        except Exception as exc:
            print("train: safety_net shrink error: %s: %s" % (type(exc).__name__, exc),
                  file=sys.stderr, flush=True)
        if ctx.task_key == "B":
            try:
                damage = _plans.plan_for_proposal(ctx, ExperimentProposal(
                    strategy="retain", params={},
                    postprocess={"type": "damage_shrink"},
                    hypothesis="damage-shrink guardrail"))
                if damage.get("ok"):
                    plans.append(damage)
            except Exception as exc:
                print("train: safety_net damage error: %s: %s" % (type(exc).__name__, exc),
                      file=sys.stderr, flush=True)
    try:
        import os as _os
        if _os.environ.get("RSI_V2_CAPACITY_ARM", "off").strip().lower() not in {"", "off", "0", "none"}:
            plans.extend([p for p in _plans.capacity_plans(ctx) if p.get("ok")])
    except Exception:
        pass
    seen, unique = set(), []
    for plan in plans:
        kind = str(plan.get("kind"))
        if kind in seen:
            continue
        seen.add(kind)
        unique.append(plan)
    return unique


def _parse_choice(text: str, upper: int) -> int | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"\{[^{}]*\}", text, re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            choice = payload.get("choice")
            if isinstance(choice, int) and 0 <= choice < upper:
                return choice
            if isinstance(choice, str):
                digits = re.search(r"\d+", choice)
                if digits and 0 <= int(digits.group(0)) < upper:
                    return int(digits.group(0))
        except json.JSONDecodeError:
            pass
    digits = re.search(r"\b([0-9])\b", text)
    if digits and 0 <= int(digits.group(1)) < upper:
        return int(digits.group(1))
    return None


def _contract_value(contract, keys, default):
    """Read a frozen-contract field by name, tolerating naming variants."""
    if isinstance(contract, dict):
        for key in keys:
            value = contract.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return default


def _adapt_arm_mode() -> str:
    value = os.environ.get("RSI_V2_ADAPT_ARM", "auto").strip().lower()
    if value in {"0", "off", "false", "no"}:
        return "off"
    return value if value in {"pair", "retrain", "retrain_pair"} else "auto"


def _adapt_arms_enabled() -> bool:
    # Adapt arms are independent of the drift-policy switch: E7 needs policy=off
    # while still letting the selection gate compare the retrain/pair arms.
    return _adapt_arm_mode() != "off"


def _b_retain_mode() -> bool:
    """B regression robustness probe.

    The static train-centre shrink is selected on the stale public validation
    split; on drifted low-RUL streams it can move predictions the wrong way.
    When enabled, B evaluates only the parent checkpoint (retain), which lets
    the platform trial measure the true baseline hidden RMSE.
    """
    value = os.environ.get("RSI_V2_B_RETAIN", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _candidate_plans(task_key: str, model: ExtendedModel, train, validation, *,
                     metric: str = "macro_f1", direction: str = "higher",
                     margin: float = 0.0, ctx=None) -> list[dict]:
    """Build candidate plan descriptors sorted by expected value (cheap first)."""
    task = model.meta.get("task")
    classes = list(model.meta.get("classes") or [])
    target_mean = float(model.meta.get("target_mean") or 0.0)
    # COMPLIANCE: ``fit_x``/``fit_y`` are built exclusively from the public
    # training projection.  Validation rows are only ever used for scoring and
    # selection below; they must never enter a gradient update.
    _assert_training_only(list(train), list(validation))
    ctx = ctx or _plans.PlanContext.from_rows(
        task_key, model, list(train), list(validation), metric=metric,
        direction=direction, margin=float(margin), classes=classes)
    fit_x, fit_y, x_val, y_val = ctx.x_train, ctx.y_train, ctx.x_val, ctx.y_val
    print("train: fit boundary rows=%d validation_rows=%d "
          "(validation_used_for_fit=False)" % (len(fit_x), len(x_val)),
          file=sys.stderr, flush=True)
    plans: list[dict] = [_plans.retain_plan(ctx)]
    parent_predicted = plans[0]["predicted"]
    efficient_c = task_key == "C" and os.environ.get("RSI_V2_C_EFFICIENT", "1") == "1"
    candidates = []
    if task == "classification":
        for weight, mode in ((0.4, "batch_mean"), (0.2, "batch_std")):
            candidates.append({
                "kind": "recenter",
                "label": f"推理期 recenter（{mode}, w={weight}），不改权重",
                "postprocess": {"type": "recenter", "mode": mode, "weight": weight},
                "checkpoint": None,
            })
        if task_key == "A":
            # NOTE(compliance): the pair delta is a *fixed historical* prior.
            # It must never be fitted from labels inside the training run, and
            # the fitted value must never be persisted into predict artifacts.
            mode = _adapt_arm_mode()
            if _adapt_arms_enabled() and mode in {"auto", "pair"}:
                candidates.append({
                    "kind": "pair_history",
                    "label": "历史校准 pair 阈值（delta=%.3f，不重新拟合）" % _plans.PAIR_HISTORY_DELTA,
                    "postprocess": {"type": "pair_threshold", "pair": [0, 3],
                                    "delta": _plans.PAIR_HISTORY_DELTA},
                    "checkpoint": None,
                })
            if _adapt_arms_enabled() and mode in {"auto", "retrain", "retrain_pair"}:
                candidates.append({
                    "kind": "continue_train",
                    "label": "累积标签续训 20 epoch（用当前 /data/train 全部已揭示行）",
                    "postprocess": {"type": "none"},
                    "checkpoint": "CONTINUE_TRAIN",
                    "epochs": 20,
                })
            if _adapt_arms_enabled() and mode in {"pair", "retrain_pair"}:
                candidates.append({
                    "kind": "continue_train_pair",
                    "label": "累积标签续训 + 历史校准 pair 阈值",
                    "postprocess": {"type": "pair_threshold", "pair": [0, 3],
                                    "delta": _plans.PAIR_HISTORY_DELTA},
                    "checkpoint": "CONTINUE_TRAIN",
                    "epochs": 20,
                })
    else:
        b_retain = task_key == "B" and _b_retain_mode()
        if b_retain:
            print("train: B retain mode: static/retrain arms disabled "
                  "(baseline measurement)", file=sys.stderr, flush=True)
        # B: static shrink/damage over-fit the dominant validation bearing and
        # collapsed on its hidden tail (0.14.5 trial: hidden RMSE 735 vs 720 for
        # retain). Keep them behind RSI_V2_B_STATIC; the default B arm continues
        # the real parent and must pass the final per-bearing guard.
        if not b_retain and not efficient_c and (task_key != "B" or _plans.b_static_mode()):
            candidates.append({
                "kind": "shrink",
                "label": "train 目标均值收缩校准（α=0.3），不改权重",
                "postprocess": {"type": "shrink", "alpha": 0.3, "center": target_mean},
                "checkpoint": None,
            })
        if task_key == "B" and not b_retain:
            if _plans.b_static_mode():
                candidates.append({
                    "kind": "damage_shrink",
                    "label": "damage_shrink 因果分段校准（train 标签拟合）",
                    "postprocess": "FIT_DAMAGE",
                    "checkpoint": None,
                })
            candidates.append({
                "kind": "continue_train", "label": "从实际父检查点短程续训 B",
                "postprocess": {"type": "none"}, "checkpoint": "CONTINUE_TRAIN",
                **_plans.b_continuation_parameters(),
            })
        if task_key == "C" and not b_retain:
            candidates.append({
                "kind": "continue_train",
                "label": ("官方 train 续训 10 epoch（lr=3e-4，RMSE/MAE/NASA 联合检查）"
                          if efficient_c else "官方 train 续训 40 epoch（lr=1e-3，checkpoint 兼容）"),
                "postprocess": {"type": "none"},
                "checkpoint": "CONTINUE_TRAIN_40",
                "epochs": 10 if efficient_c else 40,
                "learning_rate": 3e-4 if efficient_c else 1e-3,
            })
    if task == "regression" and not efficient_c and not (task_key == "B" and _b_retain_mode()):
        try:
            for plan in _plans.scratch_plans(ctx):
                if not plan.get("ok"):
                    continue
                candidates.append({
                    "kind": plan["kind"],
                    "label": plan["label"],
                    "postprocess": plan["postprocess"],
                    "checkpoint": "SCRATCH",
                    "model": plan["model"],
                    "primary": plan["primary"],
                    "values": plan["values"], "predicted": plan["predicted"],
                    "hypothesis": plan.get("hypothesis") or plan["label"],
                })
        except Exception as error:
            print("train: scratch arm unavailable: %s: %s" % (type(error).__name__, error),
                  file=sys.stderr, flush=True)
    if task == "classification" and _plans.capacity_arm_mode() not in ("", "off", "none", "0", "false"):
        try:
            for plan in _plans.capacity_plans(ctx):
                if not plan.get("ok"):
                    continue
                candidates.append({
                    "kind": plan["kind"],
                    "label": plan["label"],
                    "postprocess": plan["postprocess"],
                    "checkpoint": "CAPACITY",
                    "model": plan["model"],
                    "primary": plan["primary"],
                    "values": plan["values"], "predicted": plan["predicted"],
                    "hypothesis": plan.get("hypothesis") or plan["label"],
                })
        except Exception as error:
            print("train: capacity arms unavailable: %s: %s" % (type(error).__name__, error),
                  file=sys.stderr, flush=True)
    for plan in candidates:
        postprocess = plan["postprocess"]
        checkpoint_kind = plan.get("checkpoint")
        if checkpoint_kind in ("CONTINUE_TRAIN_40", "CONTINUE_TRAIN"):
            params = {"epochs": int(plan.get("epochs", 20)),
                      "learning_rate": float(plan.get("learning_rate", 1e-3))}
            # B preserves the actual parent's seed; A/C retain the documented
            # run seed. In all cases weights, scaling and Adam come from parent.
            if task_key != "B":
                params["seed"] = _seed()
            try:
                candidate_model, cache_note = _plans.cached_continuation(ctx, params)
                plan["experiment_cache"] = cache_note
            except Exception as exc:
                print("train: continuation rejected: %s: %s" %
                      (type(exc).__name__, exc), file=sys.stderr, flush=True)
                continue
        else:
            candidate_model = plan.get("model") or model
        if efficient_c and checkpoint_kind == "CONTINUE_TRAIN_40":
            try:
                max_alpha = float(os.environ.get("RSI_V2_C_MAX_ALPHA", "0.25"))
                if max_alpha not in (0.25, 0.5):
                    raise ValueError("RSI_V2_C_MAX_ALPHA must be 0.25 or 0.5")
                for alpha, conservative in conservative_candidates(model, candidate_model):
                    if alpha > max_alpha:
                        continue
                    values, primary, predicted = _evaluate(
                        conservative, x_val, y_val, task, classes, postprocess, profile=task_key)
                    report = group_guard(parent_predicted, predicted, y_val,
                                         [str(row.get("id", "")) for row in validation],
                                         enforce_rul=True)
                    plans.append({**plan, "kind": "continue_train_conservative",
                                  "label": "C 续训后保守权重更新 alpha=%s" % alpha,
                                  "model": conservative, "values": values, "primary": primary,
                                  "predicted": predicted, "y_train": fit_y,
                                  "alpha": alpha, "group_guard": report})
            except Exception as exc:
                print("train: conservative C rejected: %s: %s" %
                      (type(exc).__name__, exc), file=sys.stderr, flush=True)
            continue
        if postprocess == "FIT_DAMAGE":
            from rsi.inference import fit_damage_shrink
            dataset = _as_dataset(fit_x, fit_y, classes, task)
            postprocess = fit_damage_shrink(dataset)
        if all(key in plan for key in ("values", "primary", "predicted")):
            values, primary, predicted = plan["values"], plan["primary"], plan["predicted"]
        else:
            values, primary, predicted = _evaluate(
                candidate_model, x_val, y_val, task, classes, postprocess, profile=task_key)
        if task_key == "C":
            plan["group_guard"] = group_guard(
                parent_predicted, predicted, y_val,
                [str(row.get("id", "")) for row in validation], enforce_rul=True)
        if task_key == "B":
            plan["group_guard"] = bearing_guard(
                parent_predicted, predicted, y_val,
                [str(row.get("id", "")) for row in validation])
        plans.append({
            **plan,
            "postprocess": postprocess,
            "values": values,
            "primary": primary,
            "predicted": predicted,
            "model": candidate_model,
            "y_train": fit_y,
        })
    return plans


def _as_dataset(x, y, classes, task):
    from rsi.data import Dataset

    return Dataset(name="official-train", ids=[str(i) for i in range(len(x))], x=x, y=y,
                   paths=[], sha256="", groups=["all"] * len(x))


def _format_plan_table(plans: list[dict], metric: str) -> str:
    lines = []
    for position, plan in enumerate(plans):
        lines.append("%d. [%s] %s -> %s=%.6f" % (
            position, plan["kind"], plan["label"], metric, float(plan["primary"])))
    return "\n".join(lines)


def _change_summary(chosen: dict) -> str:
    kind = chosen["kind"]
    if kind == "retain":
        return "保留父模型，不改权重与推理期后处理"
    if kind.startswith("continue_train"):
        return "基于官方 train 的短续训，父谱系与结构兼容"
    if kind == "scratch_retrain":
        return "从零重训官方 v1 单隐层 tanh（仅公开 train，validation 只用于选择）"
    return "仅推理期后处理（%s），未改 checkpoint" % kind


def _dedupe_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate the projected training rows by id, keeping the newest copy.

    The platform re-projects earlier stages every round; a row can therefore
    appear more than once.  Duplicates would silently re-weight the fit, so the
    participant de-duplicates before anything is trained.
    """
    seen: dict[str, int] = {}
    ordered: list[dict] = []
    duplicates = 0
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id and row_id in seen:
            ordered[seen[row_id]] = row
            duplicates += 1
            continue
        if row_id:
            seen[row_id] = len(ordered)
        ordered.append(row)
    return ordered, duplicates


def _rows_fingerprint(rows: list[dict]) -> str:
    """Stable fingerprint of the row ids making up a public projection."""
    import hashlib

    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(str(row.get("id") or "").encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def _assert_training_only(fit_rows: list[dict], validation_rows: list[dict]) -> None:
    """Fail the round rather than silently train on the validation projection.

    TRAINING_GUIDE freezes the data boundary: gradients may only use
    ``public/train.jsonl``; ``public/validation.jsonl`` is measurement/selection
    only.  The fit call in ``_candidate_plans`` is therefore asserted to receive
    exactly the training projection, with no id overlap with validation.
    """
    train_ids = {str(row.get("id") or "") for row in fit_rows}
    validation_ids = {str(row.get("id") or "") for row in validation_rows}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            "data boundary violation: %d row ids appear in both training and "
            "validation projections" % len(overlap))


def _data_probe(train_rows: list[dict], validation_rows: list[dict],
                duplicates: int, round_index: int) -> dict:
    """Cheap, honest projection fingerprint written into the saved state."""
    import hashlib

    def digest(rows: list[dict]) -> str:
        hasher = hashlib.sha256()
        for row in rows:
            hasher.update(str(row.get("id") or "").encode("utf-8"))
        return hasher.hexdigest()[:16]

    def content_digest(rows: list[dict]) -> str:
        hasher = hashlib.sha256()
        for row in rows:
            payload = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8")
            hasher.update(len(payload).to_bytes(8, "big"))
            hasher.update(payload)
        return hasher.hexdigest()

    label_counts: dict[str, int] = {}
    for row in train_rows:
        key = str(row.get("answer"))
        label_counts[key] = label_counts.get(key, 0) + 1
    return {
        "round_index": int(round_index),
        "train_rows": len(train_rows),
        "duplicate_rows": int(duplicates),
        "train_id_sha256": digest(train_rows),
        "train_content_sha256": content_digest(train_rows),
        "validation_rows": len(validation_rows),
        "validation_id_sha256": digest(validation_rows),
        "validation_content_sha256": content_digest(validation_rows),
        "label_counts": label_counts,
    }


def _state_payload(rnd, candidates, chosen, note: str, probe: dict | None = None,
                   resource: dict | None = None) -> dict:
    compact = []
    for candidate in candidates or []:
        primary = candidate.get("primary")
        compact.append({"kind": candidate.get("kind"),
                        "primary": float(primary) if primary is not None else None})
    return {
        "round_index": int(rnd.round_index),
        "dataset_key": rnd.dataset_key,
        "chosen": chosen.get("kind"),
        "postprocess": chosen.get("postprocess"),
        "candidates": compact,
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "note": note,
        "data_probe": probe,
        "resource": resource,
    }


# ---------------------------------------------------------------------------
# controller routing: deterministic pipeline vs autonomous PI-agent kernel
# ---------------------------------------------------------------------------
def _gate_mode() -> str:
    """``on`` (default) uses the paired block-bootstrap gate for selection.

    ``RSI_V2_GATE=off`` restores the legacy point-estimate selector; it exists
    only for controlled A/B experiments (E7) and must not be used in production.
    """
    value = os.environ.get("RSI_V2_GATE", "on").strip().lower()
    return "off" if value in {"0", "off", "false", "no"} else "on"


def _controller_mode(task: str | None = None) -> str:
    """``pipeline`` by default; the packaged image enables ``auto``.

    ``auto`` keeps the PI-agent kernel for classification tasks (where drift
    diagnosis and history-calibrated post-processing matter most) and uses the
    deterministic pipeline for regression tasks (where the public-validation
    ranking has repeatedly over-fit).
    """
    explicit = os.environ.get("RSI_V2_CONTROLLER", "").strip().lower()
    if explicit == "auto":
        return "kernel" if (task or "").strip().lower() == "classification" else "pipeline"
    if explicit in ("pipeline", "kernel"):
        return explicit
    default = os.environ.get("RSI_V2_KERNEL_DEFAULT", "0").strip()
    return "kernel" if default == "1" else "pipeline"


def _drift_brief(ctx) -> dict:
    """Cheap train-vs-validation covariate summary (no labels, no hidden data)."""
    try:
        train_mean = np.asarray(ctx.x_train.mean(axis=0), dtype=np.float64)
        train_std = np.asarray(ctx.x_train.std(axis=0), dtype=np.float64) + 1e-9
        val_mean = np.asarray(ctx.x_val.mean(axis=0), dtype=np.float64)
        z = (val_mean - train_mean) / train_std
        order = np.argsort(-np.abs(z))[:5]
        top = {str(int(index)): round(float(z[index]), 3) for index in order}
        peak = float(np.max(np.abs(z))) if len(z) else 0.0
        severity = "high" if peak >= 2.0 else ("medium" if peak >= 1.0 else "low")
        arrival = {}
        try:
            # Batch-vs-reference drift: as revealed batches accumulate, the
            # aggregate train~val statistic above can wash out and hide the
            # ongoing shift. Compare the newest revealed chunk against the
            # model's frozen reference statistics instead.
            model = getattr(ctx, "model", None)
            if model is not None and len(ctx.x_train) >= 32:
                recent = np.asarray(ctx.x_train[-min(512, len(ctx.x_train)):],
                                    dtype=np.float64)
                ref_mean = np.asarray(model.mean, dtype=np.float64)
                ref_scale = np.asarray(model.scale, dtype=np.float64) + 1e-9
                z2 = (recent.mean(axis=0) - ref_mean) / ref_scale
                peak2 = float(np.max(np.abs(z2))) if z2.size else 0.0
                sev2 = "high" if peak2 >= 2.0 else ("medium" if peak2 >= 1.0 else "low")
                arrival = {"peak": round(peak2, 4), "severity": sev2,
                           "rows": int(len(recent))}
                order2 = np.argsort(-np.abs(z2))[:3]
                arrival["top"] = {str(int(i)): round(float(z2[i]), 3) for i in order2}
                rank = {"low": 0, "medium": 1, "high": 2}
                if rank.get(sev2, -1) > rank.get(severity, -1):
                    severity = sev2
                if peak2 > peak:
                    peak = peak2
        except Exception:
            arrival = {}
        return {
            "severity": severity,
            "score": round(peak, 4),
            "drift_type": "covariate_shift",
            "feature_shift": top,
            "max_feature_shift": {"index": int(order[0]) if len(order) else -1,
                                  "z": round(peak, 3)},
            "arrival": arrival,
            "evidence": {"train_rows": int(len(ctx.x_train)),
                         "val_rows": int(len(ctx.x_val))},
            "causes": ["公开验证投影与 train 投影的特征分布差异（无标签）",
                       "已揭示新批相对模型参考分布的漂移（无标签）"],
            "knowledge_hints": [],
        }
    except Exception:
        return {"severity": "unknown", "score": 0.0, "drift_type": "unknown"}


def _kernel_brief(task_key: str, model: ExtendedModel, ctx, history: list[dict],
                  round_index: int, rounds_total: int, state) -> dict:
    from rsi.knowledge import drift_context_for_brief

    drift = _drift_brief(ctx)
    memory_text = ""
    lessons_text = ""
    stats_text = ""
    skills_text = ""
    try:
        if state is not None and state.memory is not None:
            memory_text = state.memory.snapshot(task_key)
    except Exception:
        memory_text = ""
    try:
        if state is not None and state.recall is not None:
            lessons_text = state.recall.lessons_text(task_key)
            stats_text = state.recall.stats_text(task_key)
    except Exception:
        lessons_text, stats_text = "", ""
    try:
        if state is not None and state.skills is not None:
            skills_text = state.skills.index_text()
    except Exception:
        skills_text = ""
    knowledge_text = ""
    try:
        knowledge_text = drift_context_for_brief(drift, k=3)
    except Exception:
        knowledge_text = ""
    if os.environ.get("RSI_V2_KERNEL_CONTEXT", "full").strip().lower() == "minimal":
        skills_text = ""
        knowledge_text = ""
    return {
        "task": {"key": task_key, "name": task_key, "name_en": task_key,
                 "task": ctx.task, "primary_metric": ctx.metric,
                 "direction": ctx.direction,
                 "baseline_primary": float(ctx.parent_primary),
                 "margin": float(TASK_DEFAULTS[task_key]["margin"])},
        "goal": {"goal": "在漂移数据流上维持并提升 %s 的 %s" % (task_key, ctx.metric),
                 "status": "running", "turns_used": 0,
                 "max_turns": int(os.environ.get("RSI_V2_MAX_TURNS", "10") or 10)},
        "round_index": int(round_index),
        "rounds_total": int(rounds_total),
        "metrics": {"parent": {"primary": float(ctx.parent_primary),
                               "report": dict(ctx.parent_values or {})}},
        "budget": {"token_cap": int(os.environ.get("RSI_V2_TOKEN_CAP", "0") or 0),
                   "calls": 0, "note": "platform budget is enforced by the host"},
        "memory": memory_text,
        "lessons": lessons_text,
        "stats": stats_text,
        "history": list(history or []),
        "drift": drift,
        "policy": {
            "drift_level": drift.get("severity"),
            "prefer_delayed_label_on_high_drift": True,
            "note": ("高漂移分类任务：优先 pair_threshold(delta=%.1f) 且只要公开验证不劣于"
                     "父模型 2*margin 就采用；中漂移只用 recenter；低漂移 retain。"
                     % _plans.PAIR_HISTORY_DELTA),
        },
        "catalog": _plans.catalog_text(),
        "guidance": _kernel_guidance(ctx, drift),
        "skills": skills_text,
        "knowledge": knowledge_text,
    }


def _kernel_guidance(ctx, drift: dict) -> str:
    """Round guidance shown to the kernel (agent decides; this is not a rule)."""
    if ctx.task == "classification":
        return ("已揭示的累积训练行随轮次增长；公开验证(b2)已接近饱和，验证分不变不代表没有"
                "提升空间。高漂移下优先尝试 continue_train / continue_train_pair（用当前全部"
                "已揭示训练行），不要因为验证分不变就连续 retain；可用 compare_candidates 对比"
                "各候选的公开验证值后再决定。")
    return ("回归任务优先考虑 target-mean shrink 以及小步 continue_train；公开验证与隐藏批"
            "可能不同步，不要只挑验证最优，优先选择相对父模型稳定不劣的候选。")


def _open_agent_state(work: Path, rnd):
    try:
        return _agent_host.open_state(work, rnd)
    except Exception as exc:
        print("train: agent state unavailable: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr, flush=True)
        return _agent_host.AgentState()


def _choose_kernel_plan(kernel_round, ctx, direction: str) -> dict:
    """Map the kernel decision onto a concrete, already-evaluated plan."""
    decision = kernel_round.decision or {}
    if str(decision.get("decision")) == "promote":
        host = kernel_round.host
        plan = host.plan_for(decision.get("experiment_id")) if host is not None else None
        if plan is not None and plan.get("ok"):
            return plan
    return _plans.retain_plan(ctx)

def _apply_drift_policy(ctx, chosen: dict, plans=None) -> tuple[dict, str]:
    """Gen2 drift-aware override for the delayed-label arm.

    On high covariate drift (task A) the historical pair delta is preferred when
    its public validation is *not materially worse* than the parent.  This is
    deliberately not a "best validation wins" rule: the platform trial showed
    that the public-validation optimum can invert on the hidden batch.

    ``RSI_V2_DRIFT_POLICY=0`` turns the override off so controlled experiments
    can compare the live policy against the deterministic pipeline choice.
    """
    if os.environ.get("RSI_V2_DRIFT_POLICY", "1").strip().lower() in {
            "0", "off", "false", "no"}:
        print("train: drift_policy: disabled by RSI_V2_DRIFT_POLICY=0",
              file=sys.stderr, flush=True)
        return chosen, ""
    try:
        drift = _drift_brief(ctx)
        margin = float(getattr(ctx, "margin", 0.0) or 0.0)
        parent_primary = float(ctx.parent_primary)
        usable = [plan for plan in (plans or [])
                  if isinstance(plan, dict) and plan.get("primary") is not None
                  and not plan.get("error")]
        if (str(drift.get("severity")) == "high" and ctx.task == "classification"
                and usable):
            adapts = [plan for plan in usable
                      if str(plan.get("kind", "")).startswith("continue_train")]
            best = max(adapts, key=lambda plan: float(plan["primary"]), default=None)
            if best is not None and float(best["primary"]) >= parent_primary + margin:
                return best, ("drift_policy: high drift %.2f, %s val=%.4f parent=%.4f "
                              "(cumulative labels)" % (
                                  float(drift.get("score") or 0.0), best.get("kind"),
                                  float(best["primary"]), parent_primary))
            # Saturated public validation: once the parent already scores > .95 on
            # the public split, a 0.4-1.0pt drop there is noise, but freezing the
            # model loses the newly revealed labels. Prefer a non-inferior
            # cumulative retrain when the drift level is high.
            if (best is not None and parent_primary >= 0.95
                    and float(best["primary"]) >= parent_primary - 3.0 * margin):
                return best, ("drift_policy: high drift %.2f, saturated validation "
                              "(parent=%.4f), prefer %s val=%.4f (cumulative labels)" % (
                                  float(drift.get("score") or 0.0), parent_primary,
                                  best.get("kind"), float(best["primary"])))
            hist = [plan for plan in usable if str(plan.get("kind")) == "pair_history"]
            if hist and float(hist[0]["primary"]) + 2.0 * margin >= parent_primary:
                if str(chosen.get("kind")) == "pair_history":
                    return chosen, ""
                return hist[0], ("drift_policy: high drift %.2f, pair_history val=%.4f "
                                 "parent=%.4f" % (
                                     float(drift.get("score") or 0.0),
                                     float(hist[0]["primary"]), parent_primary))
        if os.environ.get("RSI_DEBUG_DRIFT", "").strip() in {"1", "true", "yes"}:
            _dbg = [("%s:%.4f" % (p.get("kind"), float(p.get("primary") or -1)))
                    for p in usable]
            _dbg_best = max([float(p.get("primary") or -1) for p in usable
                             if str(p.get("kind", "")).startswith("continue_train")],
                            default=-1)
            print("train: drift debug severity=%s score=%s arrival=%s parent=%s margin=%s "
                  "chosen=%s usable=%s best_retrain=%.4f"
                  % (drift.get("severity"), drift.get("score"),
                     (drift.get("arrival") or {}).get("peak"),
                     ctx.parent_primary, ctx.margin, chosen.get("kind"), _dbg, _dbg_best),
                  file=sys.stderr, flush=True)
        if str(drift.get("severity")) != "high" or ctx.task != "classification":
            return chosen, ""
        arm = _plans.pair_history_plan(ctx)
        if not arm.get("ok"):
            return chosen, ""
        if float(arm["primary"]) + 2.0 * margin < parent_primary:
            return chosen, ""
        if str(chosen.get("kind")) == "pair_history":
            return chosen, ""
        return arm, "drift_policy: high drift %.2f, pair_history val=%.4f parent=%.4f" % (
            float(drift.get("score") or 0.0), float(arm["primary"]), parent_primary)
    except Exception as exc:
        print("train: drift policy skipped: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr, flush=True)
        return chosen, ""


def _json_safe(value, notes=None, path="$", depth=0):
    if depth > 20:
        return "[depth limit]"
    if isinstance(value, dict):
        return {str(key): _json_safe(item, notes, path + "." + str(key), depth + 1)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, notes, path + "[%d]" % i, depth + 1)
                for i, item in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        if notes is not None and len(notes) < 16:
            notes.append(path)
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _trim_config(payload: dict, limit: int = 60_000) -> dict:
    notes = []
    safe = _json_safe(payload, notes)
    if notes:
        safe["serialization_notes"] = {"nonfinite_replaced_with_null": notes}
    text = json.dumps(safe, ensure_ascii=False, allow_nan=False)
    if len(text.encode("utf-8")) <= limit:
        return safe
    # Bound encoded bytes, not characters. Preserve useful summary identities.
    brief = {key: safe[key] for key in ("controller", "chosen", "decision", "build_identity",
             "parent_checkpoint_sha256", "candidate_checkpoint_sha256", "resource") if key in safe}
    brief.update(truncated=True, original_bytes=len(text.encode("utf-8")))
    return brief


def _gateway_model(rnd, messages, max_tokens: int):
    """One real gateway call with bounded retries.

    The platform requires at least one real call per round; transient gateway
    errors should not turn into a failed round.  Each retry receives its own
    real ``call_id`` and the last successful receipt is the one reported.
    """
    last_error = None
    for attempt in range(3):
        try:
            return rnd.model(messages, max_tokens=max_tokens, temperature=0)
        except Exception as error:  # pragma: no cover - depends on host gateway
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def _new_state_dir(work: Path, rnd) -> Path:
    """Copy the previous native state into a fresh writable directory when the
    host mounted one (workspace-v2 recommends ``rnd.state_workspace``)."""
    destination = work / ("state-round-%d-%d" % (int(rnd.round_index), os.getpid()))
    try:
        return rnd.state_workspace(str(destination))
    except Exception as exc:  # no reusable state is safe, but must be observable
        print("train: state restore unavailable: %s" % type(exc).__name__,
              file=sys.stderr, flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        return destination


def train(rnd):
    """One workspace v2 round."""
    cpu_start = time.process_time()
    wall_start = time.time()
    task_key = str(rnd.dataset_key).upper()
    if task_key not in TASK_DEFAULTS:
        raise ValueError(f"unsupported dataset key: {task_key}")
    contract = rnd.task or {}
    defaults = TASK_DEFAULTS[task_key]
    task = _contract_value(contract, ("task", "task_type", "type"), defaults["task"])
    if task not in TASKS:
        task = defaults["task"]
    metric = _contract_value(contract, ("metric", "metric_name"), defaults["metric"])
    direction = _contract_value(contract, ("direction", "metric_direction"),
                                defaults["direction"])

    parent_path = rnd.parent.path(CHECKPOINT_NAME)
    model = ExtendedModel.from_checkpoint(parent_path)
    model.meta["_source_path"] = str(parent_path)
    checkpoint_task = str(model.meta.get("task"))
    if checkpoint_task != task:
        # The frozen host contract can disagree with the official package
        # contract (observed on dataset B: host declares
        # classification/macro_f1 while the official checkpoint and package
        # are regression/RUL-minutes+RMSE). Hard-failing here happens *before*
        # the required gateway model call and makes a whole dataset unable to
        # complete any round. Follow the real checkpoint task instead and keep
        # the round protocol-compliant.
        if checkpoint_task not in TASKS:
            raise ValueError(
                "checkpoint task mismatch: %r != %r" % (checkpoint_task, task))
        print(
            "train: host contract task %r disagrees with checkpoint task %r; "
            "following checkpoint task for this round" % (task, checkpoint_task),
            file=sys.stderr,
            flush=True,
        )
        task = checkpoint_task
        metric = defaults["metric"]
        direction = defaults["direction"]

    train_rows = list(rnd.data.training_rows())
    validation_rows = list(rnd.data.validation_rows())
    if not train_rows or not validation_rows:
        raise ValueError("empty public projection")
    train_rows, duplicate_rows = _dedupe_rows(train_rows)
    probe = _data_probe(train_rows, validation_rows, duplicate_rows, int(rnd.round_index))
    print("train: data probe rows=%d duplicates=%d val=%d labels=%s" % (
        probe["train_rows"], probe["duplicate_rows"], probe["validation_rows"],
        sorted(probe["label_counts"].items())[:8]), file=sys.stderr, flush=True)

    history = []
    try:
        history = rnd.history()
    except Exception:  # host may legitimately return no history
        history = []

    # -- route: deterministic pipeline (default) or autonomous PI-agent kernel
    probe_retain = _probe_retain_tasks()
    probe_retain_applied = task_key in probe_retain
    efficient_c = task_key == "C" and os.environ.get("RSI_V2_C_EFFICIENT", "1") == "1"
    controller = "pipeline" if probe_retain_applied or efficient_c else _controller_mode(task)
    kernel_round = None
    kernel_ctx = None
    work = _work_dir()
    kernel_state = _open_agent_state(work, rnd) if controller == "kernel" else None
    state_dir = (Path(kernel_state.dir) if kernel_state is not None and kernel_state.dir
                 else _new_state_dir(work, rnd))
    experiment_cache = ExperimentCache(state_dir / "experiment_cache")
    gate_decisions = None
    if controller == "kernel":
        try:
            kernel_ctx = _plans.PlanContext.from_rows(
                task_key, model, train_rows, validation_rows,
                metric=metric, direction=direction,
                margin=float(defaults["margin"]),
                classes=list(model.meta.get("classes") or []),
                experiment_cache=experiment_cache)
            brief = _kernel_brief(task_key, model, kernel_ctx, history,
                                  int(rnd.round_index),
                                  int(getattr(rnd, "rounds_total", 10) or 10),
                                  kernel_state)
            kernel_round = _agent_host.run_kernel_round(
                rnd, kernel_ctx, brief, kernel_state,
                run_id="%s-r%d" % (task_key, int(rnd.round_index)),
                round_index=int(rnd.round_index),
                experiments_total=len(history),
                max_turns=int(os.environ.get("RSI_V2_MAX_TURNS", "10") or 10))
            print("train: kernel round calls=%d decision=%s" % (
                kernel_round.calls,
                (kernel_round.decision or {}).get("decision")),
                file=sys.stderr, flush=True)
        except Exception as exc:
            print("train: kernel route failed, falling back to pipeline: %s: %s" % (
                type(exc).__name__, exc), file=sys.stderr, flush=True)
            kernel_round, kernel_ctx, kernel_state = None, None, None

    kernel_effective = False
    kernel_call_ids = []
    kernel_inactive_note = ""
    if kernel_round is not None and kernel_round.calls > 0 and kernel_ctx is not None:
        kernel_call_ids = list(kernel_round.call_ids)
        if kernel_round.decision:
            kernel_effective = True
        else:
            kernel_inactive_note = "kernel produced no decision (%s)" % (
                kernel_round.error or "no tool action")
            print("train: kernel ineffective -> pipeline fallback: %s"
                  % kernel_inactive_note, file=sys.stderr, flush=True)

    plans = None
    policy_plans = None
    receipt = None
    plans = None
    llm_choice = None
    fallback_choice = None
    if kernel_effective:
        chosen = _choose_kernel_plan(kernel_round, kernel_ctx, direction)
        candidates = list(kernel_round.host.results) if kernel_round.host else []
        call_ids = list(kernel_round.call_ids)
        decision_note = dict(kernel_round.decision or {})
        controller_note = "kernel"
        choice_note = "内核决策 %s" % decision_note.get("decision", "retain")
        # Safety net: evaluate cumulative-retrain/capacity arms that the agent may
        # have skipped, but only override when the paired gate promotes one.
        safety = []
        try:
            safety = _safety_net_plans(kernel_ctx) if kernel_ctx is not None else []
            if len(safety) > 1:
                class_labels = list(model.meta.get("classes") or [])
                _x_val, y_val, _ = _xy_rows(validation_rows, model, class_labels)
                position, safety_gate = _gate_choose(
                    safety, task, direction, float(defaults["margin"]), y_val, class_labels)
                note = {str(plan.get("kind")): decision.to_dict()
                        for plan, decision in zip(safety, safety_gate) if decision is not None}
                decision_note["safety_gate"] = note
                if position > 0:
                    chosen = safety[position]
                    choice_note = "safety_net 覆盖内核 -> %s" % chosen.get("kind")
                    controller_note = "kernel+safety"
                    print("train: safety_net promoted %s (gate=%s)" % (
                        chosen.get("kind"),
                        safety_gate[position].decision), file=sys.stderr, flush=True)
                print("train: safety_net " + ", ".join(
                    "%s=%s(%.4f)" % (plan.get("kind"), decision.decision, decision.delta)
                    for plan, decision in zip(safety, safety_gate) if decision is not None),
                    file=sys.stderr, flush=True)
        except Exception as exc:
            print("train: safety_net unavailable: %s: %s" % (
                type(exc).__name__, exc), file=sys.stderr, flush=True)
        # Recent-block validation: once training absorbs drifted drops, the
        # frozen public validation (b2) stops predicting the next drop.  Hold out
        # the newest revealed rows, train candidates on the older rows, and use a
        # paired gate on the fresh block.  The agent still owns the decision; this
        # only surfaces an unbiased holdout to the same safety-net machinery.
        try:
            if (task == "classification" and len(train_rows) >= 800
                    and str(_drift_brief(kernel_ctx).get("severity")) == "high"):
                holdout_n = min(2000, max(200, int(len(train_rows) * 0.25)))
                holdout_rows = train_rows[-holdout_n:]
                past_rows = train_rows[:-holdout_n]
                recent_ctx = _plans.PlanContext.from_rows(
                    task_key, model, past_rows, holdout_rows,
                    metric=metric, direction=direction,
                    margin=float(defaults["margin"]),
                    classes=list(model.meta.get("classes") or []),
                experiment_cache=experiment_cache)
                recent = _safety_net_plans(recent_ctx)
                if len(recent) > 1:
                    r_labels = list(model.meta.get("classes") or [])
                    r_pos, r_gate = _gate_choose(
                        recent, task, direction, float(defaults["margin"]),
                        recent_ctx.y_val, r_labels)
                    r_note = {str(plan.get("kind")): decision.to_dict()
                              for plan, decision in zip(recent, r_gate)
                              if decision is not None}
                    decision_note["recent_holdout"] = r_note
                    promote_pos = r_pos if (r_pos > 0 and str(
                        recent[r_pos].get("kind", "")).startswith("continue_train")) else 0
                    if promote_pos == 0:
                        # No significant promote: accept a non-inferior retrain on
                        # the fresh holdout rather than freezing the model -- the
                        # cost of freezing under continuing drift is much larger.
                        parent_hold = float(recent_ctx.parent_primary)
                        retrains = [(i, plan) for i, plan in enumerate(recent)
                                    if str(plan.get("kind", "")).startswith("continue_train")]
                        if retrains:
                            best_i, best_plan = max(
                                retrains, key=lambda item: float(item[1]["primary"]))
                            if float(best_plan["primary"]) >= parent_hold - float(
                                    defaults["margin"]):
                                promote_pos = best_i
                    if promote_pos > 0:
                        chosen = recent[promote_pos]
                        choice_note = "recent_holdout 覆盖内核 -> %s" % chosen.get("kind")
                        controller_note = "kernel+recent"
                        print("train: recent_holdout promoted %s (holdout=%d, gate=%s)"
                              % (chosen.get("kind"), holdout_n,
                                 (r_gate[promote_pos].decision if promote_pos < len(r_gate)
                                  and r_gate[promote_pos] is not None else "non-inferior")),
                              file=sys.stderr, flush=True)
                    else:
                        print("train: recent_holdout no promote "
                              "(holdout=%d) -> keep %s"
                              % (holdout_n, chosen.get("kind")),
                              file=sys.stderr, flush=True)
        except Exception as exc:
            print("train: recent_holdout unavailable: %s: %s"
                  % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        # The drift policy needs the candidate list to prefer cumulative-label
        # retrain over the constant pair arm; kernel rounds otherwise have no
        # pipeline plans to inspect.
        policy_plans = list(safety)
        if all(plan is not chosen for plan in policy_plans):
            policy_plans.append(chosen)
    else:
        if probe_retain_applied:
            # A fixed-retain run cannot publish a trained candidate. Evaluate
            # its parent and spend the required gateway call on diagnosis.
            kernel_ctx = _plans.PlanContext.from_rows(
                task_key, model, train_rows, validation_rows,
                metric=metric, direction=direction,
                margin=float(defaults["margin"]),
                classes=list(model.meta.get("classes") or []),
                experiment_cache=experiment_cache)
            plans = [_plans.retain_plan(kernel_ctx)]
        else:
            if kernel_ctx is None:
                kernel_ctx = _plans.PlanContext.from_rows(
                    task_key, model, train_rows, validation_rows, metric=metric,
                    direction=direction, margin=float(defaults["margin"]),
                    classes=list(model.meta.get("classes") or []),
                    experiment_cache=experiment_cache)
            plans = _candidate_plans(task_key, model, train_rows, validation_rows,
                                     metric=metric, direction=direction,
                                     margin=float(defaults["margin"]), ctx=kernel_ctx)
        policy_plans = plans
        table = _format_plan_table(plans, metric)
        prompt = (
            "本轮是 %s 数据集第 %d 轮，指标 %s（方向 %s）。父模型在公开验证集上的 %s 已给出，"
            "候选如下（数字为本机公开验证评估，平台会独立复测）：\n%s\n"
            "请只用一行 JSON 回复：{\"choice\": <候选编号>, \"hypothesis\": \"...\", \"reason\": \"...\"}。"
            "只能选改进父模型的候选；不确定时选 0。"
        ) % (task_key, int(rnd.round_index), metric, direction, table, table)
        if probe_retain_applied:
            prompt = (
                "本轮为 %s 第 %d 轮，配置要求保留当前父模型，跳过不可上线的候选训练。"
                "请根据以下真实公开测量诊断漂移及保留模型的风险，不声称完成训练。"
                "父模型指标=%s；特征漂移=%s；历史记录数=%d。"
                "请用一行 JSON 返回 choice=0、hypothesis 和 reason。"
            ) % (task_key, int(rnd.round_index),
                 json.dumps(plans[0]["values"], ensure_ascii=False),
                 json.dumps(_drift_brief(kernel_ctx), ensure_ascii=False), len(history))
        receipt = _gateway_model(
            rnd,
            [{"role": "user", "content": prompt[:6000]}],
            max_tokens=min(256, int(rnd.max_output_tokens)),
        )
        llm_choice = _parse_choice(receipt.text, len(plans))
        class_labels = list(model.meta.get("classes") or [])
        try:
            if _gate_mode() == "off" or task != "classification":
                gate_decisions = None
                fallback_choice = _choose(plans, direction, defaults["margin"], task_key=task_key)
            else:
                _x_val, y_val, _ = _xy_rows(validation_rows, model, class_labels)
                fallback_choice, gate_decisions = _gate_choose(
                    plans, task, direction, float(defaults["margin"]), y_val, class_labels)
        except Exception as exc:
            print("train: gate unavailable, falling back to point estimate: %s: %s"
                  % (type(exc).__name__, exc), file=sys.stderr, flush=True)
            gate_decisions = None
            fallback_choice = _choose(plans, direction, defaults["margin"], task_key=task_key)
        chosen = plans[fallback_choice]
        if gate_decisions is not None:
            print("train: gate " + ", ".join(
                "%s=%s(%.4f)" % (plans[position].get("kind"), decision.decision,
                                 decision.delta)
                for position, decision in enumerate(gate_decisions) if decision is not None),
                file=sys.stderr, flush=True)
        else:
            print("train: gate off, point-estimate choice=%s"
                  % plans[fallback_choice].get("kind"), file=sys.stderr, flush=True)
        if llm_choice is not None and llm_choice != fallback_choice:
            decision = (gate_decisions[llm_choice]
                        if gate_decisions is not None and llm_choice < len(gate_decisions)
                        else None)
            if decision is not None:
                if decision.decision == "promote":
                    chosen = plans[llm_choice]
            else:
                # Gate intentionally skipped (e.g. regression): fall back to the
                # deterministic margin comparison so the LLM cannot pick a stale arm.
                candidate_primary = float(plans[llm_choice]["primary"])
                parent_primary = float(plans[0]["primary"])
                margin = float(defaults["margin"])
                better = (candidate_primary > parent_primary + margin
                          if direction == "higher"
                          else candidate_primary < parent_primary - margin)
                if better and (task_key != "C" or _c_candidate_acceptable(
                        plans[llm_choice], plans[0], margin)):
                    chosen = plans[llm_choice]
        if gate_decisions is not None:
            decision_note_gate = {}
            for position, decision in enumerate(gate_decisions):
                if decision is None:
                    continue
                decision_note_gate[str(plans[position].get("kind"))] = decision.to_dict()
        else:
            decision_note_gate = {}
        candidates = [{"kind": plan["kind"], "primary": float(plan["primary"]),
                       "metrics": plan.get("values", {}),
                       "experiment_cache": plan.get("experiment_cache"),
                       "alpha": plan.get("alpha"),
                       "group_guard": plan.get("group_guard"),
                       "training_action": ("cache_reused" if plan.get("experiment_cache", {}).get("cache_hit")
                                           else "trained" if plan.get("experiment_cache")
                                           else "not_trained" if plan["kind"] == "retain"
                                           else "not_recorded"),
                       "parameters": {"epochs": plan.get("epochs"),
                                      "learning_rate": plan.get("learning_rate"),
                                      "seed": (plan.get("model") or model).meta.get("seed")}}
                      for plan in plans]
        # Kernel calls are real gateway calls and must stay in the ledger even
        # when the kernel failed to act; the deterministic pipeline then owns the
        # candidate decision (platform trial 0.13.0: every kernel round stopped
        # after get_run_state -> experiments=0, so A never evaluated retrain arms).
        call_ids = [receipt.call_id] + kernel_call_ids
        decision_note = {"decision": "retain" if chosen is plans[0] else "promote",
                         "source": "pipeline",
                         "reason": ("按 C 联合门槛检查 RMSE 改善及 MAE/NASA 不退化"
                                    if task_key == "C" else
                                    ("分块 bootstrap + 模型选择" if gate_decisions is not None
                                               else "公开指标门槛与分组检查 + 模型选择")
                                    if gate_decisions is not None else "主指标改善门槛 + 模型选择"),
                         "gate": decision_note_gate}
        if kernel_inactive_note:
            decision_note["kernel_inactive"] = kernel_inactive_note
        controller_note = "kernel->pipeline" if kernel_inactive_note else "pipeline"
        choice_note = "pipeline 选择 %s" % chosen["kind"]
        if probe_retain_applied:
            controller_note = "probe_retain"
            decision_note.update(
                source="probe_retain",
                reason="配置要求保留父模型；网关诊断：%s" % receipt.text[:1000])
            choice_note = decision_note["reason"]

    policy_note = "probe_retain" if probe_retain_applied else ""
    try:
        policy_ctx = kernel_ctx
        if policy_ctx is None:
            policy_ctx = _plans.PlanContext.from_rows(
                task_key, model, train_rows, validation_rows,
                metric=metric, direction=direction,
                margin=float(defaults["margin"]),
                classes=list(model.meta.get("classes") or []),
                experiment_cache=experiment_cache)
        if not probe_retain_applied:
            chosen, policy_note = _apply_drift_policy(policy_ctx, chosen, policy_plans)
        if policy_note:
            print("train: %s" % policy_note, file=sys.stderr, flush=True)
    except Exception as exc:
        print("train: drift policy unavailable: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr, flush=True)

    # Apply the C check after every route and policy override, including kernel
    # choices. Missing or overflowing NASA evidence cannot authorize promotion.
    if task_key == "C" and str(chosen.get("kind")) != "retain":
        audit_ctx = kernel_ctx or policy_ctx
        parent_plan = plans[0] if plans else _plans.retain_plan(audit_ctx)
        chosen["values"], chosen["primary"], chosen["predicted"] = _plans.evaluate(
            audit_ctx, chosen.get("model") or model, chosen.get("postprocess"))
        chosen["group_guard"] = group_guard(
            parent_plan["predicted"], chosen["predicted"], audit_ctx.y_val,
            [str(row.get("id", "")) for row in validation_rows], enforce_rul=True)
        accepted = _c_candidate_acceptable(chosen, parent_plan, float(defaults["margin"]))
        decision_note["c_metric_gate"] = {
            "minimum_rmse_gain": float(defaults["margin"]), "passed": accepted,
            "parent": parent_plan.get("values", {}), "candidate": chosen.get("values", {}),
            "group_guard": chosen.get("group_guard"),
        }
        if not accepted:
            decision_note.pop("experiment_id", None)
            decision_note.pop("delta", None)
            chosen = parent_plan
            decision_note.update(decision="retain", reason="C 联合指标检查未通过，保留父模型")
            policy_note = "c_metric_guard->retain"
            choice_note = decision_note["reason"]

    if task_key == "A" and str(chosen.get("kind")) != "retain":
        audit_ctx = kernel_ctx or policy_ctx
        safety = _classification_guard(audit_ctx, chosen)
        decision_note["a_final_guard"] = safety
        if not safety["passed"]:
            chosen = _plans.retain_plan(audit_ctx)
            decision_note.pop("experiment_id", None)
            decision_note.update(decision="retain", reason="A 最终类别保持检查未通过")
            choice_note = decision_note["reason"]
            policy_note = "a_final_guard->retain"
    if task_key == "B" and str(chosen.get("kind")) != "retain":
        audit_ctx = kernel_ctx or policy_ctx
        values, chosen["primary"], chosen["predicted"] = _plans.evaluate(
            audit_ctx, chosen.get("model") or model, chosen.get("postprocess"))
        chosen["values"] = values
        parent_values = audit_ctx.parent_values
        passed = all(values.get(key) is not None and np.isfinite(values[key])
                     and values[key] <= parent_values[key] for key in ("rmse", "mae"))
        bearing_report = bearing_guard(
            audit_ctx.parent_predicted, chosen["predicted"], audit_ctx.y_val,
            [str(row.get("id", "")) for row in validation_rows])
        passed = passed and bearing_report["passed"]
        decision_note["b_final_guard"] = {"passed": bool(passed), "group_guard": bearing_report,
                                         "parent": parent_values, "candidate": values}
        if not passed:
            chosen = _plans.retain_plan(audit_ctx)
            decision_note.pop("experiment_id", None)
            decision_note.update(decision="retain", reason="B 总体或轴承分组 RMSE/MAE 保持检查未通过")
            choice_note = decision_note["reason"]
            policy_note = "b_final_guard->retain"

    # The gate also vetoes *any* non-policy arm that a stale point estimate or the
    # LLM might have picked.  A drift-policy override is exempt because it carries
    # its own delayed-label rule (and is the arm validated by E2/E7).
    if policy_note == "" and gate_decisions is not None and plans is not None \
            and chosen is not plans[0]:
        veto = None
        for position, plan in enumerate(plans):
            if plan is chosen:
                veto = gate_decisions[position] if position < len(gate_decisions) else None
                break
        if veto is not None and veto.decision != "promote":
            print("train: gate veto %s (%s) -> retain" % (
                veto.arm or "candidate", veto.reason),
                file=sys.stderr, flush=True)
            chosen = plans[0]
            policy_note = "gate_veto:%s->retain" % (veto.arm or "candidate")

    # Stage-0 safety: keep the official parent artifact for the first rounds so
    # no early scored stage can regress below the platform's static baseline.
    warmup_rounds = _warmup_rounds()
    warmup_applied = False
    if warmup_rounds and int(rnd.round_index) <= warmup_rounds \
            and str(chosen.get("kind") or "") != "retain":
        try:
            retain = _plans.retain_plan(policy_ctx)
            decision_note["warmup"] = {
                "round_index": int(rnd.round_index),
                "warmup_rounds": warmup_rounds,
                "kept": retain.get("kind"),
                "replaced": chosen.get("kind"),
            }
            chosen = retain
            policy_note = (policy_note + "+" if policy_note else "") + "warmup_parent"
            warmup_applied = True
            print("train: warmup round %d/%d -> keep parent artifact"
                  % (int(rnd.round_index), warmup_rounds), file=sys.stderr, flush=True)
        except Exception as exc:
            print("train: warmup unavailable: %s: %s" % (type(exc).__name__, exc),
                  file=sys.stderr, flush=True)

    candidate_dir = Path(tempfile.mkdtemp(prefix="candidate-", dir=work))
    checkpoint_model = chosen.get("model") or model
    if (chosen.get("checkpoint") or chosen.get("checkpoint_changed")
            or str(chosen.get("kind", "")).startswith("continue_train")):
        checkpoint_model.save(candidate_dir / CHECKPOINT_NAME)
    else:
        shutil.copyfile(parent_path, candidate_dir / CHECKPOINT_NAME)
    postprocess = chosen.get("postprocess") or {"type": "none"}
    if postprocess == "FIT_DAMAGE":
        postprocess = {"type": "none"}
    resource = {
        "cpu_seconds": round(max(0.0, time.process_time() - cpu_start), 3),
        "wall_seconds": round(max(0.0, time.time() - wall_start), 3),
        "train_rows": int(probe.get("train_rows") or 0),
        "validation_rows": int(probe.get("validation_rows") or 0),
        "gateway_calls": len(call_ids),
    }
    meta = {
        "harness": "zhixun-rsi-harness",
        "version": __version__,
        "build_identity": _build_identity(),
        "task_key": task_key,
        "task": task,
        "metric": metric,
        "classes": list(model.meta.get("classes") or []),
        "postprocess": postprocess,
        "round_index": int(rnd.round_index),
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "chosen": chosen.get("kind"),
        "controller": controller_note,
        "policy": policy_note[:200],
        "warmup_applied": warmup_applied,
        "warmup_rounds": warmup_rounds,
        "probe_retain": sorted(probe_retain),
        "probe_retain_applied": probe_retain_applied,
        "resource": resource,
        "candidate_checkpoint_sha256": None,  # filled after the file is written
        "data_boundary": {
            "fit_projection": "public/train.jsonl",
            "fit_rows": 0 if probe_retain_applied else int(probe.get("train_rows") or 0),
            "fit_id_sha256": None if probe_retain_applied else probe.get("train_id_sha256"),
            "validation_projection": "public/validation.jsonl",
            "validation_rows": int(probe.get("validation_rows") or 0),
            "validation_id_sha256": probe.get("validation_id_sha256"),
            "validation_used_for_fit": False,
        },
    }
    meta["candidate_checkpoint_sha256"] = _file_sha256(candidate_dir / CHECKPOINT_NAME)
    (candidate_dir / META_NAME).write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    (candidate_dir / BOUNDARY_NAME).write_text(
        json.dumps(meta["data_boundary"], sort_keys=True), encoding="utf-8")
    state = _state_payload(rnd, candidates, chosen,
                           (decision_note.get("reason") or choice_note or "")[:1200],
                           probe=probe, resource=resource)
    (state_dir / "state.json").write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    (state_dir / "data_probe.json").write_text(
        json.dumps(probe, sort_keys=True), encoding="utf-8")
    if kernel_round is not None:
        (state_dir / "kernel_round.json").write_text(
            json.dumps(_trim_config(kernel_round.to_dict()), ensure_ascii=False, default=str),
            encoding="utf-8")
    model_usage = {"calls": len(call_ids), "call_ids": list(call_ids[:16])}
    if receipt is not None:
        model_usage.update(_receipt_usage(receipt))
    if kernel_round is not None and isinstance(kernel_round.usage, dict):
        usage = kernel_round.usage
        model_usage["kernel"] = {
            "calls": int(usage.get("calls") or kernel_round.calls or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("used_input") or 0),
            "completion_tokens": int(usage.get("completion_tokens")
                                     or usage.get("used_output") or 0),
        }
    configuration = _trim_config({
        "controller": controller_note,
        "model_usage": model_usage,
        "build_identity": meta["build_identity"],
        "experiment_cache": chosen.get("experiment_cache"),
        "c_update_alpha": chosen.get("alpha"),
        "a_final_guard": decision_note.get("a_final_guard"),
        "b_final_guard": decision_note.get("b_final_guard"),
        "code_changes": ["rsi/platform/submission.py", "rsi/platform/plans.py", "rsi/metrics.py"],
        "candidate_checkpoint_sha256": meta["candidate_checkpoint_sha256"],
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "warmup": {"applied": warmup_applied, "rounds": warmup_rounds},
        "probe_retain": {"tasks": sorted(probe_retain),
                         "applied": probe_retain_applied},
        "resource": resource,
        "candidates": candidates[:8],
        "chosen": chosen.get("kind"),
        "postprocess": postprocess,
        "policy": policy_note[:200],
        "decision": {key: decision_note.get(key) for key in
                     ("decision", "experiment_id", "reason", "source", "delta")},
        "c_metric_gate": decision_note.get("c_metric_gate"),
        "kernel": None if kernel_round is None else {
            "calls": kernel_round.calls,
            "call_ids": kernel_call_ids[:8],
            "experiments": len(kernel_round.host.results) if kernel_round.host else 0,
            "effective": kernel_effective,
            "error": kernel_round.error,
            "trace": kernel_round.trace[-12:],
        },
        "llm_choice": llm_choice,
        "fallback_choice": fallback_choice,
        "data_probe": {key: probe.get(key) for key in
                       ("train_rows", "duplicate_rows", "validation_rows",
                        "train_content_sha256", "validation_content_sha256")},
    })
    # This local journal is part of native state; it makes no external upload.
    journal = {"version": __version__, "dataset": task_key, "round": int(rnd.round_index),
               "configuration": configuration, "selected_metrics": chosen.get("values"),
               "hypothesis": chosen.get("hypothesis") or chosen.get("label"),
               "change_summary": _change_summary(chosen)}
    with (state_dir / "round_records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(journal), ensure_ascii=False, allow_nan=False) + "\n")
    label = str(chosen.get("label") or chosen.get("kind") or "retain")
    primary_value = chosen.get("primary")
    primary_text = ("%.6f" % float(primary_value)) if primary_value is not None else "n/a"

    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis="本轮候选：%s" % label,
        change_summary=_change_summary(chosen),
        configuration=configuration,
        seed=_seed(),
        checkpoint_paths=[CHECKPOINT_NAME],
        code_paths=[META_NAME, BOUNDARY_NAME],
        predict_argv=PREDICT_ARGV,
    )
    rnd.save_state(state_dir)
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis="本轮候选：%s" % label,
        change_summary=_change_summary(chosen),
        reason="%s（%s）：决定 %s，%s=%s；宿主网关调用 %d 次。" % (
            controller_note, choice_note, chosen.get("kind"), metric, primary_text,
            len(call_ids)),
        configuration=configuration,
        seed=_seed(),
        history_ids=[item["id"] for item in history],
        model_call_ids=call_ids,
    )


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _artifact_path(model_dir) -> Path:
    raw = os.environ.get("CONTEST_ARTIFACT_PATH")
    if raw:
        return Path(raw)
    base = Path(model_dir or "/model")
    for name in (CHECKPOINT_NAME, "model.npz"):
        if (base / name).is_file():
            return base / name
    found = sorted(base.glob("*.npz"))
    if not found:
        raise FileNotFoundError(f"no checkpoint found under {base}")
    return found[0]


def _load_artifact(model_dir) -> dict:
    path = _artifact_path(model_dir)
    key = str(path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    model = ExtendedModel.from_checkpoint(path)
    meta = {}
    meta_path = path.parent / META_NAME
    if not meta_path.is_file() and os.environ.get("CONTEST_ARTIFACT_MANIFEST"):
        try:
            manifest = json.loads(Path(os.environ["CONTEST_ARTIFACT_MANIFEST"]).read_text(encoding="utf-8"))
            if isinstance(manifest, dict) and isinstance(manifest.get("submission_meta"), dict):
                meta = manifest["submission_meta"]
        except (OSError, ValueError):
            meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    postprocess = meta.get("postprocess") or {"type": "none"}
    task = meta.get("task") or model.meta.get("task")
    classes = list(meta.get("classes") or model.meta.get("classes") or [])
    if task not in TASKS:
        raise ValueError(f"unknown task in artifact: {task}")
    payload = {
        "model": model,
        "task": task,
        "classes": classes,
        "postprocess": postprocess,
        "stream": None,
    }
    kind = str(postprocess.get("type", "none"))
    if kind in ("recenter", "pair_threshold", "pair_stream"):
        payload["stream"] = PostprocessStream(postprocess, model)
    elif kind == "damage_shrink":
        payload["stream"] = DamageShrinkStream(postprocess)
    _CACHE.clear()
    _CACHE[key] = payload
    return payload


def _uniform_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        values = np.full(len(values), 1.0 / max(len(values), 1))
    else:
        values = values / total
    top = int(np.argmax(values))
    tied = np.flatnonzero(values == values[top])
    if len(tied) > 1:
        # The host rejects a non-unique top score; break ties deterministically
        # without leaving [0, 1] or altering the argmax class.
        values = values.copy()
        if values[top] <= 0.0:
            values[top] = 1e-7
        else:
            for offset, index in enumerate((int(i) for i in tied if int(i) != top), start=1):
                values[index] = max(0.0, float(values[top]) - 1e-7 * offset)
    return values


def _fallback_prediction(artifact: dict):
    """Never let one malformed row kill the whole prediction stream."""
    if artifact["task"] == "classification":
        classes = list(artifact["classes"])
        if not classes:
            raise ValueError("classification artifact has no classes")
        values = _uniform_scores(np.ones(len(classes), dtype=np.float64))
        return [{"label": label, "score": float(values[position])}
                for position, label in enumerate(classes)]
    mean = float(artifact["model"].meta.get("target_mean") or 0.0)
    if not np.isfinite(mean) or mean < 0:
        mean = 0.0
    return {"value": float(mean)}


def _degenerate_prediction(sample):
    """Last-resort shape-valid output when even the artifact cannot be loaded."""
    if (sample.task or "") == "regression":
        return {"value": 0.0}
    return [{"label": "normal", "score": 1.0}]


def _coerce_task_output(requested: str, artifact: dict, output):
    """Never break the prediction stream when the frozen request task and the
    artifact task disagree (host contract bug observed on dataset B)."""
    artifact_task = str(artifact.get("task") or "")
    requested = requested or artifact_task
    if not requested or requested == artifact_task:
        return output
    if requested == "regression":
        if isinstance(output, dict) and "value" in output:
            return {"value": float(output["value"])}
        best_index, best_score = 0, -1.0
        for position, item in enumerate(output or []):
            score = float(item.get("score", 0.0))
            if score > best_score:
                best_index, best_score = position, score
        return {"value": float(max(best_index, 0))}
    # requested classification while the artifact is regression: bucket the
    # regression value into two label buckets with a unique top score.
    value = 0.0
    if isinstance(output, dict):
        try:
            value = float(output.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
    labels = [str(label) for label in (artifact.get("classes") or []) if str(label)]
    if len(labels) < 2:
        labels = ["normal", "fault"]
    top = 0 if value <= 0.0 else 1
    scores = [0.1 / (len(labels) - 1)] * len(labels)
    scores[top] = 0.9
    return [{"label": labels[position], "score": float(scores[position])}
            for position in range(len(labels))]


def predict(sample, model_dir):
    """One validated prediction line; keep it allocation-light and streaming."""
    try:
        artifact = _load_artifact(model_dir)
    except Exception as error:  # a broken mount must not kill the whole stream
        print("predict artifact load failed: %s: %s" % (type(error).__name__, error),
              file=sys.stderr, flush=True)
        return _degenerate_prediction(sample)
    try:
        output = _predict_one(sample, artifact)
    except Exception as error:  # one bad row must not fail the hidden stream
        print("predict fallback: %s: %s" % (type(error).__name__, error),
              file=sys.stderr, flush=True)
        try:
            output = _fallback_prediction(artifact)
        except Exception:
            return _degenerate_prediction(sample)
    return _coerce_task_output(sample.task or "", artifact, output)


def _predict_one(sample, artifact: dict):
    model: ExtendedModel = artifact["model"]
    features = _flatten_row(_load_specs(model), sample.inputs or {})
    row = features.reshape(1, -1)
    postprocess = artifact.get("postprocess") or {"type": "none"}
    kind = str(postprocess.get("type", "none"))
    raw = model.predict(row)
    if artifact["task"] == "classification":
        stream = artifact.get("stream")
        if stream is not None and hasattr(stream, "predict_row"):
            scores = np.asarray(stream.predict_row(features), dtype=np.float64).reshape(-1)
        else:
            scores = np.asarray(raw, dtype=np.float64).reshape(-1)
        values = _uniform_scores(scores)
        return [
            {"label": label, "score": float(values[position])}
            for position, label in enumerate(artifact["classes"])
        ]
    value = float(np.asarray(raw).reshape(-1)[0])
    if kind == "shrink":
        alpha = float(postprocess.get("alpha", 0.3))
        center = float(postprocess.get("center", model.meta.get("target_mean", 0.0)))
        value = alpha * value + (1.0 - alpha) * center
    elif kind == "damage_shrink" and artifact.get("stream") is not None:
        value = artifact["stream"].transform(features, value)
    if not np.isfinite(value) or value < 0:
        value = max(0.0, float(model.meta.get("target_mean", 0.0)) if not np.isfinite(value) else 0.0)
    return {"value": float(value)}
