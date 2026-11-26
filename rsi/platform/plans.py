"""Candidate-plan construction for the workspace-v2 participant.

This module is shared by the deterministic ``pipeline`` route and by the
PI-agent ``kernel`` route.  A *plan* is a fully evaluated candidate:

``{"ok", "experiment_id", "kind", "label", "strategy", "params",
  "postprocess", "model", "primary", "values", "metrics", "hypothesis",
  "checkpoint_changed"}``

All fitting happens on the public training projection only; the public
validation projection is used for selection/reporting and never fitted.  Nothing
here touches hidden labels or any network service.
"""
from __future__ import annotations

import json
import copy
import hashlib
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rsi.inference import (
    POSTPROCESS_KINDS,
    apply_postprocess,
    fit_damage_shrink,
    resolve_postprocess,
)
from rsi.metrics import from_predictions, primary as primary_metric
from rsi.model import ExtendedModel
from rsi.platform.experiment_cache import ExperimentCache

TASKS = ("classification", "regression")


# ---------------------------------------------------------------------------
# workspace helpers (also used by submission.py / agent_host.py)
# ---------------------------------------------------------------------------
def work_dir() -> Path:
    configured = os.environ.get("CONTEST_WORK")
    if configured:
        path = Path(configured)
    elif Path("/workspace").is_dir():
        path = Path("/workspace")
    else:
        path = Path(tempfile.mkdtemp(prefix="contest-workspace-"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_state_dir(work: Path, rnd) -> Path:
    """Copy the previous native state into a fresh writable directory.

    Workspace-v2 recommends ``rnd.state_workspace``; the first round may omit
    ``/state``, in which case an empty directory is a valid starting point.
    """
    destination = work / ("state-round-%d-%d" % (int(rnd.round_index), os.getpid()))
    try:
        return rnd.state_workspace(str(destination))
    except Exception:  # first round may omit /state; fall back to an empty dir
        destination.mkdir(parents=True, exist_ok=True)
        return destination


# ---------------------------------------------------------------------------
# row handling
# ---------------------------------------------------------------------------
def load_specs(model: ExtendedModel) -> list[list[list[str]]]:
    raw = model.meta.get("paths") or []
    specs = []
    for item in raw:
        spec = json.loads(item) if isinstance(item, str) else item
        specs.append([list(entry) for entry in spec])
    return specs


def _ordered_leaves(value) -> list[float]:
    """Flatten an arbitrary input tree into floats, dict keys sorted.

    Used only when the checkpoint path spec no longer matches a private data
    schema (renamed keys / extra wrapper).  Sorted keys preserve the natural
    column order of ``sensor_0000 … sensor_0127``-style schemas.
    """
    if isinstance(value, dict):
        leaves: list[float] = []
        for key in sorted(value):
            leaves.extend(_ordered_leaves(value[key]))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_ordered_leaves(item))
        return leaves
    return [float(value)]


def flatten_row(specs: list[list[list[str]]], inputs: dict, *,
                positional_fallback: bool = True) -> np.ndarray:
    values = []
    try:
        for spec in specs:
            current = inputs
            for kind, argument in spec:
                if kind in ("key", "index"):
                    current = current[argument]
                else:  # pragma: no cover - malformed checkpoint metadata
                    raise ValueError(f"unsupported path spec: {spec}")
            values.append(float(current))
    except (KeyError, IndexError, TypeError, ValueError):
        if not positional_fallback:
            raise
        leaves = _ordered_leaves(inputs)
        if len(leaves) < len(specs):
            raise ValueError(
                "input schema does not expose %d feature values" % len(specs))
        values = leaves[: len(specs)]
    return np.asarray(values, dtype=np.float64)


def xy_rows(rows: list[dict], model: ExtendedModel, classes: list[str]):
    specs = load_specs(model)
    features = np.empty((len(rows), len(specs)), dtype=np.float64)
    labels = np.empty(len(rows), dtype=np.float64)
    index = {name: position for position, name in enumerate(classes)}
    task = model.meta.get("task")
    for position, row in enumerate(rows):
        features[position] = flatten_row(specs, row["inputs"])
        answer = row.get("answer")
        if task == "classification":
            if answer not in index:
                raise ValueError(f"unknown class label: {answer!r}")
            labels[position] = index[answer]
        else:
            labels[position] = float(answer)
    return features, labels


def dataset_from_rows(x: np.ndarray, y: np.ndarray, classes, task: str):
    from rsi.data import Dataset

    return Dataset(name="official-train", ids=[str(i) for i in range(len(x))], x=x, y=y,
                   paths=[], sha256="", groups=["all"] * len(x))


# ---------------------------------------------------------------------------
# plan context
# ---------------------------------------------------------------------------
def _validated_parent_postprocess(spec, model: ExtendedModel, depth: int = 0) -> dict:
    """Validate an already-resolved artifact without refitting or rounding it.

    ``resolve_postprocess`` is intentionally not used: it can replace shrink's
    stored center or round thresholds, changing the actual parent's behavior.
    """
    if depth > 8:
        raise ValueError("parent postprocess nesting is too deep")
    if spec is None or spec == {}:
        return {"type": "none"}
    if not isinstance(spec, dict) or spec.get("type") not in POSTPROCESS_KINDS:
        raise ValueError("invalid parent postprocess declaration")
    result = copy.deepcopy(spec)
    kind = spec["type"]

    def number(value, name, lower=None, upper=None):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or (lower is not None and value < lower)
                or (upper is not None and value > upper)):
            raise ValueError("invalid parent postprocess " + name)

    def positive_int(value, name):
        if type(value) is not int or value < 1:
            raise ValueError("invalid parent postprocess " + name)

    def vector(value, name, length=None):
        if not isinstance(value, list) or (length is not None and len(value) != length):
            raise ValueError("invalid parent postprocess " + name)
        for item in value:
            number(item, name)

    if kind == "none":
        return result
    if kind == "recenter":
        if spec.get("mode", "batch_mean") not in ("batch_mean", "batch_std"):
            raise ValueError("invalid parent postprocess recenter mode")
        number(spec.get("weight", 1.), "weight", 0., 1.)
    elif kind == "pair_threshold":
        if model.meta["task"] != "classification":
            raise ValueError("parent pair threshold requires classification")
        pair = spec.get("pair")
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(index) is not int or not 0 <= index < len(model.meta["classes"]) for index in pair)
                or pair[0] == pair[1]):
            raise ValueError("invalid parent postprocess pair shape or indices")
        number(spec.get("delta"), "delta")
        result["base"] = _validated_parent_postprocess(spec.get("base"), model, depth+1)
    elif kind in ("shrink", "damage_shrink"):
        if model.meta["task"] != "regression":
            raise ValueError("parent shrink requires regression")
        number(spec.get("center"), "center")
        if kind == "shrink":
            number(spec.get("alpha"), "alpha", 0., 1.)
            return result
        width = len(model.meta["paths"])
        dims = spec.get("dims", list((0, 17)))
        if (not isinstance(dims, list) or not dims
                or any(type(index) is not int or not 0 <= index < width for index in dims)
                or (len(dims) < 2 and width <= 17)):
            raise ValueError("invalid parent postprocess damage dimensions")
        vector(spec.get("robust_scale"), "robust_scale", width)
        if any(value <= 0 for value in spec["robust_scale"]):
            raise ValueError("invalid parent postprocess robust_scale")
        curve = spec.get("curve")
        if not isinstance(curve, dict):
            raise ValueError("invalid parent postprocess curve")
        vector(curve.get("edges"), "curve.edges")
        vector(curve.get("values"), "curve.values", len(curve["edges"])+1)
        if any(right <= left for left, right in zip(curve["edges"], curve["edges"][1:])):
            raise ValueError("parent postprocess curve edges must be strictly increasing")
        for name, default in (("a", .5), ("w", .6), ("fallback_alpha", .3)):
            number(spec.get(name, default), name, 0., 1.)
        positive_int(spec.get("early", 10), "early")
        detector = spec.get("detector")
        if detector is not None:
            if not isinstance(detector, dict):
                raise ValueError("invalid parent postprocess detector")
            for name, default in (("rel", .15), ("ratio", .1)):
                number(detector.get(name, default), "detector."+name, 0., 1.)
            for name, default in (("min_rows", 8), ("window", 20)):
                positive_int(detector.get(name, default), "detector."+name)
    return result


def _parent_postprocess(model: ExtendedModel, task_key: str) -> dict:
    """Load a digest-bound parent's predictor spec; missing initial metadata is none."""
    source = model.meta.get("_source_path")
    if not source:
        return {"type": "none"}
    checkpoint = Path(source)
    metadata_path = checkpoint.parent / "submission_meta.json"
    if metadata_path.is_symlink():
        raise ValueError("parent artifact metadata must be a regular file")
    if not metadata_path.exists():
        return {"type": "none"}
    if not metadata_path.is_file() or metadata_path.stat().st_size > 128*1024:
        raise ValueError("parent artifact metadata is not a bounded regular file")
    def reject_constant(value):
        raise ValueError("nonfinite parent artifact metadata: " + value)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid parent artifact metadata") from exc
    if not isinstance(metadata, dict):
        raise ValueError("parent artifact metadata must be an object")
    declared = metadata.get("candidate_checkpoint_sha256")
    if isinstance(declared, str) and declared.startswith("sha256:"):
        declared = declared[7:]
    if (not isinstance(declared, str) or len(declared) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in declared)):
        raise ValueError("parent artifact metadata requires a checkpoint SHA-256")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("parent checkpoint must be a regular file")
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if declared.lower() != actual:
        raise ValueError("parent artifact checkpoint SHA-256 mismatch")
    if metadata.get("task", model.meta["task"]) != model.meta["task"]:
        raise ValueError("parent artifact task does not match checkpoint")
    if metadata.get("task_key", task_key) != task_key:
        raise ValueError("parent artifact dataset does not match current task")
    if metadata.get("classes", model.meta["classes"]) != model.meta["classes"]:
        raise ValueError("parent artifact classes do not match checkpoint")
    return _validated_parent_postprocess(metadata.get("postprocess"), model)


@dataclass
class PlanContext:
    task_key: str
    task: str
    metric: str
    direction: str
    margin: float
    model: ExtendedModel
    classes: list[str]
    train_rows: list[dict]
    val_rows: list[dict]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    parent_values: dict = field(default_factory=dict)
    parent_primary: float = 0.0
    parent_predicted: np.ndarray | None = None
    parent_postprocess: dict = field(default_factory=lambda: {"type": "none"})
    target_mean: float = 0.0
    experiment_cache: ExperimentCache = field(default_factory=ExperimentCache)

    @classmethod
    def from_rows(cls, task_key: str, model: ExtendedModel, train_rows: list[dict],
                  val_rows: list[dict], *, metric: str, direction: str, margin: float,
                  classes: list[str] | None = None,
                  experiment_cache: ExperimentCache | None = None) -> "PlanContext":
        task = str(model.meta.get("task") or "classification")
        classes = [str(c) for c in (classes if classes is not None else model.meta.get("classes") or [])]
        x_train, y_train = xy_rows(train_rows, model, classes)
        x_val, y_val = xy_rows(val_rows, model, classes)
        ctx = cls(task_key=task_key, task=task, metric=metric, direction=direction,
                  margin=float(margin), model=model, classes=classes,
                  train_rows=train_rows, val_rows=val_rows,
                  x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                  parent_postprocess=_parent_postprocess(model, task_key),
                  target_mean=float(model.meta.get("target_mean") or 0.0),
                  experiment_cache=(experiment_cache if experiment_cache is not None
                                    else ExperimentCache()))
        ctx.parent_values, ctx.parent_primary, ctx.parent_predicted = evaluate(
            ctx, model, ctx.parent_postprocess)
        return ctx


def evaluate(ctx: PlanContext, model: ExtendedModel, postprocess: dict | None):
    predicted = apply_postprocess(postprocess or {"type": "none"}, model, ctx.x_val)
    values = from_predictions(ctx.task, predicted, ctx.y_val, ctx.classes, profile=ctx.task_key)
    primary = values.get("primary")
    if primary is None:
        primary = primary_metric(ctx.task, values)
    return values, float(primary), predicted


def _make_plan(ctx: PlanContext, *, kind: str, label: str, strategy: str,
               params: dict, postprocess: dict, model: ExtendedModel,
               hypothesis: str = "", checkpoint_changed: bool = False,
               evaluated: tuple | None = None) -> dict:
    values, primary, predicted = (evaluated if evaluated is not None
                                  else evaluate(ctx, model, postprocess))
    return {
        "ok": True,
        "predicted": predicted,
        "kind": kind,
        "label": label,
        "strategy": strategy,
        "params": dict(params or {}),
        "postprocess": postprocess,
        "model": model,
        "primary": float(primary),
        "values": values,
        "metrics": {"selection": {"primary": float(primary)}, "report": values},
        "hypothesis": hypothesis or label,
        "checkpoint_changed": bool(checkpoint_changed),
    }


def retain_plan(ctx: PlanContext) -> dict:
    evaluated = ((dict(ctx.parent_values), ctx.parent_primary, ctx.parent_predicted)
                 if ctx.parent_predicted is not None else None)
    return _make_plan(ctx, kind="retain", label="保留父模型（不做任何改动）",
                      strategy="retain", params={}, postprocess=copy.deepcopy(ctx.parent_postprocess),
                      model=ctx.model, hypothesis="保留父模型", evaluated=evaluated)


# ---------------------------------------------------------------------------
# drift profile + delayed-label prior (gen2 policy)
# ---------------------------------------------------------------------------
#: Prior delta for the Acetaldehyde/Ethanol pair flip, fitted in earlier
#: campaigns from *past* labels (never the current public validation labels).
PAIR_HISTORY_DELTA = float(os.environ.get("RSI_V2_PAIR_DELTA", "-4.0"))


def drift_profile(ctx: PlanContext) -> dict:
    """Covariate-drift summary between the public train and validation splits.

    Only feature means are compared, so no labels and no hidden data are used.
    ``level`` drives the policy hint handed to the kernel agent.
    """
    try:
        train_mean = np.asarray(ctx.x_train.mean(axis=0), dtype=np.float64)
        train_std = np.asarray(ctx.x_train.std(axis=0), dtype=np.float64) + 1e-9
        val_mean = np.asarray(ctx.x_val.mean(axis=0), dtype=np.float64)
        z = (val_mean - train_mean) / train_std
        order = np.argsort(-np.abs(z))[:5]
        top = [[int(index), round(float(z[index]), 3)] for index in order]
        score = float(np.abs(z).mean())
        peak = float(np.abs(z).max())
    except Exception:
        return {"score": 0.0, "peak": 0.0, "level": "low", "top": []}
    if score >= 0.5 or peak >= 5.0:
        level = "high"
    elif score >= 0.15 or peak >= 1.5:
        level = "medium"
    else:
        level = "low"
    return {"score": round(score, 4), "peak": round(peak, 3), "level": level, "top": top}


def pair_history_plan(ctx: PlanContext) -> dict:
    """Classification-only delayed-label arm using a *historical* pair delta.

    The delta comes from previous campaigns (labels observed after the drift
    event), not from the current public validation split.  This is the arm that
    gen2 selected for task A; it is intentionally evaluated like every other
    candidate, but is *not* required to beat the parent on the public validation
    split (neutral validation is acceptable when the drift level is high).
    """
    if ctx.task != "classification":
        return {"ok": False, "kind": "pair_history", "label": "delayed-label 先验（仅分类）",
                "error": "pair_history is classification-only"}
    plan = plan_for_proposal(ctx, _Proposal(
        "retain", {}, {"type": "pair_threshold", "pair": [0, 3],
                       "delta": PAIR_HISTORY_DELTA},
        "历史标签校准的延迟标签阈值（pair=[0,3], delta=%.1f），不拟合当前验证集"
        % PAIR_HISTORY_DELTA))
    plan["kind"] = "pair_history"
    plan["label"] = "延迟标签先验 pair_threshold(delta=%.1f)" % PAIR_HISTORY_DELTA
    return plan


def _resolve_postprocess(ctx: PlanContext, model: ExtendedModel, spec) -> dict:
    """Resolve postprocess specs.

    Compliance: shrink/damage may use training labels; pair_threshold must carry
    a fixed pre-declared delta and is never fitted from labels at train time.
    """
    if not isinstance(spec, dict):
        return {"type": "none"}
    kind = str(spec.get("type", "none"))
    if kind == "none":
        return {"type": "none"}
    if kind == "pair_threshold":
        pair = spec.get("pair", (0, 3))
        # Compliance: never fit the decision threshold from labels at train
        # time.  Only a fixed, pre-declared historical delta is allowed.
        if spec.get("delta") is None:
            return {"type": "none"}
        return {"type": "pair_threshold", "pair": [int(pair[0]), int(pair[1])],
                "delta": float(spec["delta"])}
    if kind == "damage_shrink":
        dataset = dataset_from_rows(ctx.x_train, ctx.y_train, ctx.classes, ctx.task)
        fitted = fit_damage_shrink(dataset, spec)
        return fitted if fitted else {"type": "none"}
    if kind == "shrink":
        return {"type": "shrink", "alpha": float(spec.get("alpha", 0.5)),
                "center": float(spec.get("center", ctx.target_mean))}
    if kind == "recenter":
        return resolve_postprocess(spec, model)
    return resolve_postprocess(spec, model)


def cached_continuation(ctx: PlanContext, params: dict, *,
                        strategy: str = "continue_train") -> tuple[ExtendedModel, dict]:
    """Continue the actual parent, reusing only identical training computations.

    Share ``ctx.experiment_cache`` with any holdout/alternative context explicitly.
    Validation and postprocessing are excluded from the training key: callers
    must evaluate the returned model against their current validation projection.
    An explicit ``seed`` overrides the parent seed; otherwise it is preserved.
    """
    if strategy not in {"continue_train", "continue_train_staged"}:
        raise ValueError(f"unsupported continuation strategy: {strategy!r}")
    source = Path(str(ctx.model.meta.get("_source_path", "")))
    if not source.is_file():
        raise ValueError("parent checkpoint path is not available for continuation")
    # Canonicalize defaults so equivalent A variants use the same training key.
    config = {"strategy": strategy, "seed": int(params.get("seed", ctx.model.meta.get("seed", 42))),
              "class_weight_beta": float(params.get("class_weight_beta", 0.0) or 0.0)}
    if strategy == "continue_train_staged":
        config.update(stage1_epochs=int(params.get("stage1_epochs", 200)),
                      stage1_learning_rate=float(params.get("stage1_learning_rate", 1e-3)),
                      stage2_epochs=int(params.get("stage2_epochs", 0) or 0),
                      stage2_learning_rate=float(params.get("stage2_learning_rate", 2e-4)))
    else:
        config.update(epochs=int(params.get("epochs", 20)),
                      learning_rate=float(params.get("learning_rate", 1e-3)))
    y_fit = (np.asarray(ctx.y_train, dtype=np.int64) if ctx.task == "classification"
             else np.asarray(ctx.y_train, dtype=np.float64))

    def fit():
        model = ExtendedModel.from_checkpoint(source)
        model.meta["seed"] = config["seed"]
        weight = None
        beta = config["class_weight_beta"]
        if ctx.task == "classification" and beta > 0:
            counts = np.bincount(np.asarray(ctx.y_train, dtype=np.int64),
                                 minlength=max(len(ctx.classes), 1)).astype(np.float64)
            counts[counts == 0] = 1.0
            weight = (len(ctx.y_train) / (len(ctx.classes) * counts)) ** beta
            weight = weight / float(np.mean(weight))
        if strategy == "continue_train_staged":
            model.set_learning_rate(config["stage1_learning_rate"])
            model.fit(ctx.x_train, y_fit, config["stage1_epochs"], class_weight=weight)
            polish = config["stage2_epochs"]
            if polish > 0:
                model.set_learning_rate(config["stage2_learning_rate"])
                model.fit(ctx.x_train, y_fit, polish, class_weight=weight)
        else:
            model.set_learning_rate(config["learning_rate"])
            if config["epochs"] > 0:
                model.fit(ctx.x_train, y_fit, config["epochs"], class_weight=weight)
        return model

    return ctx.experiment_cache.get_or_fit(
        parent_path=source, task_key=ctx.task_key, x_train=ctx.x_train, y_train=y_fit,
        configuration=config, fit=fit)


def _candidate_model(ctx: PlanContext, strategy: str, params: dict):
    """Build the checkpoint for a training strategy (never fits validation)."""
    if strategy == "retain":
        return ctx.model, False, {}
    if strategy in ("continue_train", "continue_train_staged"):
        model, cache = cached_continuation(ctx, params, strategy=strategy)
        return model, True, cache
    raise ValueError(f"strategy {strategy!r} is not executable on the v2 host gateway path")


def plan_for_proposal(ctx: PlanContext, proposal) -> dict:
    """Execute one validated :class:`ExperimentProposal` as a candidate plan."""
    strategy = str(getattr(proposal, "strategy", "retain"))
    params = dict(getattr(proposal, "params", {}) or {})
    raw_post = dict(getattr(proposal, "postprocess", {}) or {"type": "none"})
    hypothesis = str(getattr(proposal, "hypothesis", "") or "")[:400]
    try:
        if strategy == "retain" and raw_post.get("type", "none") == "none":
            plan = retain_plan(ctx)
            plan["hypothesis"] = hypothesis or plan["hypothesis"]
            return plan
        model, changed, cache = _candidate_model(ctx, strategy, params)
        postprocess = _resolve_postprocess(ctx, model, raw_post)
        if strategy == "continue_train":
            kind = ("continue_train_pair"
                    if str(postprocess.get("type")) == "pair_threshold"
                    else "continue_train")
        else:
            kind = strategy if strategy != "retain" else str(postprocess.get("type", "retain"))
        if strategy == "retain" and kind == "none":
            kind = "retain"
        label = "%s + %s" % (strategy, str(postprocess.get("type", "none")))
        if strategy == "retain":
            label = "推理期后处理 %s（不改权重）" % str(postprocess.get("type", "none"))
            if postprocess.get("type", "none") == "none":
                label = "保留父模型（不做任何改动）"
        plan = _make_plan(ctx, kind=kind, label=label, strategy=strategy, params=params,
                          postprocess=postprocess, model=model, hypothesis=hypothesis,
                          checkpoint_changed=changed)
        if cache:
            plan["experiment_cache"] = cache
        return plan
    except Exception as error:  # a failed proposal must not kill the round
        return {"ok": False, "kind": strategy, "strategy": strategy, "params": params,
                "postprocess": raw_post, "model": None, "primary": None, "values": {},
                "metrics": {}, "hypothesis": hypothesis,
                "error": "%s: %s" % (type(error).__name__, error),
                "checkpoint_changed": False}


def build_plans(ctx: PlanContext) -> list[dict]:
    """Deterministic seed catalogue used when the agent is unavailable."""
    plans = [retain_plan(ctx)]
    candidates: list[tuple[str, str, str, dict, dict, str]] = []
    if ctx.task == "classification":
        for weight, mode in ((0.4, "batch_mean"), (0.2, "batch_std")):
            candidates.append(("recenter", "retain", "",
                               {}, {"type": "recenter", "mode": mode, "weight": weight},
                               f"推理期 recenter（{mode}, w={weight}），不改权重"))
        if ctx.task_key == "A":
            candidates.append(("pair_history", "retain", "",
                               {}, {"type": "pair_threshold", "pair": [0, 3],
                                    "delta": PAIR_HISTORY_DELTA},
                               "历史校准 pair 阈值（delta=%.3f，不重新拟合）" % PAIR_HISTORY_DELTA))
            candidates.append(("continue_train", "continue_train", "",
                               {"learning_rate": 1e-3, "epochs": 20},
                               {"type": "none"},
                               "累积标签续训 20 epoch（用当前 /data/train 全部已揭示行）"))
            candidates.append(("continue_train_pair", "continue_train", "",
                               {"learning_rate": 1e-3, "epochs": 20},
                               {"type": "pair_threshold", "pair": [0, 3],
                                "delta": PAIR_HISTORY_DELTA},
                               "累积标签续训 + 历史校准 pair 阈值"))
    else:
        # B: the static shrink/damage arms over-fit the dominant validation
        # bearing and collapsed on its hidden tail (platform trial 0.14.5:
        # hidden RMSE 735 vs 720 for retain).  They stay behind an explicit
        # switch so the arm catalogue matches the platform candidate list.
        static_b = ctx.task_key != "B" or b_static_mode()
        if static_b:
            candidates.append(("shrink", "retain", "",
                               {}, {"type": "shrink", "alpha": 0.3, "center": ctx.target_mean},
                               "train 目标均值收缩校准（α=0.3），不改权重"))
        if ctx.task_key == "B":
            if b_static_mode():
                candidates.append(("damage_shrink", "retain", "",
                                   {}, {"type": "damage_shrink", "a": 0.5, "w": 0.6, "fit": "train"},
                                   "damage_shrink 因果分段校准（train 标签拟合）"))
            b_params = b_continuation_parameters()
            candidates.append(("continue_train", "continue_train", "",
                               b_params,
                               {"type": "none"},
                               "父检查点续训 %d epoch（仅公开 train）" % b_params["epochs"]))
        if ctx.task_key == "C":
            candidates.append(("continue_train", "continue_train", "",
                               {"learning_rate": 1e-3, "epochs": 40},
                               {"type": "none"},
                               "官方 train 续训 40 epoch（lr=1e-3，checkpoint 兼容）"))
    for kind, strategy, _unused, params, postprocess, label in candidates:
        plan = plan_for_proposal(ctx, _Proposal(strategy, params, postprocess, label))
        plan["kind"] = kind
        plan["label"] = label
        plan["seed_plan"] = True
        plans.append(plan)
    if ctx.task == "classification":
        plans.extend(capacity_plans(ctx))
    elif ctx.task == "regression":
        plans.extend(scratch_plans(ctx))
    return plans


@dataclass
class _Proposal:
    strategy: str
    params: dict
    postprocess: dict
    hypothesis: str = ""


# ---------------------------------------------------------------------------
# opt-in capacity arms (loss / hidden-layer changes, official v1 compatible)
# ---------------------------------------------------------------------------
CAPACITY_ARM_ENV = "RSI_V2_CAPACITY_ARM"
_CAPACITY_ARMS = {
    "h64": {"name": "capacity_h64", "hidden": 64, "loss": None, "epochs": 120,
            "label": "容量臂：单隐层 tanh h64（从零重训，checkpoint 为官方 v1 格式）"},
    "ls10": {"name": "capacity_ls10", "hidden": 32,
             "loss": {"type": "label_smoothing", "eps": 0.1}, "epochs": 120,
             "label": "容量臂：label smoothing eps=0.1（单隐层 tanh h32，官方 v1 兼容）"},
}


def capacity_arm_mode() -> str:
    return os.environ.get(CAPACITY_ARM_ENV, "off").strip().lower()


def _capacity_arm_list() -> list[dict]:
    mode = capacity_arm_mode()
    if mode in ("", "off", "none", "0", "false"):
        return []
    names = ["h64", "ls10"] if mode in ("both", "all", "on") else [mode]
    return [_CAPACITY_ARMS[name] for name in names if name in _CAPACITY_ARMS]


def capacity_plans(ctx: PlanContext) -> list[dict]:
    """Train opt-in capacity arms on the public train projection only.

    Every arm is converted back to the official ``contest.cpu-mlp.v1`` layout so
    the platform loader accepts it.  Only single-hidden tanh arms are allowed:
    ReLU and two-layer models are not loadable by the official forward pass.
    """
    arms = _capacity_arm_list()
    if not arms or ctx.task != "classification":
        return []
    from rsi.capacity import CapacityModel, normalize_loss, to_official

    seed = int(os.environ.get("RSI_V2_SEED", "20261117"))
    y = np.asarray(ctx.y_train, dtype=np.int64)
    plans = []
    for arm in arms:
        try:
            model = CapacityModel(
                ctx.task, list(ctx.model.meta.get("paths") or []), ctx.classes,
                ctx.x_train, y, hidden=int(arm["hidden"]), hidden2=0,
                activation="tanh", dropout=0.0,
                loss=normalize_loss(arm["loss"]) if arm.get("loss") else None,
                seed=seed, learning_rate=3e-3, batch_size=64)
            model.fit(ctx.x_train, y, int(arm["epochs"]))
            official = to_official(model)
            plan = _make_plan(ctx, kind=arm["name"], label=arm["label"],
                              strategy="capacity",
                              params={"hidden": int(arm["hidden"]),
                                      "loss": arm.get("loss"), "epochs": int(arm["epochs"])},
                              postprocess={"type": "none"}, model=official,
                              hypothesis=arm["label"], checkpoint_changed=True)
            plan["seed_plan"] = True
            plans.append(plan)
        except Exception as error:  # a broken capacity arm must not kill the round
            plans.append({"ok": False, "kind": arm["name"], "strategy": "capacity",
                          "primary": None, "values": {}, "metrics": {},
                          "error": "%s: %s" % (type(error).__name__, error),
                          "checkpoint_changed": False})
    return plans


def catalog_text(include_strategies: bool = True) -> str:
    """Compact catalogue shown to the kernel agent."""
    lines = []
    if include_strategies:
        from rsi.strategies import STRATEGIES
        lines.append("策略（strategy + params）：")
        for name in ("retain", "continue_train", "continue_train_staged"):
            spec = STRATEGIES.get(name)
            if spec is None:
                continue
            lines.append("- %s: %s" % (name, spec.description[:160]))
    lines.append("后处理（postprocess.type）：")
    for kind in ("none", "recenter", "pair_threshold", "shrink", "damage_shrink"):
        if kind not in POSTPROCESS_KINDS:
            continue
        lines.append("- %s" % kind)
    lines.append("漂移策略（gen2）：高漂移分类任务优先 pair_threshold(delta=-4.0，"
                 "历史标签先验) 且只要公开验证不劣于父模型 2*margin 就采用；"
                 "中漂移只用 recenter；低漂移 retain。")
    lines.append("约束：data 只能是 train；recenter 参数 mode∈{batch_mean,batch_std}, "
                 "weight∈[0,1]；pair_threshold 用 pair=[0,3] 并只从 train 标签拟合 delta；"
                 "shrink alpha∈[0,1] 仅回归；damage_shrink a,w∈[0,1] 仅回归。")
    return "\n".join(lines)


B_STATIC_ENV = "RSI_V2_B_STATIC"


def b_continuation_parameters() -> dict:
    """Bounded B continuation preserves parent architecture and optimizer state."""
    epochs = int(os.environ.get("RSI_V2_B_EPOCHS", "15"))
    lr = float(os.environ.get("RSI_V2_B_LR", "0.001"))
    if not 1 <= epochs <= 60 or not np.isfinite(lr) or not 0 < lr <= 0.01:
        raise ValueError("B continuation requires 1..60 epochs and learning rate in (0, 0.01]")
    return {"epochs": epochs, "learning_rate": lr}


def b_continuation_plan(ctx: PlanContext) -> dict:
    """One inexpensive B candidate based on this round's supplied parent."""
    if ctx.task_key != "B" or ctx.task != "regression":
        raise ValueError("B continuation requires a B regression context")
    parameters = b_continuation_parameters()
    return plan_for_proposal(ctx, _Proposal(
        "continue_train", parameters, {"type": "none"},
        "从本轮父检查点续训 %d epoch，检验新增公开训练数据是否改善预测" % parameters["epochs"]))


def b_static_mode() -> bool:
    """Whether B keeps the legacy static shrink/damage arms (default: off)."""
    value = (os.environ.get(B_STATIC_ENV, "0") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# from-scratch retrain arm (B/C regression, public train labels only)
# ---------------------------------------------------------------------------
SCRATCH_ARM_ENV = "RSI_V2_SCRATCH_ARM"
_SCRATCH_CONFIG: dict[str, dict] = {
    # Chosen from local validation sweeps: B benefits from a wider single
    # hidden layer (h80/e600: val RMSE 474 vs 636 at h48, baseline 681); C
    # saturates at 32 units, so keep it cheap for the 2-core sandbox.
    "B": {"hidden": 80, "epochs": 600, "learning_rate": 1e-3, "seed": 17},
    "C": {"hidden": 32, "epochs": 300, "learning_rate": 1e-3, "seed": 17},
}


def scratch_arm_mode() -> str:
    """Legacy research-only retraining; disabled unless explicitly requested.

    Enabling an official-format scratch model does not establish compliance with
    the platform's parent-continuation requirement. Production defaults stay off.
    """
    return (os.environ.get(SCRATCH_ARM_ENV, "off") or "off").strip().lower()


def _scratch_enabled(task_key: str) -> bool:
    mode = scratch_arm_mode()
    if mode in {"off", "0", "false", "no", "none"}:
        return False
    if mode in {"on", "1", "true", "yes", "all", "auto"}:
        return task_key in _SCRATCH_CONFIG
    return task_key.upper() in {part.strip().upper() for part in mode.split(",") if part.strip()}


def _model_to_official(model) -> "ExtendedModel":
    """Wrap a freshly trained ``Model`` into the official v1 checkpoint format."""
    from rsi.model import FORMAT, ExtendedModel

    task = str(model.meta.get("task"))
    hidden = int(model.meta.get("hidden") or 32)
    classes = list(model.meta.get("classes") or [])
    params = {
        name: np.array(model.params[name], dtype=np.float64, copy=True)
        for name in ("w1", "b1", "w2", "b2")
    }
    count = int(sum(int(value.size) for value in params.values()))
    meta = {
        "format": FORMAT,
        "device": "cpu",
        "task": task,
        "paths": list(model.meta.get("paths") or []),
        "classes": classes,
        "hidden": hidden,
        "seed": int(model.meta.get("seed", 42)),
        "epochs": int(model.meta.get("epochs", 0)),
        "steps": int(model.meta.get("steps", 0)),
        "learning_rate": float(model.meta.get("learning_rate", 3e-3)),
        "batch_size": int(model.meta.get("batch_size", 128)),
        "parameter_count": count,
        "target_mean": float(model.meta.get("target_mean", 0.0)),
        "target_scale": float(model.meta.get("target_scale", 1.0)),
    }
    official = ExtendedModel.__new__(ExtendedModel)
    official.params = params
    official.first = {name: np.zeros_like(value) for name, value in params.items()}
    official.second = {name: np.zeros_like(value) for name, value in params.items()}
    official.mean = np.array(model.mean, dtype=np.float64, copy=True)
    official.scale = np.array(model.scale, dtype=np.float64, copy=True)
    official.meta = meta
    official.validate()
    return official


def scratch_plans(ctx: PlanContext) -> list[dict]:
    """Train a fresh official-format model on the public training projection.

    Selection still happens on the public validation split; no validation row
    ever enters a gradient update.  The arm is only offered for regression
    tasks whose checkpoint is a single-hidden-layer tanh MLP.
    """
    if ctx.task != "regression" or not _scratch_enabled(ctx.task_key):
        return []
    config = _SCRATCH_CONFIG.get(ctx.task_key)
    if not config:
        return []
    try:
        from rsi.model import Model

        x_train = np.asarray(ctx.x_train, dtype=np.float64)
        y_train = np.asarray(ctx.y_train, dtype=np.float64)
        model = Model(
            "regression", list(ctx.model.meta.get("paths") or []), [],
            x_train, y_train, hidden=int(config["hidden"]),
            seed=int(config["seed"]), learning_rate=float(config["learning_rate"]),
        )
        model.fit(x_train, y_train, epochs=int(config["epochs"]))
        official = _model_to_official(model)
        plan = _make_plan(
            ctx, kind="scratch_retrain",
            label="从零重训 h%d %dep（仅公开 train 标签，官方 v1 checkpoint）"
                  % (int(config["hidden"]), int(config["epochs"])),
            strategy="continue_train",
            params={"hidden": int(config["hidden"]),
                    "epochs": int(config["epochs"]), "from_scratch": True},
            postprocess={"type": "none"}, model=official,
            hypothesis="父模型在公开 train 上欠拟合；从零重训单隐层 tanh 可提升跨轴承/机组泛化",
            checkpoint_changed=True,
        )
        plan["seed_plan"] = True
        return [plan]
    except Exception as error:  # a broken arm must never kill the round
        return [{"ok": False, "kind": "scratch_retrain", "strategy": "continue_train",
                 "primary": None, "values": {}, "metrics": {},
                 "error": "%s: %s" % (type(error).__name__, error),
                 "checkpoint_changed": False}]
