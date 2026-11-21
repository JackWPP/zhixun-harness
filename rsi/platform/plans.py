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
    target_mean: float = 0.0

    @classmethod
    def from_rows(cls, task_key: str, model: ExtendedModel, train_rows: list[dict],
                  val_rows: list[dict], *, metric: str, direction: str, margin: float,
                  classes: list[str] | None = None) -> "PlanContext":
        task = str(model.meta.get("task") or "classification")
        classes = [str(c) for c in (classes if classes is not None else model.meta.get("classes") or [])]
        x_train, y_train = xy_rows(train_rows, model, classes)
        x_val, y_val = xy_rows(val_rows, model, classes)
        ctx = cls(task_key=task_key, task=task, metric=metric, direction=direction,
                  margin=float(margin), model=model, classes=classes,
                  train_rows=train_rows, val_rows=val_rows,
                  x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                  target_mean=float(model.meta.get("target_mean") or 0.0))
        ctx.parent_values, ctx.parent_primary, _ = evaluate(ctx, model, {"type": "none"})
        return ctx


def evaluate(ctx: PlanContext, model: ExtendedModel, postprocess: dict | None):
    predicted = apply_postprocess(postprocess or {"type": "none"}, model, ctx.x_val)
    values = from_predictions(ctx.task, predicted, ctx.y_val, ctx.classes)
    primary = values.get("primary")
    if primary is None:
        primary = primary_metric(ctx.task, values)
    return values, float(primary), predicted


def _make_plan(ctx: PlanContext, *, kind: str, label: str, strategy: str,
               params: dict, postprocess: dict, model: ExtendedModel,
               hypothesis: str = "", checkpoint_changed: bool = False) -> dict:
    values, primary, predicted = evaluate(ctx, model, postprocess)
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
    return _make_plan(ctx, kind="retain", label="保留父模型（不做任何改动）",
                      strategy="retain", params={}, postprocess={"type": "none"},
                      model=ctx.model, hypothesis="保留父模型")


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


def _candidate_model(ctx: PlanContext, strategy: str, params: dict) -> tuple[ExtendedModel, bool]:
    """Build the checkpoint for a training strategy (never fits validation)."""
    if strategy == "retain":
        return ctx.model, False
    if strategy in ("continue_train", "continue_train_staged"):
        source = Path(str(ctx.model.meta.get("_source_path", "")))
        if not source.is_file():
            raise ValueError("parent checkpoint path is not available for continuation")
        model = ExtendedModel.from_checkpoint(source)
        y_fit = (np.asarray(ctx.y_train, dtype=np.int64)
                 if ctx.task == "classification" else np.asarray(ctx.y_train, dtype=np.float64))
        epochs = int(params.get("epochs", 20))
        lr = float(params.get("learning_rate", 1e-3))
        weight = None
        beta = float(params.get("class_weight_beta", 0.0) or 0.0)
        if ctx.task == "classification" and beta > 0:
            counts = np.bincount(np.asarray(ctx.y_train, dtype=np.int64),
                                 minlength=max(len(ctx.classes), 1)).astype(np.float64)
            counts[counts == 0] = 1.0
            weight = (len(ctx.y_train) / (len(ctx.classes) * counts)) ** beta
            weight = weight / float(np.mean(weight))
        if strategy == "continue_train_staged":
            model.set_learning_rate(float(params.get("stage1_learning_rate", 1e-3)))
            model.fit(ctx.x_train, y_fit, int(params.get("stage1_epochs", 200)), class_weight=weight)
            polish = int(params.get("stage2_epochs", 0) or 0)
            if polish > 0:
                model.set_learning_rate(float(params.get("stage2_learning_rate", 2e-4)))
                model.fit(ctx.x_train, y_fit, polish, class_weight=weight)
        else:
            model.set_learning_rate(lr)
            if epochs > 0:
                model.fit(ctx.x_train, y_fit, epochs, class_weight=weight)
        return model, True
    raise ValueError(f"strategy {strategy!r} is not executable on the v2 host gateway path")


def plan_for_proposal(ctx: PlanContext, proposal) -> dict:
    """Execute one validated :class:`ExperimentProposal` as a candidate plan."""
    strategy = str(getattr(proposal, "strategy", "retain"))
    params = dict(getattr(proposal, "params", {}) or {})
    raw_post = dict(getattr(proposal, "postprocess", {}) or {"type": "none"})
    hypothesis = str(getattr(proposal, "hypothesis", "") or "")[:400]
    try:
        model, changed = _candidate_model(ctx, strategy, params)
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
        return _make_plan(ctx, kind=kind, label=label, strategy=strategy, params=params,
                          postprocess=postprocess, model=model, hypothesis=hypothesis,
                          checkpoint_changed=changed)
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
        candidates.append(("shrink", "retain", "",
                           {}, {"type": "shrink", "alpha": 0.3, "center": ctx.target_mean},
                           "train 目标均值收缩校准（α=0.3），不改权重"))
        if ctx.task_key == "B":
            candidates.append(("damage_shrink", "retain", "",
                               {}, {"type": "damage_shrink", "a": 0.5, "w": 0.6, "fit": "train"},
                               "damage_shrink 因果分段校准（train 标签拟合）"))
            candidates.append(("continue_train", "continue_train", "",
                               {"learning_rate": 1e-3, "epochs": 15},
                               {"type": "none"},
                               "累积标签续训 15 epoch（用当前 /data/train 全部已揭示行）"))
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
