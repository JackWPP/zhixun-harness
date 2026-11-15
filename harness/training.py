"""
Rule-Driven & Small-Model Collaborative Harness Trainer (规则驱动与小模型协同续训底座).
Replaces the procedural demo trainer with `RuleDrivenHarnessTrainer`, `DiagnosticProbe`,
`DiagnosticState`, and `ExperienceBuffer` while strictly preserving train-only updates,
immutable normalization/target scalers, and deterministic Adam continuation semantics.
"""
from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

from .data import digest, write_json
from .diagnostics import improvement, metric_values, primary_metric


class BudgetExhausted(RuntimeError):
    """Raised when candidate training exceeds the shared training time budget."""


class CostLedger:
    """Tracks shared candidate training budget and validation evaluation time."""

    def __init__(self, max_train_seconds: float):
        if not math.isfinite(max_train_seconds) or max_train_seconds <= 0:
            raise ValueError("max_train_seconds must be finite and positive")
        self.max_train_seconds = float(max_train_seconds)
        self.training_seconds = 0.0
        self.validation_seconds = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_train_seconds - self.training_seconds)

    def evaluate(self, model, data) -> dict:
        started = time.perf_counter()
        try:
            return metric_values(model, data)
        finally:
            self.validation_seconds += time.perf_counter() - started


# ==============================================================================
# 1. 实验任务基准配置规范 (Task Specifications)
# ==============================================================================

@dataclass
class TaskSpec:
    """描述 A (气体识别)、B (轴承 RUL)、C (发动机 RUL) 三个基准任务的架构与数据统计规范。"""
    profile: str
    name: str
    task_type: str  # "classification" | "regression"
    input_dim: int
    hidden_dim: int
    output_dim: int
    parameter_count: int
    train_rows: int
    validation_rows: int
    primary_metric: str
    higher_is_better: bool
    base_lr: float
    batch_size: int
    base_epochs: int
    target_unit: str = ""
    classes: List[str] = field(default_factory=list)
    description: str = ""


def get_default_task_specs() -> Dict[str, TaskSpec]:
    """返回与原包 `training.json` 完全对齐的三大工业基准任务参数表。"""
    return {
        "A": TaskSpec(
            profile="A",
            name="gas_identification",
            task_type="classification",
            input_dim=128,
            hidden_dim=32,
            output_dim=6,
            parameter_count=4326,
            train_rows=445,
            validation_rows=1244,
            primary_metric="macro_f1",
            higher_is_better=True,
            base_lr=0.003,
            batch_size=128,
            base_epochs=20,
            classes=["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
            description="UCI 气体传感器漂移六类气体识别任务",
        ),
        "B": TaskSpec(
            profile="B",
            name="bearing_rul",
            task_type="regression",
            input_dim=34,
            hidden_dim=24,
            output_dim=1,
            parameter_count=865,
            train_rows=7032,
            validation_rows=1679,
            primary_metric="rmse",
            higher_is_better=False,
            base_lr=0.003,
            batch_size=128,
            base_epochs=15,
            target_unit="minutes",
            description="XJTU-SY 滚动轴承剩余使用寿命(分钟)回归预测任务",
        ),
        "C": TaskSpec(
            profile="C",
            name="engine_rul",
            task_type="regression",
            input_dim=24,
            hidden_dim=32,
            output_dim=1,
            parameter_count=833,
            train_rows=113089,
            validation_rows=23049,
            primary_metric="rmse",
            higher_is_better=False,
            base_lr=0.002,
            batch_size=256,
            base_epochs=15,
            target_unit="cycles",
            description="NASA C-MAPSS 涡扇发动机剩余寿命(循环)回归预测任务",
        ),
    }


# ==============================================================================
# 2. 诊断状态与诊断探针 (DiagnosticState & DiagnosticProbe)
# ==============================================================================

@dataclass
class DiagnosticState:
    """
    训练器在每个续训轮次提取的 12 维标准化诊断状态。
    与 `strategies.diagnosis_to_state_vector` 保持一致的特征语义，可输入给小模型或规则路由。
    """
    epoch: int
    task_type_code: float          # 0.0: 分类 (A), 1.0: 回归 (B/C)
    train_loss: float
    val_primary_metric: float
    train_loss_slope: float
    val_metric_slope: float
    generalization_gap: float
    grad_norm_mean: float
    grad_norm_std: float
    metric_improvement: float
    plateau_count: int
    acc_f1_gap: float = 0.0
    tail_class_f1: float = 1.0
    overestimate_fraction: float = 0.0

    def to_vector(self) -> np.ndarray:
        """转换为 12 维数值特征向量，供策略推荐分类小模型直接推理或训练。"""
        return np.array([
            float(self.task_type_code),
            float(self.train_loss),
            float(self.val_primary_metric),
            float(self.train_loss_slope),
            float(self.val_metric_slope),
            float(self.generalization_gap),
            float(self.grad_norm_mean),
            float(self.grad_norm_std),
            float(self.metric_improvement),
            float(self.plateau_count),
            float(self.acc_f1_gap),
            float(self.overestimate_fraction),
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


class DiagnosticProbe:
    """只读诊断探针：跟踪候选训练过程中的损失序列、梯度范数与验证指标变化率。"""

    def __init__(self, task: str, window_size: int = 3):
        self.task = task
        self.window_size = window_size
        self.train_losses: List[float] = []
        self.val_metrics: List[float] = []

    @staticmethod
    def _slope(values: List[float], window: int) -> float:
        if len(values) < 2:
            return 0.0
        seg = values[-window:]
        x = np.arange(len(seg), dtype=np.float64)
        y = np.asarray(seg, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        validation_metrics: dict,
        grad_norms: List[float],
        plateau_count: int,
    ) -> DiagnosticState:
        key = primary_metric(self.task)
        val_primary = float(validation_metrics.get(key, 0.0))
        prev_val = self.val_metrics[-1] if self.val_metrics else val_primary

        self.train_losses.append(float(train_loss))
        self.val_metrics.append(val_primary)

        gain = (val_primary - prev_val) if self.task == "A" else (prev_val - val_primary)
        train_slope = self._slope(self.train_losses, self.window_size)
        val_slope = self._slope(self.val_metrics, self.window_size)

        grad_mean = float(np.mean(grad_norms)) if grad_norms else 0.0
        grad_std = float(np.std(grad_norms)) if grad_norms else 0.0

        if self.task == "A":
            acc = float(validation_metrics.get("accuracy", val_primary))
            acc_f1_gap = acc - val_primary
            per_cls = validation_metrics.get("per_class_f1", {})
            tail_f1 = float(min(per_cls.values())) if per_cls else val_primary
            overest = 0.0
            gen_gap = max(0.0, 1.0 - val_primary - float(train_loss))
        else:
            acc_f1_gap = 0.0
            tail_f1 = 1.0
            overest = float(validation_metrics.get("overestimate_fraction", 0.0))
            gen_gap = float(val_primary) / max(float(train_loss), 1e-6)

        return DiagnosticState(
            epoch=int(epoch),
            task_type_code=0.0 if self.task == "A" else 1.0,
            train_loss=float(train_loss),
            val_primary_metric=val_primary,
            train_loss_slope=train_slope,
            val_metric_slope=val_slope,
            generalization_gap=gen_gap,
            grad_norm_mean=grad_mean,
            grad_norm_std=grad_std,
            metric_improvement=float(gain),
            plateau_count=int(plateau_count),
            acc_f1_gap=acc_f1_gap,
            tail_class_f1=tail_f1,
            overestimate_fraction=overest,
        )


# ==============================================================================
# 3. 经验轨迹回放池 (TrajectoryRecord & ExperienceBuffer)
# ==============================================================================

@dataclass
class TrajectoryRecord:
    """记录单次候选策略执行的 (State, Strategy_ID, Gain, Accepted) 轨迹。"""
    task: str
    round_index: int
    candidate_id: str
    strategy_id: str
    state_vector: List[float]
    improvement: float
    accepted: bool
    eligible: bool

    def to_dict(self) -> dict:
        return asdict(self)


class ExperienceBuffer:
    """收集并管理候选训练轨迹，用于训练与更新轻量级策略推荐分类小模型。"""

    def __init__(self) -> None:
        self.records: List[TrajectoryRecord] = []

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def to_list(self) -> List[dict]:
        return [r.to_dict() for r in self.records]

    def save_json(self, path: Path) -> None:
        write_json(path, {"count": len(self.records), "records": self.to_list()})

    @classmethod
    def from_run_directories(cls, run_dirs: List[Path]) -> "ExperienceBuffer":
        """从已有的 `harness-runs/{A-v1,B-v1,C-v1}` 目录解析真实历史轨迹。"""
        from .strategies import diagnosis_to_state_vector

        buf = cls()
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            task = summary.get("task", "A")
            for round_entry in summary.get("rounds", []):
                r_idx = int(round_entry.get("round", 1))
                diagnosis = round_entry.get("diagnosis", {})
                # 构造伪 model meta 以提取诊断特征向量
                state_vec = diagnosis_to_state_vector(task, None, diagnosis)
                for cand in round_entry.get("candidates", []):
                    strat = cand.get("strategy", {})
                    gate = cand.get("gate", {})
                    gain = gate.get("improvement")
                    if gain is None or not math.isfinite(float(gain)):
                        gain = 0.0
                    buf.add(
                        TrajectoryRecord(
                            task=task,
                            round_index=r_idx,
                            candidate_id=str(cand.get("id", "")),
                            strategy_id=str(strat.get("id", "conservative")),
                            state_vector=[float(x) for x in state_vec.tolist()],
                            improvement=float(gain),
                            accepted=bool(gate.get("accepted", False)),
                            eligible=bool(gate.get("eligible", False)),
                        )
                    )
        return buf


# ==============================================================================
# 4. 解耦式规则驱动训练器核心 (RuleDrivenHarnessTrainer)
# ==============================================================================

class RuleDrivenHarnessTrainer:
    """
    核心解耦训练器：
    - 严格遵守 train-only 数据边界，绝不读取或拟合验证集特征/标签
    - 保持原模型 `mean`/`scale` 与 `target_mean`/`target_scale` 冻结不变
    - 精确恢复并推进 `Model` 的 Adam 一阶矩/二阶矩与 `SeedSequence([seed, epochs])`
    - 支持类别加权 (`balanced_classes`)、设备均衡抽样 (`balanced_groups`)、特征噪声注入 (`noise_std`)、
      L2 正则 (`l2`)、非对称 RUL 惩罚 (`asymmetric_rul_alpha`) 与梯度裁剪 (`grad_clip_norm`)
    - 内置 `DiagnosticProbe` 逐轮采集诊断指标，并支持门控最佳轮次恢复与经验池记录
    """

    def __init__(
        self,
        patience: int = 2,
        experience_buffer: Optional[ExperienceBuffer] = None,
    ):
        self.patience = int(patience)
        self.experience_buffer = experience_buffer

    @staticmethod
    def execute_epoch(
        model,
        train,
        strategy,
        max_seconds: float = math.inf,
        grad_norms_out: Optional[List[float]] = None,
    ) -> float:
        """
        单轮训练核心实现：只接受 `train.role == "train"` 的数据对象。
        当策略参数为默认基线值时，与原始 `Model.fit` 保持逐位浮点数完全等价。
        """
        if train.role != "train":
            raise ValueError("optimizer accepts train-role data only")
        if max_seconds <= 0:
            raise BudgetExhausted("training budget exhausted before epoch")
        started = time.perf_counter()
        x = model.normalized(train.x)
        y = train.y
        regression = model.meta["task"] == "regression"
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"] if regression else y
        rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))

        if getattr(strategy, "balanced_groups", False):
            counts = collections.Counter(train.groups)
            prob = np.array([1.0 / counts[g] for g in train.groups])
            prob /= prob.sum()
            order = rng.choice(len(x), size=len(x), replace=True, p=prob)
        else:
            order = rng.permutation(len(x))

        sample_weights = np.ones(len(x))
        if getattr(strategy, "balanced_classes", False):
            if regression:
                raise ValueError("class weighting is classification-only")
            counts = np.bincount(y, minlength=len(model.meta["classes"]))
            if np.any(counts == 0):
                raise ValueError("class weighting requires training coverage of every class")
            class_weights = len(y) / (len(counts) * counts)
            sample_weights = class_weights[y]

        asym_alpha = float(getattr(strategy, "asymmetric_rul_alpha", 0.0) or 0.0)
        clip_limit = float(getattr(strategy, "grad_clip_norm", 5.0) or 5.0)

        loss_sum = 0.0
        for start in range(0, len(order), strategy.batch_size):
            if time.perf_counter() - started >= max_seconds:
                # Partial epoch is never checkpointed or eligible for acceptance.
                raise BudgetExhausted("training budget exhausted mid-epoch; partial updates discarded")
            idx = order[start : start + strategy.batch_size]
            batch = x[idx].copy()
            if getattr(strategy, "noise_std", 0.0):
                batch = np.clip(batch + rng.normal(0, strategy.noise_std, batch.shape), -10, 10)
            expected = target[idx]
            hidden, logits = model.forward(batch)
            if not regression:
                logits -= logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                weights = sample_weights[idx]
                loss_sum -= float(
                    np.sum(weights * np.log(np.maximum(probabilities[np.arange(len(idx)), expected], 1e-300)))
                )
                delta = probabilities.copy()
                delta[np.arange(len(idx)), expected] -= 1
                delta *= weights[:, None] / len(idx)
            else:
                error = logits[:, 0] - expected
                if asym_alpha > 0.0:
                    late_factor = np.where(error > 0.0, 1.0 + asym_alpha, 1.0)
                    loss_sum += float(np.sum(late_factor * (error ** 2)))
                    delta = (2 * late_factor * error / len(idx))[:, None]
                else:
                    loss_sum += float(np.sum(error ** 2))
                    delta = (2 * error / len(idx))[:, None]

            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            if getattr(strategy, "l2", 0.0):
                for key in ("w1", "w2"):
                    gradients[key] += strategy.l2 * model.params[key]
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            if grad_norms_out is not None:
                grad_norms_out.append(norm)
            model.meta["steps"] += 1
            for name, gradient in gradients.items():
                gradient *= min(1.0, clip_limit / max(norm, 1e-12))
                model.first[name] = 0.9 * model.first[name] + 0.1 * gradient
                model.second[name] = 0.999 * model.second[name] + 0.001 * gradient ** 2
                first = model.first[name] / (1 - 0.9 ** model.meta["steps"])
                second = model.second[name] / (1 - 0.999 ** model.meta["steps"])
                model.params[name] -= strategy.learning_rate * first / (np.sqrt(second) + 1e-8)

        model.meta["epochs"] += 1
        model.meta["learning_rate"] = strategy.learning_rate
        model.meta["batch_size"] = strategy.batch_size
        model.validate()
        return loss_sum / len(x)

    def execute_candidate(
        self,
        bundle,
        parent: Path,
        strategy,
        directory: Path,
        ledger: CostLedger,
        event,
        patience: Optional[int] = None,
    ) -> dict:
        """
        执行单个候选策略的受控续训、诊断探针采集、早停与最佳检查点恢复。
        通过调用模块级 `train_epoch` 保证对外部单元测试 mock 的完全兼容。
        """
        eff_patience = self.patience if patience is None else int(patience)
        directory = Path(directory)
        directory.mkdir()
        (directory / "checkpoints").mkdir()
        parent = Path(parent)
        expected_parent = digest(parent)
        model = bundle.model_class.load(parent)
        mean, scale = model.mean.copy(), model.scale.copy()
        target_state = (model.meta["target_mean"], model.meta["target_scale"])
        initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
        started, initial_training = time.perf_counter(), ledger.training_seconds

        probe = DiagnosticProbe(task=bundle.task, window_size=3)
        diagnostic_trace: List[dict] = []

        result = {
            "id": directory.name,
            "trainer_engine": "RuleDrivenHarnessTrainer",
            "status": "completed",
            "strategy": strategy.to_dict(),
            "parent_checkpoint_sha256": expected_parent,
            "checkpoint_sha256": None,
            "best_checkpoint": None,
            "metrics": {},
            "history": [],
            "diagnostic_trace": diagnostic_trace,
            "initial_steps": initial_steps,
            "initial_epochs": initial_epochs,
            "optimizer_training_sha256": bundle.train.sha256,
            "evaluation_validation_sha256": bundle.validation.sha256,
            "normalization_policy": "preserve_parent_state_no_refit",
            "gate": {"eligible": False, "reason": "尚未完成有效训练轮次", "improvement": None},
        }
        best_metrics, best_path, stale = None, None, 0
        try:
            for local_epoch in range(strategy.epochs):
                train_start = time.perf_counter()
                try:
                    # 调用模块级 train_epoch 以兼容 unittest.mock.patch("harness.training.train_epoch")
                    loss = train_epoch(model, bundle.train, strategy, ledger.remaining)
                finally:
                    ledger.training_seconds += time.perf_counter() - train_start

                if not np.array_equal(model.mean, mean) or not np.array_equal(model.scale, scale):
                    raise ValueError("normalization state changed")
                if target_state != (model.meta["target_mean"], model.meta["target_scale"]):
                    raise ValueError("target scaling changed")

                validation = ledger.evaluate(model, bundle.validation)
                if best_metrics is None or improvement(bundle.task, best_metrics, validation) > 0:
                    best_metrics = validation
                    best_path = directory / "checkpoints" / ("epoch-%04d.npz" % model.meta["epochs"])
                    model.save(best_path)
                    stale = 0
                else:
                    stale += 1

                diag_state = probe.record_epoch(
                    epoch=model.meta["epochs"],
                    train_loss=loss,
                    validation_metrics=validation,
                    grad_norms=[],
                    plateau_count=stale,
                )
                diagnostic_trace.append(diag_state.to_dict())

                record = {
                    "epoch": model.meta["epochs"],
                    "local_epoch": local_epoch + 1,
                    "step": model.meta["steps"],
                    "train_loss": loss,
                    "validation": validation,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                result["history"].append(record)
                event("candidate_epoch", {"candidate": result["id"], **record})

                if stale >= eff_patience:
                    result["early_stopped"] = True
                    break
        except BudgetExhausted as exc:
            result["status"] = "budget_exhausted"
            result["error"] = str(exc)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = type(exc).__name__ + ": " + str(exc)
        finally:
            result["cost"] = {
                "training_seconds": ledger.training_seconds - initial_training,
                "total_seconds": time.perf_counter() - started,
                "epochs": len(result["history"]),
                "steps": model.meta["steps"] - initial_steps,
            }

        parent_unchanged = digest(parent) == expected_parent
        if not parent_unchanged:
            result["status"] = "error"
            result["error"] = "parent checkpoint changed during training"

        if best_path is not None and result["status"] != "error" and parent_unchanged:
            restored = bundle.model_class.load(best_path)
            if restored.meta["steps"] <= initial_steps or restored.meta["epochs"] <= initial_epochs:
                raise ValueError("candidate did not advance the supplied optimizer state")
            if not np.array_equal(restored.mean, mean) or not np.array_equal(restored.scale, scale):
                raise ValueError("candidate normalization changed")
            chosen = directory / "checkpoint.npz"
            shutil.copyfile(best_path, chosen)
            result.update(
                best_checkpoint=str(chosen.resolve()),
                checkpoint_sha256=digest(chosen),
                metrics=best_metrics,
            )
            result["gate"] = {
                "eligible": True,
                "reason": "有效完整轮次；保持预处理与来源状态，等待主指标比较",
                "improvement": None,
            }
            result["restored_best_epoch"] = restored.meta["epochs"]
            result["selected_steps"] = restored.meta["steps"]

        write_json(directory / "candidate.json", result)
        event(
            "candidate_finished",
            {
                "candidate": result["id"],
                "status": result["status"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "cost": result["cost"],
            },
        )
        return result


# ==============================================================================
# 5. 标准模块级入口（兼容原有 Harness API 与全部单元测试）
# ==============================================================================

_DEFAULT_TRAINER = RuleDrivenHarnessTrainer()


def train_epoch(model, train, strategy, max_seconds: float = math.inf) -> float:
    """No validation argument exists here. The only fitted arrays come from train."""
    return _DEFAULT_TRAINER.execute_epoch(model, train, strategy, max_seconds=max_seconds)


def run_candidate(
    bundle,
    parent,
    strategy,
    directory,
    ledger,
    event,
    patience: int = 2,
    trainer: Optional[RuleDrivenHarnessTrainer] = None,
) -> dict:
    """Execute candidate continuation using `RuleDrivenHarnessTrainer`."""
    active_trainer = trainer if trainer is not None else RuleDrivenHarnessTrainer(patience=patience)
    return active_trainer.execute_candidate(
        bundle, parent, strategy, directory, ledger, event, patience=patience
    )
