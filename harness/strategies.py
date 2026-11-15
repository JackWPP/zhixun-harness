"""
Rule-Driven Candidate Router (`rule_select_candidates`) & Small-Model Strategy Recommender
(`StrategyRecommenderMLP`).
Diagnosis chooses actions based on statistical evidence or lightweight classifier inference;
never fits parameters on validation data.
"""
from __future__ import annotations

import collections
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

import numpy as np


STRATEGY_IDS: List[str] = [
    "conservative",
    "class_weighted",
    "robust_noise",
    "device_balanced",
    "regularized",
    "penalize_late_rul",
]


@dataclass(frozen=True)
class Strategy:
    id: str
    label: str
    reason: str
    learning_rate: float
    batch_size: int
    epochs: int
    l2: float = 0.0
    noise_std: float = 0.0
    balanced_classes: bool = False
    balanced_groups: bool = False
    asymmetric_rul_alpha: float = 0.0
    grad_clip_norm: float = 5.0

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.asymmetric_rul_alpha == 0.0:
            data.pop("asymmetric_rul_alpha", None)
        if self.grad_clip_norm == 5.0:
            data.pop("grad_clip_norm", None)
        return data


def diagnosis_to_state_vector(task: str, model, diagnosis: dict) -> np.ndarray:
    """
    将 `diagnose()` 产生的诊断字典转换为 12 维标准化特征向量 (`DiagnosticState` 向量)：
    [task_type_code, train_primary, val_primary, rel_gap, feature_shift_median,
     feature_shift_max, train_groups, val_groups, acc_f1_gap, weak_class_count,
     overestimate_fraction, worse_than_constant_flag]
    """
    t_metrics = diagnosis.get("train_metrics", {})
    v_metrics = diagnosis.get("validation_metrics", {})
    details = diagnosis.get("details", {})
    f_shift = details.get("feature_shift", {})
    top_shifts = f_shift.get("top", [])
    max_shift = float(top_shifts[0]["standardized_mean_difference"]) if top_shifts else 0.0
    med_shift = float(f_shift.get("median", 0.0))

    codes = {f.get("code") for f in diagnosis.get("findings", [])}
    train_groups = float(details.get("train_group_count", 1))
    val_groups = float(details.get("validation_group_count", 1))

    if task == "A":
        t_f1 = float(t_metrics.get("macro_f1", 0.0))
        v_f1 = float(v_metrics.get("macro_f1", 0.0))
        v_acc = float(v_metrics.get("accuracy", v_f1))
        rel_gap = max(0.0, t_f1 - v_f1)
        acc_f1_gap = max(0.0, v_acc - v_f1)
        per_cls = v_metrics.get("per_class_f1", {})
        weak_count = float(sum(1 for score in per_cls.values() if float(score) < 0.5))
        overest = 0.0
        worse_const = 0.0
        t_val, v_val = t_f1, v_f1
    else:
        t_rmse = float(t_metrics.get("rmse", 0.0))
        v_rmse = float(v_metrics.get("rmse", 0.0))
        rel_gap = (v_rmse - t_rmse) / max(t_rmse, 1.0)
        acc_f1_gap = 0.0
        weak_count = 0.0
        overest = float(v_metrics.get("overestimate_fraction", 0.0))
        worse_const = 1.0 if "worse_than_constant" in codes else 0.0
        t_val = float(np.log1p(max(0.0, t_rmse)))
        v_val = float(np.log1p(max(0.0, v_rmse)))

    return np.array([
        0.0 if task == "A" else (0.5 if task == "B" else 1.0),
        t_val,
        v_val,
        rel_gap,
        med_shift,
        max_shift,
        train_groups,
        val_groups,
        acc_f1_gap,
        weak_count,
        overest,
        worse_const,
    ], dtype=np.float64)


def build_strategy_by_id(
    strategy_id: str,
    task: str,
    low_lr: float,
    batch_size: int,
    epochs: int,
    reason_override: Optional[str] = None,
) -> Strategy:
    """根据策略 ID 构建标准化的 `Strategy` 实例。"""
    if strategy_id == "conservative":
        return Strategy(
            "conservative",
            "低学习率续训",
            reason_override or "安全对照：保留全部状态，仅降低学习率并选最佳轮。",
            low_lr,
            batch_size,
            epochs,
        )
    if strategy_id == "class_weighted":
        return Strategy(
            "class_weighted",
            "训练类别加权＋L2",
            reason_override or "训练类不均衡或弱类混淆；权重只由训练标签频率计算。",
            low_lr,
            batch_size,
            epochs,
            l2=0.001,
            balanced_classes=True,
        )
    if strategy_id == "robust_noise":
        if task == "A":
            return Strategy(
                "robust_noise",
                "训练扰动＋L2",
                reason_override or "存在输入差异或泛化差距；在训练归一化特征上注入固定小幅噪声。",
                low_lr,
                batch_size,
                epochs,
                l2=0.003,
                noise_std=0.04,
                balanced_classes=True,
            )
        return Strategy(
            "robust_noise",
            "训练扰动＋L2",
            reason_override or "对工况泛化做训练侧扰动对照；不在验证侧拟合归一化。",
            low_lr,
            batch_size,
            epochs,
            l2=0.001,
            noise_std=0.02,
        )
    if strategy_id == "device_balanced":
        return Strategy(
            "device_balanced",
            "训练设备均衡抽样＋L2",
            reason_override or "跨设备泛化不足；每根训练设备总采样概率相同，不改变预测输入。",
            low_lr,
            batch_size,
            epochs,
            l2=0.002,
            balanced_groups=True,
        )
    if strategy_id == "penalize_late_rul":
        return Strategy(
            "penalize_late_rul",
            "非对称滞后惩罚＋L2",
            reason_override or "回归高估比例偏高；在训练侧对高估误差施加非对称惩罚项。",
            low_lr,
            batch_size,
            epochs,
            l2=0.002,
            asymmetric_rul_alpha=0.25,
        )
    return Strategy(
        "regularized",
        "L2 正则续训",
        reason_override or "鲁棒性对照：保持原结构和输入，抑制权重过度适配。",
        low_lr,
        batch_size,
        epochs,
        l2=0.003,
    )


def rule_select_candidates(
    task: str,
    model,
    diagnosis: dict,
    epochs: int = 5,
    max_candidates: int = 3,
) -> List[Strategy]:
    """
    核心规则路由函数 (`rule_select_candidates`)：
    解析 `diagnose()` 生成的统计证据与异常代码，映射为确定性的候选策略列表。
    """
    if not 1 <= epochs <= 1000 or not 1 <= max_candidates <= 4:
        raise ValueError("epochs in 1..1000; max_candidates in 1..4")
    codes = {f["code"] for f in diagnosis["findings"]}
    lr = float(model.meta["learning_rate"])
    batch = int(model.meta["batch_size"])
    low = max(lr * 0.3, 1e-6)

    strategies: List[Strategy] = [
        build_strategy_by_id("conservative", task, low, batch, epochs)
    ]
    if task == "A":
        if "train_class_imbalance" in codes or "weak_classes" in codes:
            strategies.append(build_strategy_by_id("class_weighted", task, low, batch, epochs))
        if "observable_input_shift" in codes or "generalization_gap" in codes:
            strategies.append(build_strategy_by_id("robust_noise", task, low, batch, epochs))
    else:
        if "generalization_gap" in codes or "worse_than_constant" in codes:
            strategies.append(build_strategy_by_id("device_balanced", task, low, batch, epochs))
        strategies.append(build_strategy_by_id("regularized", task, low, batch, epochs))
        if task == "C":
            strategies.append(build_strategy_by_id("robust_noise", task, low, batch, epochs))
        v_metrics = diagnosis.get("validation_metrics", {})
        if float(v_metrics.get("overestimate_fraction", 0.0)) > 0.50:
            strategies.append(build_strategy_by_id("penalize_late_rul", task, low, batch, epochs))

    if len(strategies) < max_candidates and not any(s.id == "regularized" for s in strategies):
        strategies.append(
            Strategy("regularized", "L2 正则续训", "通用低成本候选，不改变模型结构。", low, batch, epochs, l2=0.003)
        )
    return strategies[:max_candidates]


# ==============================================================================
# 轻量级策略推荐分类小模型 (StrategyRecommenderMLP)
# ==============================================================================

class StrategyRecommenderMLP:
    """
    基于纯 NumPy 实现的轻量级策略推荐分类小模型：
    - 输入：`diagnosis_to_state_vector` 提取的 12 维诊断状态向量
    - 输出：`STRATEGY_IDS` 候选动作的概率排序（带任务兼容性掩码）
    - 训练数据：由 `ExperienceBuffer` 从历史运行记录汇总的 `(state_vector, strategy_id, improvement)`
    """

    def __init__(self, state_dim: int = 12, hidden_dim: int = 24, seed: int = 42):
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = len(STRATEGY_IDS)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / self.state_dim), (self.state_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, self.num_actions))
        self.b2 = np.zeros(self.num_actions, dtype=np.float64)

        self.mean = np.zeros(self.state_dim, dtype=np.float64)
        self.scale = np.ones(self.state_dim, dtype=np.float64)
        self.trained_samples = 0

    def _forward(self, x_norm: np.ndarray):
        z1 = x_norm @ self.w1 + self.b1
        h1 = np.maximum(0.0, z1)
        logits = h1 @ self.w2 + self.b2
        return z1, h1, logits

    def fit_from_buffer(self, buffer, epochs: int = 150, lr: float = 0.02) -> Dict[str, float]:
        """
        利用 `ExperienceBuffer` 中的历史候选轨迹拟合策略推荐分类器。
        对同一诊断状态下的多个候选（如同一轮的 3 个候选策略），提取胜出/最优增益策略作为监督标签。
        """
        grouped = collections.defaultdict(list)
        total_eligible = 0
        for rec in buffer.records:
            if not rec.eligible or rec.strategy_id not in STRATEGY_IDS:
                continue
            total_eligible += 1
            key = (rec.task, rec.round_index, tuple(np.round(rec.state_vector, 6)))
            grouped[key].append(rec)

        if not grouped:
            raise ValueError("experience buffer has no eligible records to train StrategyRecommenderMLP")

        xs, ys = [], []
        for _, recs in grouped.items():
            accepted_recs = [r for r in recs if r.accepted]
            if accepted_recs:
                winner = accepted_recs[0]
            else:
                best_rec = max(recs, key=lambda r: r.improvement)
                if best_rec.improvement > 0:
                    winner = best_rec
                else:
                    cons = [r for r in recs if r.strategy_id == "conservative"]
                    winner = cons[0] if cons else best_rec
            xs.append(np.asarray(winner.state_vector, dtype=np.float64))
            ys.append(STRATEGY_IDS.index(winner.strategy_id))

        x_mat = np.stack(xs, axis=0)
        y_vec = np.asarray(ys, dtype=np.int64)

        self.mean = x_mat.mean(axis=0)
        self.scale = x_mat.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        x_norm = np.clip((x_mat - self.mean) / self.scale, -10.0, 10.0)

        m_w1, v_w1 = np.zeros_like(self.w1), np.zeros_like(self.w1)
        m_b1, v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        m_w2, v_w2 = np.zeros_like(self.w2), np.zeros_like(self.w2)
        m_b2, v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        final_loss = 0.0
        n = len(y_vec)
        for step in range(1, epochs + 1):
            z1, h1, logits = self._forward(x_norm)
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            loss = -float(np.mean(np.log(np.maximum(probs[np.arange(n), y_vec], 1e-12))))
            final_loss = loss

            d_logits = probs.copy()
            d_logits[np.arange(n), y_vec] -= 1.0
            d_logits /= n

            gw2 = h1.T @ d_logits + 1e-4 * self.w2
            gb2 = d_logits.sum(axis=0)
            dh1 = (d_logits @ self.w2.T) * (z1 > 0.0)
            gw1 = x_norm.T @ dh1 + 1e-4 * self.w1
            gb1 = dh1.sum(axis=0)

            for param, grad, m, v in (
                (self.w1, gw1, m_w1, v_w1),
                (self.b1, gb1, m_b1, v_b1),
                (self.w2, gw2, m_w2, v_w2),
                (self.b2, gb2, m_b2, v_b2),
            ):
                m[:] = 0.9 * m + 0.1 * grad
                v[:] = 0.999 * v + 0.001 * (grad ** 2)
                m_hat = m / (1.0 - 0.9 ** step)
                v_hat = v / (1.0 - 0.999 ** step)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        self.trained_samples = int(total_eligible)
        _, _, logits = self._forward(x_norm)
        preds = np.argmax(logits, axis=1)
        acc = float(np.mean(preds == y_vec))
        return {
            "trained_samples": int(total_eligible),
            "unique_decision_states": int(n),
            "final_loss": final_loss,
            "train_accuracy": acc,
        }

    def select_candidates(
        self,
        task: str,
        model,
        diagnosis: dict,
        epochs: int = 5,
        max_candidates: int = 3,
    ) -> List[Strategy]:
        """根据诊断向量预测最优候选策略列表，自动施加任务兼容性掩码。"""
        if not 1 <= epochs <= 1000 or not 1 <= max_candidates <= 4:
            raise ValueError("epochs in 1..1000; max_candidates in 1..4")
        vec = diagnosis_to_state_vector(task, model, diagnosis)
        x_norm = np.clip((vec - self.mean) / self.scale, -10.0, 10.0).reshape(1, -1)
        _, _, logits = self._forward(x_norm)
        scores = logits.reshape(-1).copy()

        # 施加任务兼容掩码：分类任务屏蔽设备均衡与 RUL 非对称惩罚；回归任务屏蔽类别加权
        if task == "A":
            scores[STRATEGY_IDS.index("device_balanced")] = -1e9
            scores[STRATEGY_IDS.index("penalize_late_rul")] = -1e9
        else:
            scores[STRATEGY_IDS.index("class_weighted")] = -1e9

        shifted = scores - np.max(scores)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        ranked_indices = np.argsort(-scores)

        lr = float(model.meta["learning_rate"])
        batch = int(model.meta["batch_size"])
        low = max(lr * 0.3, 1e-6)

        selected: List[Strategy] = []
        for idx in ranked_indices:
            if len(selected) >= max_candidates or scores[idx] <= -1e8:
                break
            sid = STRATEGY_IDS[int(idx)]
            reason = f"策略小模型推荐 (strategy={sid}, prob={probs[idx]:.3f})"
            selected.append(build_strategy_by_id(sid, task, low, batch, epochs, reason_override=reason))

        if not any(s.id == "conservative" for s in selected):
            if len(selected) < max_candidates:
                selected.insert(0, build_strategy_by_id("conservative", task, low, batch, epochs))
            else:
                selected[-1] = build_strategy_by_id("conservative", task, low, batch, epochs)
        return selected[:max_candidates]

    def save(self, path: Path) -> None:
        path = Path(path)
        meta = {
            "format": "harness.strategy-mlp.v1",
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "trained_samples": self.trained_samples,
            "strategy_ids": STRATEGY_IDS,
        }
        arrays = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "mean": self.mean,
            "scale": self.scale,
            "metadata": np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        }
        with path.open("wb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as zf:
            for name in sorted(arrays):
                buf = io.BytesIO()
                np.lib.format.write_array(buf, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                zf.writestr(info, buf.getvalue())

    @classmethod
    def load(cls, path: Path) -> "StrategyRecommenderMLP":
        with np.load(Path(path), allow_pickle=False) as arrays:
            meta = json.loads(str(arrays["metadata"]))
            inst = cls(
                state_dim=int(meta["state_dim"]),
                hidden_dim=int(meta["hidden_dim"]),
                seed=int(meta["seed"]),
            )
            inst.w1 = arrays["w1"].copy()
            inst.b1 = arrays["b1"].copy()
            inst.w2 = arrays["w2"].copy()
            inst.b2 = arrays["b2"].copy()
            inst.mean = arrays["mean"].copy()
            inst.scale = arrays["scale"].copy()
            inst.trained_samples = int(meta.get("trained_samples", 0))
            return inst


def choose_strategies(
    task: str,
    model,
    diagnosis: dict,
    epochs: int = 5,
    max_candidates: int = 3,
    decision_mode: str = "rule",
    strategy_model: Optional[StrategyRecommenderMLP] = None,
) -> List[Strategy]:
    """
    统一候选策略选择入口：
    - `decision_mode == "rule"`（默认）：调用 `rule_select_candidates` 规则路由表
    - `decision_mode == "small_model"`：使用 `StrategyRecommenderMLP` 小模型推荐候选策略
    - `decision_mode == "hybrid"`：融合规则路由首选与小模型推荐候选
    """
    if decision_mode == "small_model" and strategy_model is not None:
        return strategy_model.select_candidates(
            task, model, diagnosis, epochs=epochs, max_candidates=max_candidates
        )
    if decision_mode == "hybrid" and strategy_model is not None:
        rule_cands = rule_select_candidates(task, model, diagnosis, epochs=epochs, max_candidates=max_candidates)
        model_cands = strategy_model.select_candidates(
            task, model, diagnosis, epochs=epochs, max_candidates=max_candidates
        )
        merged: List[Strategy] = []
        seen = set()
        for cand in rule_cands[:2] + model_cands + rule_cands[2:]:
            if cand.id not in seen and len(merged) < max_candidates:
                merged.append(cand)
                seen.add(cand.id)
        return merged
    return rule_select_candidates(task, model, diagnosis, epochs=epochs, max_candidates=max_candidates)
