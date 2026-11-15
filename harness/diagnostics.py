"""Read-only, descriptive diagnostics. No fitted state is updated here."""
import collections
import math
import numpy as np


def primary_metric(task):
    return "macro_f1" if task == "A" else "rmse"


def improvement(task, old, new):
    key = primary_metric(task)
    return float(new[key] - old[key] if task == "A" else old[key] - new[key])


def metric_values(model, dataset, indices=None):
    x, y = dataset.x, dataset.y
    if indices is not None:
        x, y = x[indices], y[indices]
    pred = model.predict(x)
    if model.meta["task"] == "classification":
        labels = pred.argmax(axis=1)
        classes = model.meta["classes"]
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        np.add.at(cm, (y, labels), 1)
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(len(classes)), where=denom > 0)
        observed = denom > 0
        return {"macro_f1": float(f1[observed].mean()), "accuracy": float(np.mean(labels == y)),
                "macro_f1_fixed_classes": float(f1.mean()),
                "per_class_f1": dict(zip(classes, map(float, f1))),
                "confusion_matrix": cm.tolist(), "classes": classes,
                "target_counts": dict(zip(classes, map(int, cm.sum(axis=1))))}
    error = pred - y
    result = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
              "mean_signed_error": float(np.mean(error)),
              "overestimate_fraction": float(np.mean(error > 0))}
    if model.meta.get("profile") == "C":
        # Conventional C-MAPSS formula; platform formula/protocol must be confirmed.
        exponent = np.where(error < 0, -error / 13.0, error / 10.0)
        if exponent.max() + math.log(len(exponent)) < 700:
            result["nasa_score"] = float(np.expm1(exponent).sum())
        else:
            result["nasa_score"] = None
        result["nasa_score_protocol"] = "local_conventional_sum_not_verified_platform_score"
    return result


def _group_metrics(model, data, keys=None):
    indices = collections.defaultdict(list)
    for i, key in enumerate(keys if keys is not None else data.groups):
        indices[str(key)].append(i)
    return {key: {"rows": len(idx), "metrics": metric_values(model, data, np.asarray(idx))}
            for key, idx in sorted(indices.items())}


def _operating_clusters(model, train, validation):
    """Fit descriptive setting clusters on train ONLY; assign validation unchanged."""
    settings = [i for i, path in enumerate(model.meta["paths"]) if '["key","settings"]' in path]
    if not settings:
        return {}
    x = train.x[:, settings]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    rng = np.random.default_rng(42)
    centers = z[rng.choice(len(z), min(6, len(z)), replace=False)].copy()
    for _ in range(12):
        labels = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for k in range(len(centers)):
            if np.any(labels == k):
                centers[k] = z[labels == k].mean(axis=0)
    v = (validation.x[:, settings] - mean) / scale
    val_labels = ((v[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return {"method": "train_only_kmeans_settings_k6_descriptive_not_physical_ground_truth",
            "train": _group_metrics(model, train, labels),
            "validation": _group_metrics(model, validation, val_labels),
            "fit_role": "train", "centers_standardized": centers.tolist()}


def diagnose(model, train, validation, task):
    if train.role != "train" or validation.role != "validation":
        raise ValueError("diagnosis requires explicit train and validation roles")
    t, v = metric_values(model, train), metric_values(model, validation)
    findings = []

    def add(code, severity, title, evidence, recommendation):
        findings.append({"code": code, "severity": severity, "title": title,
                         "evidence": evidence, "recommendation": recommendation})

    # Descriptive validation statistics are for diagnosis only, not preprocessing fit.
    shifts = np.abs(validation.x.mean(axis=0) - train.x.mean(axis=0)) / model.scale
    indices = np.argsort(shifts)[-8:][::-1]
    top = [{"feature": model.meta["paths"][int(i)], "standardized_mean_difference": float(shifts[i])}
           for i in indices]
    details = {"train_rows": len(train.x), "validation_rows": len(validation.x),
               "train_group_count": len(set(train.groups)), "validation_group_count": len(set(validation.groups)),
               "feature_shift": {"top": top, "median": float(np.median(shifts)),
                                 "interpretation": "descriptive_input_change_not_causal_or_conditional_drift_proof"}}
    if shifts.max() > 0.5:
        add("observable_input_shift", "notice", "输入均值存在跨集合差异", {"top_features": top[:3]},
            "仅在训练侧尝试合理扰动与正则；不重新拟合验证集标准化，不据此断言物理原因。")
    if task == "A":
        gap = t["macro_f1"] - v["macro_f1"]
        if gap > 0.08:
            add("generalization_gap", "warning", "训练与跨批次验证存在泛化差距", {"macro_f1_gap": gap},
                "比较较小学习率、训练类别加权、L2 正则和训练内轻量扰动。")
        counts = t["target_counts"]
        if min(counts.values()) > 0 and max(counts.values()) / min(counts.values()) > 2:
            add("train_class_imbalance", "notice", "训练类别数量不均衡", {"train_counts": counts},
                "类别权重只能从训练标签计算；验证集类别频率不用于拟合。")
        cm = np.array(v["confusion_matrix"])
        weak = []
        for i, label in enumerate(model.meta["classes"]):
            if cm[i].sum() and v["per_class_f1"][label] < 0.5:
                confusion = cm[i].copy()
                confusion[i] = 0
                j = int(confusion.argmax())
                weak.append({"class": label, "f1": v["per_class_f1"][label], "count": int(cm[i].sum()),
                             "most_confused_with": model.meta["classes"][j], "confused_count": int(confusion[j])})
        if weak:
            add("weak_classes", "warning", "部分气体类别在验证中识别失效", {"classes": weak},
                "检查相应训练类别的覆盖与混淆，不使用验证目标训练校正器。")
        if len(set(train.groups)) < 3:
            add("few_drift_examples", "notice", "跨期训练证据不足以训练通用漂移类型分类器",
                {"training_batches": len(set(train.groups)), "validation_batches": len(set(validation.groups))},
                "先用规则诊断；未来在训练内构造受控扰动，标签仅表示合成扰动。")
    else:
        details["train_by_device"] = _group_metrics(model, train)
        details["validation_by_device"] = _group_metrics(model, validation)
        mean = float(train.y.mean())
        constant_rmse = float(np.sqrt(np.mean((validation.y - mean) ** 2)))
        details["constant_training_mean_reference"] = {"value": mean, "validation_rmse": constant_rmse,
                                                      "fit_role": "train", "not_a_continuation_candidate": True}
        if v["rmse"] > t["rmse"] * 1.3:
            add("generalization_gap", "warning", "跨设备验证误差明显高于训练误差",
                {"train_rmse": t["rmse"], "validation_rmse": v["rmse"]},
                "尝试较小学习率、L2 与训练内设备均衡抽样；不能把设备 ID 当预测特征。")
        if v["rmse"] > constant_rmse:
            add("worse_than_constant", "warning", "原模型不及训练均值常数对照",
                {"model_rmse": v["rmse"], "constant_rmse": constant_rmse},
                "优先检查泛化与目标偏置，常数仅作研究参照而非替换续训模型。")
        counts = collections.Counter(validation.groups)
        if max(counts.values()) / len(validation.x) > 0.5:
            add("validation_device_dominance", "notice", "总体误差由少量长序列设备主导",
                {"largest_device_fraction": max(counts.values()) / len(validation.x), "counts": dict(counts)},
                "同时报告总体与逐设备误差；仍按原主指标门控。")
        if task == "C":
            details["operating_settings_clusters"] = _operating_clusters(model, train, validation)
            add("snapshot_protocol", "notice", "当前参考协议是单时刻输入而非序列会话",
                {"window_sizes": sorted(set(m.get("window_size", 1) for m in train.meta)),
                 "prediction_fields": ["task", "inputs"]},
                "不拼接不同设备或未来观测；NASA 分数使用本地常规公式，仅作辅助。")
        else:
            add("rul_label_delay", "notice", "RUL 真值依赖完整寿命记录", {},
                "当前只做离线续训，不假设在线失效前已经获得真实 RUL。")
    add("no_causal_labels", "info", "没有漂移原因标签，诊断不是因果证明", {},
        "规则输出的是统计证据和可检验候选策略，不宣称已识别真实物理原因。")
    return {"summary": [f["title"] for f in findings], "findings": findings,
            "train_metrics": t, "validation_metrics": v, "details": details}
