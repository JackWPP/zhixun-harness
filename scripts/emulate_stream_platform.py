#!/usr/bin/env python3
"""批次流平台 emulator（task-4 交付物；独立于 rsi/stream，不 import 仓库任何模块）。

目的
----
官方评测未开放、平台没有公开 SDK/文档，镜像内部的批次流契约只能推断。本脚本把
「平台最可能的样子」做成可执行 fixture，用来：

1. `make`     —— 用三个官方 baseline 包里的 `public/validation.jsonl` 切出 N 批
                 （默认 10 批），生成每一轮的平台注入布局：
                 `round-XX/input/{batch,train}.jsonl` + `round-XX/package/public/{train,validation}.jsonl`
                 + 父 `checkpoint.npz` + `expected.json`（期望 schema/行数/类分布/文件摘要）。
2. `demo`     —— 内置一个「不训练、只 retain」的最小 harness，按官方 predict 协议
                 逐批产出 `predictions.jsonl` / `decision.json` / `workspace_report.json`，
                 然后把所有输出用 `validate` 校验一遍（端到端自检，不需要 Lead 的 rsi/stream）。
3. `validate` —— 对 make/demo 的产物做期望 schema 校验（行列合法性、累积重发语义、
                 批间不重叠、预测/决策/报告结构、指标可复算）。
4. `run`      —— 用外部 harness 命令（如 `./.venv/bin/python -m rsi.cli ...`）逐轮跑，
                 跑完校验产物。用于 Lead 的 rsi/stream 对照，但本脚本自身零依赖。

官方数据来源（只读）：`.scratch/contract/baselines/unpacked/<id>/`，缺失时自动从
`archives/*.zip` 读取。所有生成物只写到 `.scratch/contract/emulator/`（--out 可改）。

用法
----
    ./.venv/bin/python scripts/emulate_stream_platform.py make            # 默认 A，10 批
    ./.venv/bin/python scripts/emulate_stream_platform.py make --dataset all
    ./.venv/bin/python scripts/emulate_stream_platform.py validate --dataset A
    ./.venv/bin/python scripts/emulate_stream_platform.py demo            # make + 模拟 harness + 校验
    ./.venv/bin/python scripts/emulate_stream_platform.py run --dataset A \
        --harness-cmd './.venv/bin/python -m rsi.cli run --tasks A --rounds 1'

退出码：0 = 通过；1 = 校验失败；2 = 通过但有告警（例如输出缺失/指标超出容差）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / ".scratch" / "contract" / "emulator"
PACKAGE_SEARCH = ROOT / ".scratch" / "contract" / "baselines" / "unpacked"
ARCHIVES = ROOT / "archives"
RULES_SNAPSHOT = ROOT / ".scratch" / "contract" / "api" / "body" / "rules.json"

FORBIDDEN_INPUT_KEYS = {
    "rul",
    "true_rul",
    "normalized_rul",
    "remaining_useful_life",
    "remaining_life",
    "total_snapshots",
    "target",
    "label",
    "answer",
}
SUPPORTED_DATASETS = ("A", "B", "C")
DEFAULT_BATCHES = 10

EXIT_OK, EXIT_FAIL, EXIT_WARN = 0, 1, 2


# ------------------------------------------------------------------ utils


class Check:
    """校验结果收集器。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.rows.append({"level": "PASS", "name": name, "detail": detail})

    def info(self, name: str, detail: str = "") -> None:
        self.rows.append({"level": "INFO", "name": name, "detail": detail})

    def warn(self, name: str, detail: str) -> None:
        self.rows.append({"level": "WARN", "name": name, "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        self.rows.append({"level": "FAIL", "name": name, "detail": detail})

    def counts(self) -> dict:
        return {
            lvl: sum(1 for r in self.rows if r["level"] == lvl)
            for lvl in ("PASS", "INFO", "WARN", "FAIL")
        }

    def exit_code(self) -> int:
        c = self.counts()
        if c["FAIL"]:
            return EXIT_FAIL
        if c["WARN"]:
            return EXIT_WARN
        return EXIT_OK

    def print(self, verbose: bool = False) -> None:
        width = max((len(r["name"]) for r in self.rows), default=12)
        for row in self.rows:
            if not verbose and row["level"] == "PASS" and not row["detail"]:
                continue
            print(f"[{row['level']:4}] {row['name']:<{width}}  {row['detail']}")
        c = self.counts()
        print(f"\n合计: PASS={c['PASS']} INFO={c['INFO']} WARN={c['WARN']} FAIL={c['FAIL']}")


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def link_or_copy(src: Path, dst: Path) -> None:
    """input/ 与 package/public/ 内容相同：优先硬链接，省一半磁盘；跨设备时复制。"""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ package discovery


class Package:
    def __init__(
        self,
        dataset_key: str,
        baseline_id: str,
        source_dir: Path,
        info: dict,
        config: dict,
        train_rows: list[dict],
        validation_rows: list[dict],
        checkpoint_bytes: bytes,
        validation_path: Path,
    ) -> None:
        self.dataset_key = dataset_key
        self.baseline_id = baseline_id
        self.source_dir = source_dir
        self.info = info
        self.config = config
        self.train_rows = train_rows
        self.validation_rows = validation_rows
        self.checkpoint_bytes = checkpoint_bytes
        self.validation_path = validation_path

    @property
    def task(self) -> str:
        return self.config.get("task") or self.info.get("task") or "classification"

    @property
    def classes(self) -> list[str]:
        return list(self.config.get("classes") or [])

    @property
    def target_unit(self) -> str:
        return self.config.get("target_unit") or self.info.get("target_unit") or ("class" if self.task == "classification" else "unknown")

    @property
    def checkpoint_sha256(self) -> str:
        return "sha256:" + sha256_bytes(self.checkpoint_bytes)

    @property
    def code_sha256(self) -> str:
        return self.info.get("code_sha256") or "sha256:" + "0" * 64

    def describe(self) -> dict:
        return {
            "dataset_key": self.dataset_key,
            "baseline_id": self.baseline_id,
            "task": self.task,
            "classes": self.classes,
            "target_unit": self.target_unit,
            "train_rows": len(self.train_rows),
            "validation_rows": len(self.validation_rows),
            "baseline_checkpoint_sha256": self.checkpoint_sha256,
            "baseline_code_sha256": self.code_sha256,
            "label_status": self.info.get("label_status"),
            "source": rel(self.source_dir),
        }


def _load_package_from_dir(directory: Path) -> Package:
    info = json.loads((directory / "baseline-info.json").read_text(encoding="utf-8"))
    key = str(info["dataset_key"])
    configs = sorted((directory / "baselines" / "configs").glob(f"{key}.json"))
    config = json.loads(configs[0].read_text(encoding="utf-8")) if configs else {}
    train_rows = read_jsonl(directory / "public" / "train.jsonl")
    validation_path = directory / "public" / "validation.jsonl"
    return Package(
        dataset_key=key,
        baseline_id=info.get("id") or directory.name,
        source_dir=directory,
        info=info,
        config=config,
        train_rows=train_rows,
        validation_rows=read_jsonl(validation_path),
        checkpoint_bytes=(directory / "checkpoint.npz").read_bytes(),
        validation_path=validation_path,
    )


def load_packages(keys: list[str]) -> dict[str, Package]:
    found: dict[str, Package] = {}
    if PACKAGE_SEARCH.is_dir():
        for directory in sorted(PACKAGE_SEARCH.iterdir()):
            if not (directory / "baseline-info.json").is_file():
                continue
            pkg = _load_package_from_dir(directory)
            found.setdefault(pkg.dataset_key, pkg)
    # 兜底：直接读 archives 里的 zip
    for zip_path in sorted(ARCHIVES.glob("*.zip")):
        if not re.match(r"cpu-[abc]-", zip_path.stem):
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                info_name = next(n for n in zf.namelist() if n.endswith("baseline-info.json"))
                info = json.loads(zf.read(info_name).decode("utf-8"))
                key = str(info["dataset_key"])
                if key in found or key not in keys:
                    continue
                prefix = info_name.rsplit("/", 1)[0]
                config_name = f"{prefix}/baselines/configs/{key}.json"
                config = json.loads(zf.read(config_name).decode("utf-8")) if config_name in zf.namelist() else {}
                train_rows = [json.loads(l) for l in zf.read(f"{prefix}/public/train.jsonl").decode("utf-8").splitlines() if l.strip()]
                val_rows = [json.loads(l) for l in zf.read(f"{prefix}/public/validation.jsonl").decode("utf-8").splitlines() if l.strip()]
                found[key] = Package(
                    dataset_key=key,
                    baseline_id=info.get("id") or zip_path.stem,
                    source_dir=zip_path,
                    info=info,
                    config=config,
                    train_rows=train_rows,
                    validation_rows=val_rows,
                    checkpoint_bytes=zf.read(f"{prefix}/checkpoint.npz"),
                    validation_path=zip_path,
                )
        except (KeyError, StopIteration, zipfile.BadZipFile, ValueError):
            continue
    missing = [k for k in keys if k not in found]
    if missing:
        raise SystemExit(
            f"找不到数据集 {missing} 的官方 baseline 包。先解压 archives/*.zip 到 {rel(PACKAGE_SEARCH)}/"
        )
    return {k: found[k] for k in keys}


# ------------------------------------------------------------------ schema helpers


def describe_value(value):
    if isinstance(value, list):
        return {"kind": "list", "len": len(value), "elem": describe_scalar(value[0]) if value else "unknown"}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": {k: describe_value(v) for k, v in sorted(value.items())}}
    return {"kind": "scalar", "type": describe_scalar(value)}


def describe_scalar(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def input_schema(rows: list[dict]) -> dict:
    first = rows[0].get("inputs")
    return describe_value(first) if isinstance(first, dict) else {"kind": "unknown"}


def schema_matches(expected, actual) -> bool:
    if expected.get("kind") != actual.get("kind"):
        return False
    if expected["kind"] == "list":
        return expected["len"] == actual["len"] and expected.get("elem") == actual.get("elem")
    if expected["kind"] == "dict":
        if set(expected["keys"]) != set(actual["keys"]):
            return False
        return all(schema_matches(expected["keys"][k], actual["keys"][k]) for k in expected["keys"])
    return expected.get("type") == actual.get("type")


def find_forbidden_keys(value, path: str = "inputs") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            sub_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_INPUT_KEYS:
                hits.append(sub_path)
            hits.extend(find_forbidden_keys(sub, sub_path))
    elif isinstance(value, list):
        for idx, sub in enumerate(value[:8]):  # 只抽查前 8 个元素，避免大列表开销
            hits.extend(find_forbidden_keys(sub, f"{path}[{idx}]"))
    return hits


def answer_ok(task: str, answer, classes: list[str]) -> bool:
    if task == "classification":
        return isinstance(answer, str) and (not classes or answer in classes)
    return isinstance(answer, (int, float)) and not isinstance(answer, bool) and answer >= 0


# ------------------------------------------------------------------ batching / make


def split_batches(rows: list[dict], n: int) -> list[list[dict]]:
    n = max(1, min(n, len(rows)))
    size, remainder = divmod(len(rows), n)
    batches, start = [], 0
    for i in range(n):
        take = size + (1 if i < remainder else 0)
        batches.append(rows[start : start + take])
        start += take
    return batches


def batch_manifest(rows: list[dict], task: str) -> dict:
    manifest = {"rows": len(rows), "ids_sha256": sha256_bytes("\n".join(str(r.get("id")) for r in rows).encode())}
    if task == "classification":
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row.get("answer"))] = counts.get(str(row.get("answer")), 0) + 1
        manifest["class_counts"] = dict(sorted(counts.items()))
    return manifest


def build_suite(keys: list[str], batches: int, out: Path, force: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    packages = load_packages(keys)
    generated_at = iso_now()
    suite = {"generated_at": generated_at, "batches": batches, "datasets": [], "notes": []}

    for key in keys:
        pkg = packages[key]
        ds_dir = out / key
        if ds_dir.exists() and force:
            shutil.rmtree(ds_dir)
        ds_dir.mkdir(parents=True, exist_ok=True)

        task_doc = {
            **pkg.describe(),
            "id_field": "id",
            "inputs_field": "inputs",
            "answer_field": "answer",
            "rounds": batches,
            "input_schema": input_schema(pkg.validation_rows),
            "source_validation": rel(pkg.validation_path),
            "source_validation_sha256": sha256_file(pkg.validation_path) if pkg.validation_path.suffix == ".jsonl" else None,
            "generated_at": generated_at,
        }
        write_json(ds_dir / "task.json", task_doc)

        splits = split_batches(pkg.validation_rows, batches)
        cumulative_validation: list[dict] = []
        for round_index, batch_rows in enumerate(splits, start=1):
            round_dir = ds_dir / f"round-{round_index:02d}"
            input_dir = round_dir / "input"
            package_dir = round_dir / "package"
            public_dir = package_dir / "public"
            for path in (input_dir, public_dir):
                path.mkdir(parents=True, exist_ok=True)

            # 累积重发语义：第 r 轮可用的 train = 官方 train + 前 r 批 validation（标签随批次解锁）
            cumulative_validation.extend(batch_rows)
            train_rows = pkg.train_rows + cumulative_validation
            write_jsonl(input_dir / "batch.jsonl", batch_rows)
            write_jsonl(input_dir / "train.jsonl", train_rows)
            link_or_copy(input_dir / "batch.jsonl", public_dir / "validation.jsonl")
            link_or_copy(input_dir / "train.jsonl", public_dir / "train.jsonl")
            (package_dir / "checkpoint.npz").write_bytes(pkg.checkpoint_bytes)
            (round_dir / "task.json").write_text(json.dumps(task_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            expected = {
                "round_index": round_index,
                "batch_index": round_index,
                "dataset_key": key,
                "task": pkg.task,
                "batch": batch_manifest(batch_rows, pkg.task),
                "cumulative_train_rows": len(train_rows),
                "parent_checkpoint_sha256": pkg.checkpoint_sha256,
                "parent_code_sha256": pkg.code_sha256,
                "expected_prediction_ids": [str(r.get("id")) for r in batch_rows],
                "input_schema": task_doc["input_schema"],
            }
            write_json(round_dir / "expected.json", expected)

        suite["datasets"].append(
            {
                "key": key,
                "baseline_id": pkg.baseline_id,
                "task": pkg.task,
                "rounds": batches,
                "batch_sizes": [len(b) for b in splits],
                "train_rows": len(pkg.train_rows),
                "validation_rows": len(pkg.validation_rows),
                "dir": rel(ds_dir),
            }
        )
        print(
            f"[make] {key}: {pkg.baseline_id} task={pkg.task} train={len(pkg.train_rows)} "
            f"validation={len(pkg.validation_rows)} -> {batches} 批 {[len(b) for b in splits]} @ {rel(ds_dir)}"
        )

    suite["notes"] = [
        "batch 流语义（推断）：每批带标签、累积重发；第 r 轮可见 train = 官方 train + 前 r 批 validation。",
        "父 checkpoint = 对应官方 baseline 包 checkpoint.npz；后续轮次由 harness 产出候选 checkpoint。",
        "平台真实注入路径/入口命令未知，见 docs/19_SUBMISSION_CONTRACT.md 19.3/19.4。",
    ]
    write_json(out / "suite.json", suite)
    return out


# ------------------------------------------------------------------ validation


def validate_prediction_line(line: dict, task: str, classes: list[str], errors: list[str], label: str) -> None:
    if not isinstance(line, dict):
        errors.append(f"{label}: 行不是对象")
        return
    if "id" not in line or not isinstance(line.get("id"), str):
        errors.append(f"{label}: 缺少字符串 id")
    payload = line.get("outputs")
    # 官方 predict.py 的 stdout 行 = {"outputs": payload}；predictions.jsonl 行 = {"id","outputs": payload}
    if task == "classification":
        if not isinstance(payload, list) or not payload:
            errors.append(f"{label}: 分类 outputs 应为非空列表 [{{label,score}}]")
            return
        total = 0.0
        for pred in payload:
            if not isinstance(pred, dict) or "label" not in pred or "score" not in pred:
                errors.append(f"{label}: 预测项必须是 {{label,score}}")
                return
            if classes and pred["label"] not in classes:
                errors.append(f"{label}: 未知类别 {pred['label']!r}")
            if not isinstance(pred["score"], (int, float)) or isinstance(pred["score"], bool):
                errors.append(f"{label}: score 非数值")
            else:
                total += float(pred["score"])
        if abs(total - 1.0) > 1e-3:
            errors.append(f"{label}: score 之和 {total:.4f} != 1")
    else:
        if not isinstance(payload, dict) or "value" not in payload:
            errors.append(f"{label}: 回归 outputs 应为 {{\"value\": number}}")
            return
        value = payload.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{label}: 回归 value 非数值")
        elif value < 0:
            errors.append(f"{label}: 回归 value 为负（{value}）")


def macro_f1(answers: list[str], preds: list[str], classes: list[str]) -> float:
    scores = []
    for cls in classes or sorted(set(answers)):
        tp = sum(1 for a, p in zip(answers, preds) if a == cls and p == cls)
        fp = sum(1 for a, p in zip(answers, preds) if a != cls and p == cls)
        fn = sum(1 for a, p in zip(answers, preds) if a == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def metrics_from_predictions(
    rows: list[dict], predictions: list[dict], task: str, classes: list[str]
) -> dict:
    by_id = {str(p.get("id")): p for p in predictions}
    answers = [row.get("answer") for row in rows]
    if task == "classification":
        preds = []
        for row in rows:
            item = by_id.get(str(row.get("id"))) or {}
            payload = item.get("outputs")
            outs = payload if isinstance(payload, list) else []
            best = max(outs, key=lambda o: o.get("score", 0)) if outs else None
            preds.append(best.get("label") if best else None)
        correct = sum(1 for a, p in zip(answers, preds) if a == p)
        return {
            "macro_f1": round(macro_f1([str(a) for a in answers], [str(p) for p in preds], classes), 6),
            "accuracy": round(correct / len(answers), 6) if answers else 0.0,
        }
    values = []
    for row in rows:
        item = by_id.get(str(row.get("id"))) or {}
        payload = item.get("outputs")
        value = payload.get("value") if isinstance(payload, dict) else None
        values.append(float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0)
    diffs = [float(a) - v for a, v in zip(answers, values)]
    return {
        "rmse": round((sum(d * d for d in diffs) / len(diffs)) ** 0.5, 6) if diffs else 0.0,
        "mae": round(sum(abs(d) for d in diffs) / len(diffs), 6) if diffs else 0.0,
    }


def validate_round(
    check: Check,
    dataset_key: str,
    round_dir: Path,
    task: dict,
    outputs_dir: Path | None,
    require_outputs: bool,
    seen_validation_ids: set[str],
) -> dict:
    label = f"{dataset_key}/round-{round_dir.name.split('-')[-1]}"
    expected = json.loads((round_dir / "expected.json").read_text(encoding="utf-8"))
    batch_rows = read_jsonl(round_dir / "input" / "batch.jsonl")
    train_rows = read_jsonl(round_dir / "input" / "train.jsonl")
    public_batch = read_jsonl(round_dir / "package" / "public" / "validation.jsonl")
    public_train = read_jsonl(round_dir / "package" / "public" / "train.jsonl")

    # 1) 行数与累积语义
    if len(batch_rows) != expected["batch"]["rows"]:
        check.fail(f"{label}.batch_rows", f"{len(batch_rows)} != 期望 {expected['batch']['rows']}")
    if len(train_rows) != expected["cumulative_train_rows"]:
        check.fail(f"{label}.train_rows", f"{len(train_rows)} != 期望 {expected['cumulative_train_rows']}")
    if len(public_train) != len(train_rows) or len(public_batch) != len(batch_rows):
        check.fail(f"{label}.package", "package/public 与 input 行数不一致")
    if batch_rows and train_rows[: len(train_rows) - len(batch_rows)] == []:
        check.warn(f"{label}.train_accum", "train 未包含历史行（累积重发语义可能不同）")

    # 2) 批间不重叠（validation 只出现一次）
    ids = [str(row.get("id")) for row in batch_rows]
    overlap = seen_validation_ids.intersection(ids)
    if overlap:
        check.fail(f"{label}.overlap", f"批间重复 id: {sorted(overlap)[:5]}")
    seen_validation_ids.update(ids)

    # 3) 行 schema / 泄漏
    errors: list[str] = []
    for row in batch_rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            errors.append("id 缺失")
        inputs = row.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            errors.append(f"{rid}: inputs 缺失/为空")
        else:
            if not schema_matches(task["input_schema"], describe_value(inputs)):
                errors.append(f"{rid}: inputs 结构漂移")
            leaks = find_forbidden_keys(inputs)
            if leaks:
                errors.append(f"{rid}: inputs 含派生答案键 {leaks[:3]}")
        if not answer_ok(task["task"], row.get("answer"), task.get("classes") or []):
            errors.append(f"{rid}: answer 非法 {row.get('answer')!r}")
    if errors:
        check.fail(f"{label}.rows", f"{len(errors)} 处行错误，例如 {errors[:3]}")
    else:
        check.ok(f"{label}.rows", f"{len(batch_rows)} 行通过 schema/泄漏检查")

    # 4) 输出（demo/run 之后）
    round_out = None
    if outputs_dir is not None:
        candidate = outputs_dir / f"round-{round_dir.name.split('-')[-1]}"
        if candidate.is_dir():
            round_out = candidate
    if round_out is None:
        if require_outputs:
            check.fail(f"{label}.outputs", "缺少本轮输出目录")
        else:
            check.info(f"{label}.outputs", "尚未产生输出（validate 只校验输入 fixture）")
        return {"round": round_dir.name, "batch": len(batch_rows)}

    decision_path = round_out / "decision.json"
    pred_path = round_out / "predictions.jsonl"
    if not decision_path.is_file():
        check.fail(f"{label}.decision", "缺少 decision.json")
    else:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("decision") not in ("promote", "retain", "rollback"):
            check.fail(f"{label}.decision", f"decision 非法: {decision.get('decision')!r}")
        for field in ("round_index", "parent_checkpoint_sha256"):
            if field not in decision:
                check.warn(f"{label}.decision", f"缺少字段 {field}")
        if decision.get("decision") == "promote" and decision.get("candidate_checkpoint_sha256") == decision.get("parent_checkpoint_sha256"):
            check.warn(f"{label}.decision", "promote 但候选与父 checkpoint 摘要相同")
        check.ok(f"{label}.decision", f"decision={decision.get('decision')}")

    if not pred_path.is_file():
        check.fail(f"{label}.predictions", "缺少 predictions.jsonl")
        return {"round": round_dir.name, "batch": len(batch_rows)}

    predictions = read_jsonl(pred_path)
    if len(predictions) != len(batch_rows):
        check.fail(f"{label}.predictions", f"{len(predictions)} 条 != 批次 {len(batch_rows)} 行")
    if [str(p.get("id")) for p in predictions] != ids:
        check.warn(f"{label}.predictions", "预测 id 顺序与批次不一致（平台可能按 id 对齐）")
    per_line_errors: list[str] = []
    for idx, line in enumerate(predictions):
        validate_prediction_line(line, task["task"], task.get("classes") or [], per_line_errors, f"line{idx + 1}")
    if per_line_errors:
        check.fail(f"{label}.predictions.schema", f"{len(per_line_errors)} 处错误，例如 {per_line_errors[:3]}")
    else:
        check.ok(f"{label}.predictions.schema", f"{len(predictions)} 条预测符合官方 predict 协议")

    metrics = metrics_from_predictions(batch_rows, predictions, task["task"], task.get("classes") or [])
    return {"round": round_dir.name, "batch": len(batch_rows), "metrics": metrics, "predictions": len(predictions)}


def validate_dataset(
    check: Check,
    ds_dir: Path,
    outputs_root: Path | None,
    require_outputs: bool,
) -> dict | None:
    task = json.loads((ds_dir / "task.json").read_text(encoding="utf-8"))
    rounds = sorted(p for p in ds_dir.glob("round-*") if p.is_dir())
    if not rounds:
        check.fail(f"{ds_dir.name}.rounds", "没有任何 round-* 目录（先 make）")
        return None
    seen: set[str] = set()
    round_metrics = []
    outputs_dir = None
    if outputs_root is not None:
        candidate = outputs_root / ds_dir.name
        outputs_dir = candidate if candidate.is_dir() else None
    for round_dir in rounds:
        summary = validate_round(check, ds_dir.name, round_dir, task, outputs_dir, require_outputs, seen)
        if summary:
            round_metrics.append(summary)
    total_validation = sum(r["batch"] for r in round_metrics)
    if total_validation != task["validation_rows"]:
        check.fail(f"{ds_dir.name}.coverage", f"{total_validation} != validation_rows {task['validation_rows']}")
    else:
        check.ok(f"{ds_dir.name}.coverage", f"{len(rounds)} 批覆盖全部 {total_validation} 条 validation，无重叠")
    return {"task": task, "rounds": round_metrics}


def report_paths(out_root: Path, key: str, outputs_root: Path | None) -> dict:
    ds = out_root / key
    rounds = sorted(p for p in ds.glob("round-*") if p.is_dir())
    outputs = None
    if outputs_root is not None and (outputs_root / key).is_dir():
        outputs = outputs_root / key
    return {"dataset_dir": ds, "rounds": rounds, "outputs_dir": outputs}


def cmd_validate(args) -> int:
    out_root = Path(args.out)
    check = Check()
    keys = parse_datasets(args.dataset)
    for key in keys:
        ds = out_root / key
        if not (ds / "task.json").is_file():
            check.fail(f"{key}.fixture", f"缺少 {rel(ds)}/task.json（先 make --dataset {key}）")
            continue
        outputs_root = Path(args.outputs) if args.outputs else (out_root / "outputs" if (out_root / "outputs").is_dir() else None)
        validate_dataset(check, ds, outputs_root, require_outputs=args.require_outputs)
        if outputs_root is None and args.require_outputs:
            check.fail(f"{key}.outputs", "找不到 outputs 根目录")
    check.print(verbose=args.verbose)
    return check.exit_code()


# ------------------------------------------------------------------ demo (内置最小 harness)


def baseline_metrics(task_doc: dict) -> dict:
    if task_doc["task"] == "classification":
        return {"macro_f1": None, "accuracy": None}
    return {"rmse": None, "mae": None}


def cmd_demo(args) -> int:
    out_root = Path(args.out)
    keys = parse_datasets(args.dataset)
    out_root.mkdir(parents=True, exist_ok=True)
    suite = build_suite(keys, args.batches, out_root, force=args.force)
    check = Check()
    check.ok("make", f"生成 {len(keys)} 个数据集 x {args.batches} 批 @ {rel(out_root)}")

    outputs_root = out_root / "outputs"
    if outputs_root.exists() and args.force:
        shutil.rmtree(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    rules = json.loads(RULES_SNAPSHOT.read_text(encoding="utf-8")) if RULES_SNAPSHOT.is_file() else {}
    evaluation_config_id = (rules.get("rules") or {}).get("evaluation_config_id")

    report = {"datasets": []}
    started = time.time()
    for key in keys:
        ds = out_root / key
        task_doc = json.loads((ds / "task.json").read_text(encoding="utf-8"))
        classes = task_doc.get("classes") or []
        rounds = sorted(p for p in ds.glob("round-*") if p.is_dir())
        out_ds = outputs_root / key
        out_ds.mkdir(parents=True, exist_ok=True)
        ds_rounds = []
        last_checkpoint = task_doc["baseline_checkpoint_sha256"]
        last_code = task_doc["baseline_code_sha256"]
        prediction_total = 0
        all_rows: list[dict] = []
        all_predictions: list[dict] = []
        for round_dir in rounds:
            idx = int(round_dir.name.split("-")[-1])
            batch_rows = read_jsonl(round_dir / "input" / "batch.jsonl")
            round_out = out_ds / round_dir.name
            round_out.mkdir(parents=True, exist_ok=True)

            # 不训练的最小模型：分类均匀分布，回归 0（retain 父 checkpoint，合法例外）
            predictions = []
            for row in batch_rows:
                if task_doc["task"] == "classification":
                    score = round(1.0 / max(1, len(classes)), 6)
                    payload = [{"label": cls, "score": score} for cls in classes]
                else:
                    payload = {"value": 0.0}
                predictions.append({"id": str(row.get("id")), "outputs": payload})
            write_jsonl(round_out / "predictions.jsonl", predictions)
            prediction_total += len(predictions)

            round_metrics = metrics_from_predictions(batch_rows, predictions, task_doc["task"], classes)
            all_rows.extend(batch_rows)
            all_predictions.extend(predictions)

            # retain：候选 = 父模型，无代码变更（合法例外：不训练保留父模型）
            decision = {
                "decision": "retain",
                "round_index": idx,
                "dataset_key": key,
                "reason": "emulator demo: 不训练，保留官方 baseline 父模型（retain 例外）",
                "parent_checkpoint_sha256": last_checkpoint,
                "candidate_checkpoint_sha256": last_checkpoint,
                "candidate_code_sha256": last_code,
                "checkpoint_sha256": last_checkpoint,
                "code_sha256": last_code,
                "predictions": "predictions.jsonl",
                "validation_metrics": round_metrics,
                "history_ids": [],
            }
            write_json(round_out / "decision.json", decision)

            predictions_sha = "sha256:" + sha256_file(round_out / "predictions.jsonl")
            ds_rounds.append(
                {
                    "id": f"{key}-round-{idx:02d}",
                    "round_index": idx,
                    "decision": "retain",
                    "validation_metrics": round_metrics,
                    "history_ids": [],
                    "hypothesis": "批次流 emulator：不训练，验证预测入口与产物契约",
                    "change_summary": "无代码/权重变更（retain）",
                    "parent_checkpoint_sha256": last_checkpoint,
                    "candidate_checkpoint_sha256": last_checkpoint,
                    "candidate_code_sha256": last_code,
                    "checkpoint_sha256": last_checkpoint,
                    "code_sha256": last_code,
                    "predictions_sha256": predictions_sha,
                }
            )

        metrics = metrics_from_predictions(all_rows, all_predictions, task_doc["task"], classes)
        artifact_sha = "sha256:" + sha256_bytes(
            json.dumps(
                {"dataset": key, "rounds": ds_rounds, "metrics": metrics}, ensure_ascii=False, sort_keys=True
            ).encode()
        )
        report["datasets"].append(
            {
                "key": key,
                "rounds": ds_rounds,
                "record_complete": True,
                "prediction_count": prediction_total,
                "cpu_accounting_status": "unavailable",
                "cpu_seconds": round(time.time() - started, 3),
                "wall_seconds": round(time.time() - started, 3),
                "metrics": metrics,
                "baseline_checkpoint_sha256": task_doc["baseline_checkpoint_sha256"],
                "baseline_code_sha256": task_doc["baseline_code_sha256"],
                "checkpoint_sha256": last_checkpoint,
                "code_sha256": last_code,
                "artifact_sha256": artifact_sha,
                "evaluation_config_id": evaluation_config_id,
                "evaluation_config_digest": None,
            }
        )
        print(f"[demo] {key}: {len(rounds)} 轮, {prediction_total} 条预测, metrics={metrics}")

    workspace_report = {"datasets": report["datasets"]}
    write_json(outputs_root / "workspace_report.json", workspace_report)
    write_json(
        outputs_root / "summary.json",
        {
            "generated_at": iso_now(),
            "mode": "emulator-demo",
            "datasets": report["datasets"],
            "workspace_suite": {"rounds": args.batches, "datasets": [{"key": k, "baseline_id": json.loads((suite / k / 'task.json').read_text())['baseline_id']} for k in keys]},
        },
    )
    check.ok("demo.outputs", f"写 {rel(outputs_root)}/{{workspace_report,summary}}.json")

    # 校验 fixture + 输出
    for key in keys:
        validate_dataset(check, out_root / key, outputs_root, require_outputs=True)
    check.print(verbose=args.verbose)
    return check.exit_code()


# ------------------------------------------------------------------ run (外部 harness 对照)


def cmd_run(args) -> int:
    out_root = Path(args.out).resolve()  # 子进程 cwd 是 round workspace，env 必须用绝对路径
    keys = parse_datasets(args.dataset)
    out_root.mkdir(parents=True, exist_ok=True)
    build_suite(keys, args.batches, out_root, force=args.force)
    check = Check()
    outputs_root = out_root / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    extra_env = {}
    for item in args.env or []:
        if "=" not in item:
            raise SystemExit(f"--env 需要 KEY=VALUE 形式: {item}")
        k, v = item.split("=", 1)
        extra_env[k] = v

    for key in keys:
        ds = out_root / key
        task_doc = json.loads((ds / "task.json").read_text(encoding="utf-8"))
        rounds = sorted(p for p in ds.glob("round-*") if p.is_dir())
        out_ds = outputs_root / key
        out_ds.mkdir(parents=True, exist_ok=True)
        for round_dir in rounds:
            idx = int(round_dir.name.split("-")[-1])
            round_out = out_ds / round_dir.name
            round_out.mkdir(parents=True, exist_ok=True)
            for stale in ("decision.json", "predictions.jsonl"):
                if (round_out / stale).exists():
                    (round_out / stale).unlink()
            run_cwd = round_dir / "workspace"
            if run_cwd.exists():
                shutil.rmtree(run_cwd)
            run_cwd.mkdir(parents=True, exist_ok=True)
            env = {
                **os.environ,
                **extra_env,
                "RSI_DATASET": key,
                "RSI_ROUND": str(idx),
                "RSI_BATCH_FILE": str(round_dir / "input" / "batch.jsonl"),
                "RSI_TRAIN_FILE": str(round_dir / "input" / "train.jsonl"),
                "RSI_TASK_FILE": str(round_dir / "task.json"),
                "RSI_PACKAGE_DIR": str(round_dir / "package"),
                "RSI_PARENT_CHECKPOINT": str(round_dir / "package" / "checkpoint.npz"),
                "RSI_OUTPUT_DIR": str(round_out),
                "RSI_WORKSPACE": str(run_cwd),
                "RSI_WORK": str(run_cwd),
            }
            print(f"[run] {key} round-{idx:02d}: {args.harness_cmd}")
            started = time.time()
            try:
                proc = subprocess.run(
                    args.harness_cmd,
                    shell=True,
                    cwd=run_cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired:
                check.fail(f"{key}/round-{idx:02d}.harness", f"超时 >{args.timeout}s")
                continue
            elapsed = time.time() - started
            if proc.returncode != 0:
                check.fail(
                    f"{key}/round-{idx:02d}.harness",
                    f"退出码 {proc.returncode}；stderr 尾部: {proc.stderr.strip().splitlines()[-3:] if proc.stderr else '无'}",
                )
            else:
                check.ok(f"{key}/round-{idx:02d}.harness", f"退出码 0，{elapsed:.1f}s")

            # 收集产物：harness 可能写到 cwd、$RSI_OUTPUT_DIR(=round_out)、$RSI_WORKSPACE
            copied = (round_out / "decision.json").is_file() or (round_out / "predictions.jsonl").is_file()
            for candidate_dir in (run_cwd, round_dir, round_out):
                for name in ("decision.json", "predictions.jsonl"):
                    src = candidate_dir / name
                    dst = round_out / name
                    if not src.is_file() or src.resolve() == dst.resolve():
                        continue
                    shutil.copy2(src, dst)
                    copied = True
            if not copied:
                check.warn(
                    f"{key}/round-{idx:02d}.artifacts",
                    "未找到 decision.json/predictions.jsonl（可能写到自定义路径，见 RSI_OUTPUT_DIR）",
                )
            ws = None
            for candidate in (run_cwd, round_dir, round_out):
                if (candidate / "workspace_report.json").is_file():
                    ws = candidate / "workspace_report.json"
                    break
            if ws is not None:
                shutil.copy2(ws, outputs_root / "workspace_report.json")
                check.ok(f"{key}/round-{idx:02d}.report", f"workspace_report: {rel(ws)}")
    for key in keys:
        validate_dataset(check, out_root / key, outputs_root, require_outputs=False)
    check.print(verbose=args.verbose)
    return check.exit_code()


# ------------------------------------------------------------------ cli


def parse_datasets(value: str) -> list[str]:
    if value == "all":
        return list(SUPPORTED_DATASETS)
    keys = [item.strip().upper() for item in value.split(",") if item.strip()]
    bad = [k for k in keys if k not in SUPPORTED_DATASETS]
    if bad:
        raise SystemExit(f"不支持的数据集: {bad}（可选 A/B/C/all）")
    return keys or ["A"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="批次流平台 emulator：用官方 validation 造批次、模拟 harness、校验契约（独立于 rsi/stream）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="退出码: 0=通过 1=校验失败 2=通过但有告警",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出根目录（默认 {rel(DEFAULT_OUT)}）")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text, aliases in (
        ("make", "用官方 validation 生成 N 批平台布局 fixture", ()),
        ("validate", "校验 fixture（及 demo/run 产物）schema", ()),
        ("demo", "make + 内置最小 harness + 自检", ()),
        ("run", "make + 外部 harness 命令逐轮运行 + 校验", ()),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dataset", default="A", help="A / B / C / all 或逗号组合（默认 A）")
        p.add_argument("--out", default=None, help="覆盖全局 --out")
        p.add_argument("--force", action="store_true", help="覆盖已有 fixture")
        p.add_argument("--batches", type=int, default=DEFAULT_BATCHES, help=f"批数（默认 {DEFAULT_BATCHES}）")
        if name == "validate":
            p.add_argument("--outputs", default=None, help="产物根目录（默认 <out>/outputs）")
            p.add_argument("--require-outputs", action="store_true", help="必须存在 demo/run 产物")
        if name == "run":
            p.add_argument("--harness-cmd", required=True, help="每轮执行的 shell 命令（环境变量见脚本头注释）")
            p.add_argument("--env", action="append", help="额外环境变量 KEY=VALUE（可多次）")
            p.add_argument("--timeout", type=float, default=1800.0, help="单轮超时秒数（默认 1800）")

    args = parser.parse_args(argv)
    if getattr(args, "out", None) is None:
        args.out = str(DEFAULT_OUT)
    os.chdir(ROOT)

    if args.command == "make":
        keys = parse_datasets(args.dataset)
        out = build_suite(keys, args.batches, Path(args.out), force=args.force)
        print(f"[make] suite: {rel(out / 'suite.json')}")
        return EXIT_OK
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "run":
        return cmd_run(args)
    parser.error("unknown command")
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
