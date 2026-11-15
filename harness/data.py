"""Explicit archive intake, integrity checks and benchmark data boundaries."""
import collections
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


def digest_bytes(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(path):
    return digest_bytes(Path(path).read_bytes())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Dataset:
    """Only x/y are optimizer inputs. groups are traceability-based sampling IDs."""
    x: np.ndarray
    y: np.ndarray
    ids: tuple
    groups: tuple
    meta: tuple
    paths: tuple
    sha256: str
    role: str


@dataclass
class Bundle:
    task: str
    root: Path
    model_class: object
    train: Dataset
    validation: Dataset
    checkpoint: Path
    source: dict
    original_report: dict


def import_model(root):
    """Import the explicitly staged vendor module, not arbitrary model hooks."""
    path = Path(root) / "baselines" / "model.py"
    name = "_harness_cpu_model_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Model


def _read_dataset(path, model, role, task):
    paths = tuple(model.meta["paths"])
    parsed = [json.loads(p) for p in paths]
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty dataset")
    ids, groups, features, targets, metadata = [], [], [], [], []
    classes = model.meta["classes"]

    def flatten(value, prefix=()):
        result = []
        if isinstance(value, dict):
            for key in sorted(value):
                result.extend(flatten(value[key], prefix + (("key", key),)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                result.extend(flatten(child, prefix + (("index", index),)))
        elif isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
            result.append((json.dumps(prefix, ensure_ascii=False, separators=(",", ":")), float(value)))
        else:
            raise ValueError("inputs must contain finite numeric features")
        return result

    for row in rows:
        identity = row["id"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("invalid sample ID")
        pairs = flatten(row["inputs"])
        if tuple(p[0] for p in pairs) != paths:
            raise ValueError("dataset input schema differs from checkpoint")
        answer = row["answer"]
        if classes:
            target = classes.index(answer)
        else:
            if isinstance(answer, bool) or not isinstance(answer, (float, int)) or not np.isfinite(answer) or answer < 0:
                raise ValueError("RUL must be finite and nonnegative")
            target = float(answer)
        meta = row.get("meta", {})
        if task == "A":
            group = str(meta["source_batch"])
        elif task == "B":
            group = str(meta["bearing"])
        else:
            group = str(meta["source_subset"]) + ":" + str(meta["unit"])
        ids.append(identity)
        groups.append(group)
        metadata.append(meta)
        features.append([p[1] for p in pairs])
        targets.append(target)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs")
    x = np.array(features, dtype=np.float64)
    y = np.array(targets, dtype=np.int64 if classes else np.float64)
    x.flags.writeable = False
    y.flags.writeable = False
    return Dataset(x, y, tuple(ids), tuple(groups), tuple(metadata), paths, digest_bytes(raw), role)


def prepare_bundle(archive_path, task, destination):
    """Copy only known members into a NEW run-local directory; never modify source."""
    archive_path, destination = Path(archive_path).resolve(), Path(destination).resolve()
    if task not in ("A", "B", "C"):
        raise ValueError("task must be A, B or C")
    if not archive_path.is_file():
        raise FileNotFoundError(str(archive_path))
    if destination.exists():
        raise ValueError("intake destination must not already exist")
    if not destination.parent.is_dir():
        raise ValueError("intake parent must exist")
    required = ["checkpoint.npz", "training.json", "baseline-info.json", "public/train.jsonl",
                "public/validation.jsonl", "baselines/model.py", "baselines/data.py",
                "baselines/train.py", "baselines/predict.py", "baselines/configs/" + task + ".json"]
    optional = ["TASK.md", "TRAINING_GUIDE.md", "START.md", "history.jsonl", "baselines/requirements.txt"]
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        if any(n not in names for n in required):
            raise ValueError("archive missing required benchmark files")
        if any(archive.getinfo(n).file_size > 128 * 1024 * 1024 for n in required):
            raise ValueError("benchmark member too large")
        report = json.loads(archive.read("training.json"))
        if report["profile"] != task:
            raise ValueError("requested task does not match archive profile")
        raw_by_name = {name: archive.read(name) for name in required + optional if name in names}
        expected = {"checkpoint.npz": report["checkpoint_sha256"],
                    "public/train.jsonl": report["train_sha256"],
                    "public/validation.jsonl": report["validation_sha256"]}
        for name, sha in report["source_sha256"].items():
            expected["baselines/" + name] = sha
        for name, sha in expected.items():
            if name not in raw_by_name or digest_bytes(raw_by_name[name]) != sha:
                raise ValueError("source integrity mismatch: " + name)
        destination.mkdir()
        for name, raw in raw_by_name.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    cls = import_model(destination)
    checkpoint = destination / "checkpoint.npz"
    model = cls.load(checkpoint)
    if model.meta.get("profile") != task:
        raise ValueError("checkpoint profile mismatch")
    expected_task = "classification" if task == "A" else "regression"
    if model.meta["task"] != expected_task:
        raise ValueError("wrong task type; use the B RUL package, not historical B classification")
    if task == "B" and model.meta.get("target_unit") != "minutes":
        raise ValueError("B checkpoint must predict minutes")
    train = _read_dataset(destination / "public/train.jsonl", model, "train", task)
    validation = _read_dataset(destination / "public/validation.jsonl", model, "validation", task)
    if set(train.ids) & set(validation.ids):
        raise ValueError("train/validation sample overlap")
    if set(train.groups) & set(validation.groups):
        raise ValueError("train/validation device or batch overlap")
    source = {"archive": str(archive_path), "archive_sha256": digest(archive_path),
              "checkpoint_sha256": digest(checkpoint), "train_sha256": train.sha256,
              "validation_sha256": validation.sha256,
              "official_scoring_ready": report.get("official_scoring_ready", False),
              "source_code_sha256": report["source_sha256"],
              "train_groups": len(set(train.groups)), "validation_groups": len(set(validation.groups))}
    write_json(destination / "intake.json", source)
    return Bundle(task, destination, cls, train, validation, checkpoint, source, report)
