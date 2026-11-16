"""Sequential batch-stream supervisor under the honest platform protocol.

For every batch the harness:

1. materialises a package from **revealed history only** (``train`` + ``dev``);
   the current batch inputs are written separately, without labels;
2. optionally adapts the parent on the revealed history (``adapt`` / ``full``);
3. optionally fits a delayed-label pair calibration from revealed history
   (``pair`` / ``full``);
4. for ``full`` runs the unchanged Runner on train/dev to select a candidate;
5. predicts the current batch and *then* scores it against the answers kept
   outside the package (never fed back into training or calibration);
6. reveals the batch labels into the history pool only when the source says the
   platform would have returned them; hidden batches are never revealed.

Arms: ``frozen`` / ``recenter`` / ``pair`` / ``adapt`` / ``full``.  All of them
share the same prediction and scoring path, so differences are exactly the
adaptation mechanism under test.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data import flatten
from ..inference import fit_pair_threshold
from ..model import ExtendedModel, checkpoint_sha256
from ..runner import Runner
from ..metrics import improved, primary as primary_of
from ..tasks import TaskConfig, resolve_tasks
from .protocol import (
    atomic_write_json,
    decision_for,
    evaluate_rows,
    find_run_state,
    materialise_package,
    predict_rows,
    score_predictions,
    write_jsonl,
)
from .sources import make_source

ARMS = ("frozen", "recenter", "pair", "adapt", "full")

# Route selection: no single mechanism wins on every task/source, so the
# default ``auto`` arm is a small deterministic policy derived from the
# private-sim evidence (see docs/25_PRIVATE_SIM.md).
AUTO_ARMS = {
    "official": {"A": "pair", "B": "adapt", "C": "frozen"},
    "private-sim": {"A": "full", "B": "adapt", "C": "frozen"},
}
AUTO_ARMS["external-a"] = AUTO_ARMS["private-sim"]


def resolve_arm(arm: str, task_key: str, source: str) -> str:
    """Expand ``arm="auto"`` (or an unknown arm) into a concrete arm."""

    if arm in ARMS:
        return arm
    table = AUTO_ARMS.get(source)
    if table is None:
        return "frozen"
    return table.get(task_key, "frozen")


@dataclass
class TaskStreamResult:
    key: str
    source: str
    batches: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    retention: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    final: dict = field(default_factory=dict)
    report_path: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key, "source": self.source, "batches": self.batches,
            "hidden": self.hidden, "retention": self.retention,
            "errors": self.errors, "final": self.final,
            "report_path": self.report_path,
        }


def _adapt_on_history(batch_dir: Path, config: TaskConfig, parent_checkpoint: Path,
                      epochs: int, learning_rate: float) -> Path:
    """Continue-train the parent on already-revealed history labels only."""

    package = batch_dir / "package"
    train_path = package / "public" / "train.jsonl"
    model = ExtendedModel.from_checkpoint(Path(parent_checkpoint))
    rows = []
    with open(train_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return Path(parent_checkpoint)
    x = np.asarray([flatten(row.get("inputs") or {})[1] for row in rows], dtype=np.float64)
    if config.task == "classification":
        index = {name: i for i, name in enumerate(config.classes)}
        y = np.asarray([index[str(row.get("answer"))] for row in rows], dtype=np.int64)
    else:
        y = np.asarray([float(row.get("answer")) for row in rows], dtype=np.float64)
    model.set_learning_rate(float(learning_rate))
    model.fit(x, y, epochs=int(epochs))
    out = batch_dir / "adapted_parent.npz"
    if out.exists():
        out.unlink()
    model.save(out)
    shutil.copy2(out, package / "checkpoint.npz")
    return out


def _rows_xy(rows: list[dict], config: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([flatten(row.get("inputs") or {})[1] for row in rows], dtype=np.float64)
    if config.task == "classification":
        index = {name: i for i, name in enumerate(config.classes)}
        y = np.asarray([index[str(row.get("answer"))] for row in rows], dtype=np.int64)
    else:
        y = np.asarray([float(row.get("answer")) for row in rows], dtype=np.float64)
    return x, y


def _compose(base: dict, delta: float | None, pair: tuple[int, int], meta: dict) -> dict:
    if delta is None:
        return base
    return {"type": "pair_threshold", "pair": [int(pair[0]), int(pair[1])],
            "delta": round(float(delta), 6), "base": base, "source": "history_labels",
            "history_rows": int(meta.get("rows", 0))}


def run_task_stream(payload: dict) -> dict:
    key = str(payload["key"])
    source_kind = payload.get("source", "official")
    n_batches = int(payload.get("n_batches", 6))
    seed = int(payload.get("seed", 42))
    arm = str(payload.get("arm", "auto"))
    if arm != "auto" and arm not in ARMS:
        raise ValueError(f"unknown arm {arm}; choices: {ARMS} or 'auto'")
    state_dir = Path(payload["state_dir"])
    controller = payload.get("controller", "auto")
    llm_mode = payload.get("llm_mode", "off")
    memory = bool(payload.get("memory", True))
    experiments_cap = int(payload.get("experiments_cap", 1))
    batch_wall = float(payload.get("batch_wall", 1800.0))
    retention_every = int(payload.get("retention_every", 2))
    verbose = bool(payload.get("verbose", True))
    adapt_epochs = int(payload.get("adapt_epochs", 30))
    adapt_lr = float(payload.get("adapt_lr", 0.0005))
    adapt_weight = float(payload.get("adapt_weight", 0.5))
    recenter_weight = float(payload.get("recenter_weight", 0.4))
    pair_enabled = bool(payload.get("pair_calibration", True)) and arm in ("pair", "full")
    pair = tuple(int(v) for v in payload.get("pair", [0, 3]))
    dev_fraction = float(payload.get("dev_fraction", 0.25))
    # "dev" = initial dev pool; "recent" = last revealed batch rows (rolling
    # validation, tracks drift much better on the official chunk stream).
    adapt_gate_mode = str(payload.get("adapt_gate_mode", "recent"))

    configs = resolve_tasks([key])
    if not configs:
        raise SystemExit(f"unknown task {key}")
    config: TaskConfig = configs[0]
    arm = resolve_arm(arm, config.key, source_kind)

    batch_source = make_source(
        source_kind, config, n_batches,
        external_root=payload.get("external_root"),
        external_start=payload.get("external_start", 3),
        external_end=payload.get("external_end", 10),
        arrival=(payload.get("arrival_start", 3), payload.get("arrival_end", 7)),
        hidden=(payload.get("hidden_start", 8), payload.get("hidden_end", 10)),
        dev_fraction=dev_fraction,
        label_every=int(payload.get("label_every", 1)),
        label_fraction=float(payload.get("label_fraction", 1.0)),
        perturb=str(payload.get("perturb", "none")),
        missing_class=payload.get("missing_class") or None,
        repeat=int(payload.get("repeat", 1)),
        seed=seed,
    )

    task_root = state_dir / "tasks" / key
    ckpt_dir = task_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = task_root / "memory"
    db_path = task_root / "stream.db"
    result = TaskStreamResult(key=key, source=source_kind)
    parent_checkpoint = Path(config.checkpoint_path)
    parent_base_pp: dict = {"type": "none"}
    parent_primary = None
    history_x: list[np.ndarray] = []
    history_y: list[np.ndarray] = []
    recent_reveal: list[dict] = []
    previous: list[tuple[int, Path, dict]] = []
    started = time.time()

    for batch in batch_source:
        index = int(batch.index)
        hidden = bool(batch.meta.get("hidden"))
        batch_dir = task_root / f"batch_{index:04d}"
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_started = time.time()
        phase = "hidden" if hidden else "arrival"
        print(f"[stream:{key}] batch {index} phase={phase} arm={arm} "
              f"train={len(batch.train)} dev={len(batch.dev)} pred={len(batch.prediction)}", flush=True)
        try:
            package_spec = materialise_package(batch_dir, config, batch, parent_checkpoint)
            adapted = False
            adapt_gate = {"attempted": False}
            if arm in ("adapt", "full") and history_x and index > 0:
                before = Path(parent_checkpoint)
                adapted_path = _adapt_on_history(batch_dir, config, before, adapt_epochs, adapt_lr)
                base_pp = {"type": "recenter", "mode": "batch_mean", "weight": adapt_weight}
                # Gate set: recent revealed rows (rolling validation) when
                # available, otherwise the initial dev pool.
                gate_mode = "dev"
                gate_rows = recent_reveal[-2000:]
                if adapt_gate_mode == "recent" and len(gate_rows) >= 50:
                    gate_mode = "recent"
                    gate_path = batch_dir / "gate_recent.jsonl"
                    with open(gate_path, "w", encoding="utf-8") as fh:
                        for row in gate_rows:
                            fh.write(json.dumps({"id": str(row.get("id", "")),
                                                 "inputs": row.get("inputs"),
                                                 "answer": row.get("answer")},
                                                ensure_ascii=False) + "\n")
                else:
                    gate_path = batch_dir / "package" / "public" / "validation.jsonl"
                before_gate = evaluate_rows(config.task, gate_path, before, base_pp, config.classes)
                after_gate = evaluate_rows(config.task, gate_path, adapted_path, base_pp, config.classes)
                keep = False
                if before_gate and after_gate:
                    keep = improved(config.task, after_gate.get("primary"),
                                    before_gate.get("primary"), config.margin)
                adapt_gate = {
                    "attempted": True, "kept": keep, "gate_mode": gate_mode,
                    "before_dev_primary": before_gate.get("primary") if before_gate else None,
                    "after_dev_primary": after_gate.get("primary") if after_gate else None,
                    "gate_rows": len(gate_rows) if gate_mode == "recent" else None,
                    "mode": "rolling-reveal validated adaptation (rollback if not better)",
                }
                if keep:
                    parent_checkpoint = adapted_path
                    adapted = True
                else:
                    # controlled RSI: the adaptation candidate is rejected -> keep parent
                    shutil.copy2(before, batch_dir / "package" / "checkpoint.npz")
            if arm == "recenter":
                parent_base_pp = {"type": "recenter", "mode": "batch_mean", "weight": recenter_weight}
            elif adapted:
                parent_base_pp = {"type": "recenter", "mode": "batch_mean", "weight": adapt_weight}

            pair_delta = None
            pair_meta: dict = {}
            if pair_enabled and history_x:
                model = ExtendedModel.from_checkpoint(parent_checkpoint)
                fit = fit_pair_threshold(np.concatenate(history_x), np.concatenate(history_y),
                                         model, pair=(pair[0], pair[1]))
                if fit:
                    pair_delta = float(fit["delta"])
                    pair_meta = fit
            parent_pp = _compose(parent_base_pp, pair_delta, pair, pair_meta)

            candidate_checkpoint = None
            candidate_base_pp: dict | None = None
            candidate_pp: dict | None = None
            candidate_primary = None
            decision = {"decision": "retain", "delta": 0.0, "reason": f"arm={arm}: no trained candidate"}
            if arm == "full":
                runner = Runner(
                    keys=(key,), rounds=1, llm_mode=llm_mode, seed=seed + index,
                    state_dir=batch_dir, wall_cap=batch_wall, cpu_cap=36000.0,
                    experiments_cap=experiments_cap, swanlab=False,
                    run_dir=batch_dir / "run", verbose=verbose, controller=controller,
                    memory=memory, memory_dir=str(memory_dir), db_path=str(db_path),
                    round_turns=int(payload.get("round_turns", 20)),
                    configs=[package_spec],
                )
                runner.run()
                info = find_run_state(batch_dir / "run")
                state = info.get("state") or {}
                summary = info.get("summary") or {}
                dataset_row = (summary.get("datasets") or [{}])[0]
                task_state = (state.get("tasks") or [{}])[0]
                candidate_checkpoint = task_state.get("checkpoint_path")
                candidate_base_pp = task_state.get("postprocess") or {"type": "none"}
                candidate_pp = _compose(candidate_base_pp, pair_delta, pair, pair_meta)
                candidate_primary = (dataset_row.get("metrics") or {}).get("primary")
                dev_path = batch_dir / "package" / "public" / "validation.jsonl"
                parent_dev = evaluate_rows(config.task, dev_path, parent_checkpoint,
                                           parent_pp, config.classes)
                parent_primary = parent_dev.get("primary")
                if pair_delta is not None and candidate_checkpoint:
                    candidate_eval = evaluate_rows(config.task, dev_path, Path(candidate_checkpoint),
                                                   candidate_pp, config.classes)
                    candidate_primary = candidate_eval.get("primary")
                decision = decision_for(config.task, config.direction, parent_primary,
                                        candidate_primary, config.margin)

            chosen_checkpoint = Path(parent_checkpoint)
            chosen_pp = parent_pp
            chosen_primary = parent_primary
            promote = decision.get("decision") == "promote" and candidate_checkpoint
            if promote:
                chosen_checkpoint = Path(candidate_checkpoint)
                chosen_pp = candidate_pp or {"type": "none"}
                chosen_primary = candidate_primary

            suffix = decision.get("decision", "retain")
            deployed = ckpt_dir / f"batch_{index:04d}_{suffix}.npz"
            shutil.copy2(chosen_checkpoint, deployed)

            prediction_input = batch_dir / "package" / "public" / "prediction.jsonl"
            predictions_path = batch_dir / "predictions.jsonl"
            n_predictions = predict_rows(prediction_input, deployed, chosen_pp, config,
                                         predictions_path)
            prediction_metrics = score_predictions(predictions_path, batch.answers,
                                                   config.task, config.classes)

            # decision.json must be a pure function of the revealed history and
            # current-batch *inputs*: it is written before the answers are used,
            # so permuting / corrupting the current batch answers cannot change
            # it (E0 protocol-invariance test).
            decision_core = {
                "batch": index,
                "phase": phase,
                "arm": arm,
                "revealed": bool(batch.meta.get("revealed")),
                "decision": decision.get("decision"),
                "decision_detail": decision,
                "delta": decision.get("delta"),
                "reason": decision.get("reason"),
                "parent_checkpoint_sha256": checkpoint_sha256(Path(parent_checkpoint)),
                "candidate_checkpoint_sha256": (checkpoint_sha256(Path(candidate_checkpoint))
                                                if candidate_checkpoint else None),
                "chosen_checkpoint_sha256": checkpoint_sha256(Path(deployed)),
                "postprocess": chosen_pp,
                "pair_calibration": ({"delta": pair_delta, **pair_meta}
                                     if pair_delta is not None else {"enabled": False}),
                "adapt_gate": adapt_gate,
                "dev_primary": parent_primary if not promote else chosen_primary,
                "meta": batch.meta,
            }
            atomic_write_json(batch_dir / "decision.json", decision_core)
            evaluation = {
                "prediction_metrics": prediction_metrics,
                "prediction_count": n_predictions,
                "started_at": batch_started,
                "wall_seconds": round(time.time() - batch_started, 3),
            }
            atomic_write_json(batch_dir / "evaluation.json", evaluation)
            record = {**decision_core, **evaluation}
            if hidden:
                result.hidden.append(record)
            else:
                result.batches.append(record)
            previous.append((index, predictions_path, dict(batch.answers)))
            if promote:
                parent_checkpoint = deployed
                parent_base_pp = candidate_base_pp or {"type": "none"}
                parent_primary = chosen_primary
            if batch.reveal and not hidden:
                try:
                    recent_reveal = list(batch.reveal)[-2000:]
                    x, y = _rows_xy(batch.reveal, config)
                    history_x.append(x)
                    history_y.append(y)
                except Exception as exc:  # history indexing is best-effort
                    print(f"[stream:{key}] warning: cannot index revealed labels: {exc}", flush=True)

            if retention_every > 0 and not hidden and (index + 1) % retention_every == 0 and len(previous) > 1:
                for prev_index, prev_pred, prev_answers in previous[-3:-1]:
                    try:
                        tmp_pred = batch_dir / f"retention_{prev_index}.jsonl"
                        predict_rows(prev_pred, deployed, chosen_pp, config, tmp_pred)
                        metrics = score_predictions(tmp_pred, prev_answers, config.task, config.classes)
                        result.retention.append({"batch": index, "evaluated_batch": prev_index,
                                                 "primary": metrics.get("primary")})
                    except Exception as exc:
                        result.retention.append({"batch": index, "evaluated_batch": prev_index,
                                                 "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # keep the stream alive
            error = f"batch {index} failed: {type(exc).__name__}: {exc}"
            result.errors.append(error)
            traceback.print_exc()
            break

    arrival = [b for b in result.batches if b.get("phase") == "arrival"]
    hidden_rows = [b for b in result.hidden]
    result.final = {
        "arm": arm,
        "batches": len(arrival),
        "hidden_batches": len(hidden_rows),
        "errors": len(result.errors),
        "wall_seconds": round(time.time() - started, 3),
        "first_primary": (arrival[0].get("prediction_metrics") or {}).get("primary") if arrival else None,
        "final_primary": (arrival[-1].get("prediction_metrics") or {}).get("primary") if arrival else None,
        "mean_primary": (float(np.mean([(b.get("prediction_metrics") or {}).get("primary") or 0.0
                                        for b in arrival])) if arrival else None),
        "hidden_mean_primary": (float(np.mean([(b.get("prediction_metrics") or {}).get("primary") or 0.0
                                               for b in hidden_rows])) if hidden_rows else None),
        "hidden_metrics": [(b["batch"], (b.get("prediction_metrics") or {}).get("primary"))
                           for b in hidden_rows],
        "final_checkpoint": str(parent_checkpoint),
        "history_rows": int(sum(len(x) for x in history_x)),
    }
    report_path = task_root / "stream_report.json"
    result.report_path = str(report_path)
    atomic_write_json(report_path, result.to_dict())
    return result.to_dict()


def run_stream(keys, source: str, n_batches: int, state_dir: str | Path,
               controller: str = "auto", llm_mode: str = "off", seed: int = 42,
               parallel_tasks: int = 1, memory: bool = True,
               experiments_cap: int = 1, batch_wall: float = 1800.0,
               retention_every: int = 2, **kwargs) -> dict:
    """Run one or more task streams and merge their reports."""

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for key in keys:
        payloads.append({
            "key": key, "source": source, "n_batches": n_batches, "seed": seed,
            "state_dir": str(state_dir), "controller": controller, "llm_mode": llm_mode,
            "memory": memory, "experiments_cap": experiments_cap, "batch_wall": batch_wall,
            "retention_every": retention_every, "verbose": True, **kwargs,
        })
    results: dict[str, dict] = {}
    jobs = max(1, int(parallel_tasks))
    if jobs > 1 and len(payloads) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(payloads))) as pool:
            futures = {pool.submit(run_task_stream, payload): payload["key"] for payload in payloads}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    results[key] = {"key": key, "errors": [f"{type(exc).__name__}: {exc}"], "batches": []}
    else:
        for payload in payloads:
            results[payload["key"]] = run_task_stream(payload)
    report = {
        "protocol": "zhixun-stream/2-honest",
        "source": source,
        "n_batches": n_batches,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tasks": results,
        "artifacts": {"state_dir": str(state_dir.resolve())},
    }
    atomic_write_json(state_dir / "stream_report.json", report)
    return report
