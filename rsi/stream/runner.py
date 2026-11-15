"""Sequential batch-stream supervisor.

For every batch the harness:

1. materialises a tiny package (cumulative labelled train + current validation),
2. runs the *unchanged* ``rsi.runner.Runner`` for one round on that package,
3. compares the resulting model with the current parent on the batch's public
   selection split, and writes a promote / retain / rollback decision,
4. emits predictions for the current batch, then only advances the parent if
   the gate promoted, and
5. periodically re-evaluates the deployed model on all earlier batches to
   measure retention (catastrophic forgetting).

The important property is that nothing about the competition loop is bypassed:
the same Guard, memory, goal loop, records and report code path run, only the
package contents change per batch.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ..model import checkpoint_sha256
from ..runner import Runner
from ..tasks import TaskConfig, resolve_tasks
from .protocol import (
    atomic_write_json,
    checkpoint_of,
    decision_for,
    evaluate_rows,
    find_run_state,
    materialise_package,
    write_jsonl,
    write_predictions,
)
from .sources import make_source


@dataclass
class TaskStreamResult:
    key: str
    source: str
    batches: list[dict] = field(default_factory=list)
    retention: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    report_path: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "source": self.source,
            "batches": self.batches,
            "retention": self.retention,
            "errors": self.errors,
            "final": self.final,
            "report_path": self.report_path,
        }


def _read_primary(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text())
        return float(payload["primary"])
    except Exception:
        return None


def run_task_stream(payload: dict) -> dict:
    key = payload["key"]
    source_kind = payload.get("source", "official")
    n_batches = int(payload.get("n_batches", 6))
    seed = int(payload.get("seed", 42))
    state_dir = Path(payload["state_dir"])
    controller = payload.get("controller", "auto")
    llm_mode = payload.get("llm_mode", "off")
    memory = bool(payload.get("memory", True))
    experiments_cap = int(payload.get("experiments_cap", 40))
    batch_wall = float(payload.get("batch_wall", 1800.0))
    retention_every = int(payload.get("retention_every", 2))
    verbose = bool(payload.get("verbose", True))

    configs = resolve_tasks([key])
    if not configs:
        raise SystemExit(f"unknown task {key}")
    config: TaskConfig = configs[0]

    batch_source = make_source(
        source_kind, config, n_batches,
        external_root=payload.get("external_root"),
        external_start=payload.get("external_start", 3),
        external_end=payload.get("external_end", 10),
        holdout=payload.get("holdout", 0.3),
        seed=seed,
    )

    task_root = state_dir / "tasks" / key
    ckpt_dir = task_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = task_root / "memory"
    db_path = task_root / "stream.db"
    result = TaskStreamResult(key=key, source=source_kind)
    parent_checkpoint = Path(config.checkpoint_path)
    parent_postprocess: dict = {"type": "none"}
    parent_primary: float | None = None
    cumulative_rows = 0
    started = time.time()

    for batch in batch_source:
        index = batch.index
        batch_dir = task_root / f"batch_{index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_started = time.time()
        print(f"[stream:{key}] batch {index} source={source_kind} rows train={len(batch.train)} "
              f"val={len(batch.validation)} parent={parent_checkpoint.name}", flush=True)
        try:
            spec = materialise_package(batch_dir, config, batch, parent_checkpoint)
            runner = Runner(
                keys=(key,), rounds=1, llm_mode=llm_mode, seed=seed + index,
                state_dir=batch_dir, wall_cap=batch_wall, cpu_cap=36000.0,
                experiments_cap=experiments_cap, swanlab=False,
                run_dir=batch_dir / "run", verbose=verbose, controller=controller,
                memory=memory, memory_dir=str(memory_dir), db_path=str(db_path),
                round_turns=int(payload.get("round_turns", 20)),
                configs=[spec],
            )
            runner.run()
            info = find_run_state(batch_dir / "run")
            state = info.get("state") or {}
            summary = info.get("summary") or {}
            datasets = summary.get("datasets") or [{}]
            dataset_row = datasets[0] if datasets else {}
            task_state = (state.get("tasks") or [{}])[0]
            candidate_primary = (dataset_row.get("metrics") or {}).get("primary")
            candidate_checkpoint = task_state.get("checkpoint_path")
            candidate_postprocess = task_state.get("postprocess") or {"type": "none"}
            selection_path = batch_dir / "run" / key / "splits" / "selection.jsonl"
            if parent_primary is None:
                parent_primary = _read_primary(selection_path)  # noqa: F841 - replaced below
            parent_sel = evaluate_rows(
                config.task, selection_path, parent_checkpoint, parent_postprocess, config.classes,
            ) if selection_path.exists() else {"primary": parent_primary}
            parent_primary = parent_sel.get("primary")
            decision = decision_for(
                config.task, config.direction, parent_primary, candidate_primary, config.margin,
            )
            if decision["decision"] == "promote" and candidate_checkpoint:
                chosen_checkpoint = Path(candidate_checkpoint)
                chosen_postprocess = candidate_postprocess
                chosen_primary = candidate_primary
            else:
                chosen_checkpoint = parent_checkpoint
                chosen_postprocess = parent_postprocess
                chosen_primary = parent_primary
            suffix = decision["decision"]
            deployed = ckpt_dir / f"batch_{index:04d}_{suffix}.npz"
            shutil.copy2(chosen_checkpoint, deployed)
            predictions_path = batch_dir / "predictions.jsonl"
            validation_path = batch_dir / "package" / "public" / "validation.jsonl"
            n_predictions = write_predictions(
                validation_path, deployed, chosen_postprocess, config, predictions_path,
            )
            candidate_sha = (checkpoint_sha256(Path(candidate_checkpoint))
                             if candidate_checkpoint else None)
            record = {
                "batch": index,
                "round_index": index,
                # Flat fields: the most likely platform-facing shape (docs/19).
                "decision": decision["decision"],
                "decision_detail": decision,
                "delta": decision.get("delta"),
                "reason": decision.get("reason"),
                "parent_checkpoint_sha256": checkpoint_sha256(Path(parent_checkpoint)),
                "candidate_checkpoint_sha256": candidate_sha,
                "checkpoint_sha256": checkpoint_sha256(Path(deployed)),
                "validation_metrics": dataset_row.get("metrics"),
                "meta": batch.meta,
                "started_at": batch_started,
                "wall_seconds": round(time.time() - batch_started, 3),
                "parent": {"checkpoint": str(parent_checkpoint), "primary": parent_primary,
                           "postprocess": parent_postprocess},
                "candidate": {"checkpoint": candidate_checkpoint,
                              "primary": candidate_primary,
                              "postprocess": candidate_postprocess,
                              "selection": dataset_row.get("metrics")},
                "chosen": {"checkpoint": str(deployed), "primary": chosen_primary,
                           "postprocess": chosen_postprocess},
                "predictions": {"count": n_predictions, "path": str(predictions_path)},
                "run_dir": str(batch_dir / "run"),
                "run_summary": str(batch_dir / "run" / "summary.json"),
                "controller": state.get("run", {}).get("agent", {}).get("controller"),
                "degraded": task_state.get("degraded"),
                "batch_round": (task_state.get("rounds") or [{}])[-1] if task_state.get("rounds") else {},
            }
            atomic_write_json(batch_dir / "decision.json", record)
            result.batches.append(record)
            parent_checkpoint = deployed
            parent_postprocess = chosen_postprocess
            parent_primary = chosen_primary
            cumulative_rows = len(batch.train)
        except Exception as exc:  # keep the stream alive: record and stop this task
            error = f"batch {index} failed: {type(exc).__name__}: {exc}"
            result.errors.append(error)
            traceback.print_exc()
            break

        if retention_every > 0 and (index + 1) % retention_every == 0:
            for previous in range(index + 1):
                prev_val = task_root / f"batch_{previous:04d}" / "package" / "public" / "validation.jsonl"
                if not prev_val.exists():
                    continue
                try:
                    metrics = evaluate_rows(
                        config.task, prev_val, parent_checkpoint, parent_postprocess, config.classes,
                    )
                except Exception as exc:
                    metrics = {"error": f"{type(exc).__name__}: {exc}"}
                item = {"batch": index, "evaluated_batch": previous,
                        "primary": metrics.get("primary"), "metrics": metrics,
                        "checkpoint": str(parent_checkpoint)}
                result.retention.append(item)
                with (task_root / "retention.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    first = result.batches[0] if result.batches else {}
    last = result.batches[-1] if result.batches else {}
    result.final = {
        "batches": len(result.batches),
        "errors": len(result.errors),
        "wall_seconds": round(time.time() - started, 3),
        "first_primary": (first.get("chosen") or {}).get("primary"),
        "final_primary": (last.get("chosen") or {}).get("primary"),
        "first_parent_primary": (first.get("parent") or {}).get("primary"),
        "final_checkpoint": str(parent_checkpoint),
        "final_postprocess": parent_postprocess,
        "cumulative_train_rows": cumulative_rows,
        "deployed_batches": sum(1 for b in result.batches if b.get("decision") == "promote"),
    }
    report_path = task_root / "stream_report.json"
    result.report_path = str(report_path)
    atomic_write_json(report_path, result.to_dict())
    return result.to_dict()


def run_stream(keys, source: str, n_batches: int, state_dir: str | Path,
               controller: str = "auto", llm_mode: str = "off", seed: int = 42,
               parallel_tasks: int = 1, memory: bool = True,
               experiments_cap: int = 40, batch_wall: float = 1800.0,
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
        "protocol": "zhixun-stream/1",
        "source": source,
        "n_batches": n_batches,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tasks": results,
        "artifacts": {"state_dir": str(state_dir.resolve())},
    }
    atomic_write_json(state_dir / "stream_report.json", report)
    return report
