"""E0 protocol-invariance check for the honest private-sim stream.

Run the same two arrival batches twice: once with real answers, once with the
answers permuted across rows *after* the batches were built.  The harness never
sees the current batch answers, so ``decision.json`` and ``predictions.jsonl``
must be byte-identical between the two runs.

Usage: PYTHONPATH=. ./.venv/bin/python scripts/research/run_e0_invariance.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsi.stream import runner as stream_runner  # noqa: E402
from rsi.stream.sources import private_sim_batches  # noqa: E402
from rsi.tasks import resolve_tasks  # noqa: E402

OUT = Path(".scratch/ps-results/e0_invariance.json")


def _permute(batches):
    ids = sorted(batches[0].answers.keys()) if batches else []
    labels = [batches[0].answers[i] for i in ids]
    rng = random.Random(7)
    rng.shuffle(labels)
    permuted = dict(zip(ids, labels))
    return permuted


def _run(tag: str, batches, extra_answers=None):
    original = stream_runner.make_source
    payload_batches = batches

    def fake_make_source(*_args, **_kwargs):
        return payload_batches

    stream_runner.make_source = fake_make_source
    try:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            report = stream_runner.run_stream(
                keys=["A"], source="private-sim", n_batches=len(batches),
                state_dir=f".scratch/ps-e0/{tag}", arm="frozen", seed=42,
                controller="pipeline", llm_mode="off", memory=False,
                experiments_cap=1, retention_every=0,
            )
    finally:
        stream_runner.make_source = original
    return report


def main() -> int:
    config = resolve_tasks(["A"])[0]
    batches = private_sim_batches(config, root=None, arrival=(3, 4), hidden=(5, 5),
                                  dev_fraction=0.25, seed=42)
    permuted = _permute(batches)
    batches_permuted = [
        type(b)(index=b.index, train=b.train, dev=b.dev, prediction=b.prediction,
                answers=permuted if i == 0 else b.answers, reveal=b.reveal, meta=b.meta)
        for i, b in enumerate(batches)
    ]
    _run("real", batches)
    _run("permuted", batches_permuted)

    task_root = Path(".scratch/ps-e0")
    differences = []
    for batch in batches:
        idx = batch.index
        real = task_root / "real" / "tasks" / "A" / f"batch_{idx:04d}"
        perm = task_root / "permuted" / "tasks" / "A" / f"batch_{idx:04d}"
        for name in ("decision.json", "predictions.jsonl"):
            a = (real / name).read_bytes() if (real / name).exists() else b""
            b = (perm / name).read_bytes() if (perm / name).exists() else b""
            if a != b:
                differences.append({"batch": idx, "file": name})
    result = {"batches": len(batches), "differences": differences,
              "verdict": "PASS" if not differences else "FAIL"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
