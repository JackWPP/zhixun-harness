# Porting the RSI harness to a new task / dataset

The runtime is dataset-agnostic: a task is declared by a small `task.json` plus
a package directory in the official layout. No Python file has to change.

## 1. Package contract

```
my-task/
  task.json                      # declaration (see below)
  public/train.jsonl             # labelled rows, official schema
  public/validation.jsonl        # labelled rows, official schema
  checkpoint.npz                 # parent checkpoint (continuation starting point)
```

Rows are `{"id": str, "inputs": {...}, "answer": ...}` exactly as in the official
packages; `rsi/data_official.py` performs all structural validation (unique ids,
numeric finite features, no answer/lifetime fields inside `inputs`, size caps).
The checkpoint may be produced by `baselines.train` (competition: always start
from the official baseline/previous-round checkpoint) or, for a new private
dataset, by a scratch training run (see `scripts/make_demo_task.py`).

## 2. `task.json`

```json
{
  "key": "D",
  "name": "demo",
  "task": "classification",            // or "regression"
  "classes": ["a", "b", "c"],          // required for classification
  "package": ".",                      // package dir relative to data_root
  "data_root": ".",                    // resolved relative to this JSON file
  "primary_metric": "macro_f1",        // macro_f1 | accuracy | rmse | mae | nasa_score
  "direction": "higher",               // higher | lower
  "baseline_primary": 0.80,            // documented static baseline (display/gate reference)
  "margin": 0.01,                      // promotion margin in primary units
  "rounds": 10,
  "split_strategy": "auto",            // auto | stratified | group | units
  "probe": "none",                     // none | uci_gas
  "train_file": "public/train.jsonl",
  "validation_file": "public/validation.jsonl",
  "checkpoint_file": "checkpoint.npz"
}
```

`split_strategy: auto` = class-stratified rows for classification datasets,
whole-group splits when the validation set has >= 10 groups (engines/bearings),
otherwise within-group row splits. Validation is only ever split into
calibration/selection/report; it is **never fitted** (Guard enforces this).

## 3. Run

```bash
# register one or more extra specs (file or directory), then run by key
python -m rsi.cli run --task-spec tasks/D/task.json --tasks D \
    --rounds 10 --llm auto --state-dir /work/rsi-runs

# equivalent environment-variable interface used by the Docker image
RSI_TASK_SPECS=/work/specs RSI_TASKS=D,E python -m rsi.cli run
```

`python -m rsi.cli tasks` lists everything registered (built-in A/B/C + specs).

Proof that this works end-to-end: `scripts/make_demo_task.py` generates a
synthetic fourth task and `tests/test_portable.py` runs a full honest round on
it (no rsi/ edits, no official data).

## 4. Reuse on the competition platform

The image is a general RSI worker, not an A/B/C script:

1. Mount the round's packages under `/work/specs/<key>/` (task.json + data +
   parent checkpoint). The official A/B/C packages are pre-registered in
   `/opt/zhixun/app/tasks/` and read from `RSI_DATA_ROOT`.
2. Set `RSI_TASK_SPECS`, `RSI_TASKS`, `RSI_ROUNDS`, `RSI_LLM` and (optionally)
   `RSI_TOKEN_CAP` / `RSI_CONTROLLER`.
3. The container writes, per task: `state.json` + `rounds.jsonl` +
   `workspace_report.json` (platform schema) + the selected checkpoint and its
   `postprocess.json`. `scripts/make_submission.py` packages them under 64 MiB.
4. Hidden streaming evaluation uses the independent prediction entry:
   `python -m rsi.predict --checkpoint <ckpt> --postprocess <json> --mode batch`
   (reads all rows, applies the frozen postprocess once, answers in order).

## 5. Structure changes / weight migration

Changing width/inputs is allowed by the competition **only with an explicit
compatible weight migration** (TRAINING_GUIDE §4; announcement "不兼容提交请基于
官方 baseline 修改"). The harness keeps a strict loader and lineage: a migrated
checkpoint records the parent sha, candidate sha and code sha in every round
record; the default runtime only uses strategies whose checkpoints stay
loadable by the reference contract. Planned migration strategies
(`rsi/strategies.py`) will be opt-in and never silently reinitialise weights.
