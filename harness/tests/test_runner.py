"""Real-package smoke tests for orchestration, RuleDrivenHarnessTrainer, StrategyRecommenderMLP, fail-closed gates and HTML output."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from harness.data import digest, digest_bytes, import_model
from harness.reporting import write_report
from harness.runner import run

ARCHIVE = Path(os.environ.get("HARNESS_TEST_VENDOR_ARCHIVE", "C:/Users/zhao jiahao/Downloads/cpu-a-guide-20261110-v1.zip"))


def _ensure_test_archive(temp_dir: Path) -> Path:
    if ARCHIVE.is_file():
        return ARCHIVE
    # 当跨机器环境中不存在硬编码 Downloads 路径时，使用原包真实的 baselines/* 源码构造合规的测试归档包
    here = Path(__file__).resolve()
    source_dir = None
    for parent in here.parents:
        for cand in (
            parent / "harness-runs" / "A-v1" / "source",
            parent / "unzipped" / "harness-runs" / "A-v1" / "source",
        ):
            if (cand / "baselines" / "model.py").is_file():
                source_dir = cand
                break
        if source_dir is not None:
            break
    if source_dir is None:
        raise FileNotFoundError(f"Neither {ARCHIVE} nor source/baselines/model.py found")

    Model = import_model(source_dir)
    classes = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
    paths = [json.dumps([["key", "features"], ["index", i]], separators=(",", ":")) for i in range(8)]
    rng = np.random.default_rng(42)
    x_tr = rng.normal(0.0, 1.0, (24, 8))
    y_tr = np.array([i % 6 for i in range(24)], dtype=np.int64)
    model = Model("classification", paths, classes, x_tr, y_tr, hidden=12, seed=42, learning_rate=0.003, batch_size=8)
    model.meta["profile"] = "A"
    model.fit(x_tr, y_tr, 2)
    ckpt_path = temp_dir / "checkpoint.npz"
    model.save(ckpt_path)

    train_rows = [
        {
            "id": f"tr-{i}",
            "inputs": {"features": [float(v) for v in x_tr[i]]},
            "answer": classes[int(y_tr[i])],
            "meta": {"source_batch": "batch-1" if i < 12 else "batch-2"},
        }
        for i in range(24)
    ]
    x_va = rng.normal(0.1, 1.0, (18, 8))
    y_va = np.array([i % 6 for i in range(18)], dtype=np.int64)
    val_rows = [
        {
            "id": f"va-{i}",
            "inputs": {"features": [float(v) for v in x_va[i]]},
            "answer": classes[int(y_va[i])],
            "meta": {"source_batch": "batch-3"},
        }
        for i in range(18)
    ]
    train_bytes = ("\n".join(json.dumps(r) for r in train_rows) + "\n").encode("utf-8")
    val_bytes = ("\n".join(json.dumps(r) for r in val_rows) + "\n").encode("utf-8")

    baselines_files = {}
    for name in ("model.py", "data.py", "train.py", "predict.py"):
        baselines_files[name] = (source_dir / "baselines" / name).read_bytes()

    training_report = {
        "profile": "A",
        "task": "classification",
        "official_scoring_ready": False,
        "checkpoint_sha256": digest(ckpt_path),
        "train_sha256": digest_bytes(train_bytes),
        "validation_sha256": digest_bytes(val_bytes),
        "source_sha256": {name: digest_bytes(raw) for name, raw in baselines_files.items()},
    }

    archive_path = temp_dir / "synthetic-cpu-a.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("checkpoint.npz", ckpt_path.read_bytes())
        zf.writestr("training.json", json.dumps(training_report).encode("utf-8"))
        zf.writestr("baseline-info.json", (source_dir / "baseline-info.json").read_bytes())
        zf.writestr("public/train.jsonl", train_bytes)
        zf.writestr("public/validation.jsonl", val_bytes)
        for name, raw in baselines_files.items():
            zf.writestr("baselines/" + name, raw)
        zf.writestr("baselines/configs/A.json", (source_dir / "baselines" / "configs" / "A.json").read_bytes())
    return archive_path


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.archive_path = _ensure_test_archive(Path(cls.fixture_temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_run_lineage_and_delivered_metrics(self):
        source_sha = digest(self.archive_path)
        result = run("A", self.archive_path, self.root / "run", epochs=2, max_candidates=2, rounds=2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["configuration"]["trainer_engine"], "RuleDrivenHarnessTrainer")
        expected_parent = result["baseline"]["checkpoint_sha256"]
        incumbent = result["baseline"]["metrics"]["macro_f1"]
        for record in result["rounds"]:
            self.assertEqual(record["parent_checkpoint_sha256"], expected_parent)
            for candidate in record["candidates"]:
                self.assertEqual(candidate["parent_checkpoint_sha256"], expected_parent)
                self.assertEqual(candidate["optimizer_training_sha256"], result["source"]["train_sha256"])
                self.assertEqual(candidate["evaluation_validation_sha256"], result["source"]["validation_sha256"])
                self.assertGreater(candidate["selected_steps"], candidate["initial_steps"])
            decision = record["decision"]
            self.assertGreaterEqual(decision["metrics_after"]["macro_f1"], incumbent)
            incumbent = decision["metrics_after"]["macro_f1"]
            expected_parent = decision["selected_checkpoint_sha256"]
        self.assertEqual(result["selected"]["checkpoint_sha256"], expected_parent)
        self.assertEqual(digest(result["selected"]["checkpoint"]), expected_parent)
        self.assertEqual(digest(self.archive_path), source_sha)
        self.assertTrue((self.root / "run/experience_trajectories.json").is_file())
        report = (self.root / "run/report.html").read_text(encoding="utf-8")
        self.assertIn("验证集", report)
        self.assertIn("selected/checkpoint.npz", report)
        self.assertEqual(json.loads((self.root / "run/summary.json").read_text(encoding="utf-8"))["status"], "completed")

    def test_rejection_keeps_baseline_and_stops_further_rounds(self):
        result = run("A", self.archive_path, self.root / "rejected", epochs=1, max_candidates=2, rounds=3, min_improvement=1.0)
        self.assertEqual(len(result["rounds"]), 1)
        self.assertFalse(result["rounds"][0]["decision"]["accepted"])
        self.assertEqual(result["selected"]["checkpoint_sha256"], result["baseline"]["checkpoint_sha256"])
        self.assertEqual(result["selected"]["improvement"], 0.0)

    def test_same_seed_produces_same_delivered_weights(self):
        a = run("A", self.archive_path, self.root / "a", epochs=2, max_candidates=2)
        b = run("A", self.archive_path, self.root / "b", epochs=2, max_candidates=2)
        self.assertEqual(a["selected"]["checkpoint_sha256"], b["selected"]["checkpoint_sha256"])
        self.assertEqual(a["selected"]["metrics"], b["selected"]["metrics"])

    def test_budget_exhaustion_keeps_original_if_no_full_epoch(self):
        result = run("A", self.archive_path, self.root / "budget", epochs=2, max_candidates=2, max_train_seconds=1e-12)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["selected"]["checkpoint_sha256"], result["baseline"]["checkpoint_sha256"])
        self.assertFalse(result["rounds"][0]["decision"]["accepted"])

    def test_all_errors_are_not_reported_as_success(self):
        with mock.patch("harness.training.train_epoch", side_effect=RuntimeError("controlled failure")):
            result = run("A", self.archive_path, self.root / "failed", epochs=1, max_candidates=2)
        self.assertEqual(result["status"], "failed")
        self.assertIn("所有候选", result["error"])
        self.assertEqual(result["selected"]["checkpoint_sha256"], result["baseline"]["checkpoint_sha256"])

    def test_small_model_and_hybrid_decision_modes(self):
        res_sm = run("A", self.archive_path, self.root / "sm_run", epochs=1, max_candidates=2, decision_mode="small_model")
        self.assertEqual(res_sm["status"], "completed")
        self.assertEqual(res_sm["configuration"]["decision_mode"], "small_model")
        res_hyb = run("A", self.archive_path, self.root / "hyb_run", epochs=1, max_candidates=2, decision_mode="hybrid")
        self.assertEqual(res_hyb["status"], "completed")
        self.assertEqual(res_hyb["configuration"]["decision_mode"], "hybrid")

    def test_html_escapes_injected_text_and_handles_failed_run(self):
        result = {"status": "failed", "finished_at": "done", "task": "A", "run_id": '<script>alert("x")</script>',
                  "error": "controlled & <failure>", "warnings": ["<img onerror=alert(1)>"]}
        path = write_report(result, self.root)
        html = path.read_text(encoding="utf-8")
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img onerror", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;failure&gt;", html)

    def test_unfinished_report_is_not_published(self):
        with self.assertRaises(ValueError):
            write_report({"status": "running"}, self.root)
        self.assertFalse((self.root / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
