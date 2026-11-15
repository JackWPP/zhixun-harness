"""使用原始模型和合成数据验证离线续训边界与 RuleDrivenHarnessTrainer 扩展组件。"""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

from harness.data import Dataset, digest, digest_bytes, import_model, prepare_bundle
from harness.diagnostics import diagnose, metric_values
from harness.runner import select_candidate
from harness.strategies import (
    Strategy,
    StrategyRecommenderMLP,
    choose_strategies,
    diagnosis_to_state_vector,
    rule_select_candidates,
)
from harness.training import (
    BudgetExhausted,
    CostLedger,
    DiagnosticProbe,
    ExperienceBuffer,
    RuleDrivenHarnessTrainer,
    TrajectoryRecord,
    get_default_task_specs,
    run_candidate,
    train_epoch,
)


VENDOR_ARCHIVE = Path(os.environ.get("HARNESS_TEST_VENDOR_ARCHIVE", "C:/Users/zhao jiahao/Downloads/cpu-a-guide-20261110-v1.zip"))
PATHS = [json.dumps([["key", "a"]], separators=(",", ":")),
         json.dumps([["key", "b"]], separators=(",", ":"))]


def _load_vendor_model_bytes() -> bytes:
    if VENDOR_ARCHIVE.is_file():
        with zipfile.ZipFile(VENDOR_ARCHIVE) as archive:
            return archive.read("baselines/model.py")
    # 自动回退：从相邻的 harness-runs/*/source/baselines/model.py 读取原始模型源码
    here = Path(__file__).resolve()
    for parent in here.parents:
        for candidate_path in (
            parent / "harness-runs" / "A-v1" / "source" / "baselines" / "model.py",
            parent / "unzipped" / "harness-runs" / "A-v1" / "source" / "baselines" / "model.py",
        ):
            if candidate_path.is_file():
                return candidate_path.read_bytes()
    raise FileNotFoundError(f"Vendor archive or baselines/model.py not found (checked {VENDOR_ARCHIVE})")


def dataset(task="regression", role="train"):
    x = np.array([[0., 2.], [1., 5.], [2., 1.], [4., 3.], [8., 7.],
                  [3., 9.], [7., 4.], [5., 6.], [9., 8.]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2], dtype=np.int64) if task == "classification" else np.array([3., 7., 2., 8., 15., 4., 9., 6., 12.])
    x.flags.writeable = y.flags.writeable = False
    return Dataset(x, y, tuple(role + str(i) for i in range(len(x))),
                   tuple(role + g for g in ("a", "a", "a", "a", "a", "b", "b", "c", "c")),
                   tuple({} for _ in x), tuple(PATHS), digest_bytes(x.tobytes() + y.tobytes()), role)


def strategy(**changes):
    return replace(Strategy("test", "合成测试", "验证续训状态", .003, 4, 3), **changes)


def candidate(value, task="B", **changes):
    result = {"id": "synthetic", "status": "completed", "gate": {"eligible": True},
              "metrics": {"macro_f1" if task == "A" else "rmse": value},
              "best_checkpoint": "synthetic.npz"}
    result.update(changes)
    return result


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vendor_temp = tempfile.TemporaryDirectory()
        cls.vendor_root = Path(cls.vendor_temp.name)
        (cls.vendor_root / "baselines").mkdir()
        # 原始包只读取模型源代码，所有数据和检查点均在临时目录自行生成。
        cls.vendor_code = _load_vendor_model_bytes()
        (cls.vendor_root / "baselines/model.py").write_bytes(cls.vendor_code)
        cls.Model = import_model(cls.vendor_root)

    @classmethod
    def tearDownClass(cls):
        cls.vendor_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def model(self, task="regression"):
        data = dataset(task)
        model = self.Model(task, PATHS, ["zero", "one", "two"] if task == "classification" else [],
                           data.x, data.y, hidden=5, seed=1729, batch_size=4)
        model.meta["profile"] = "A" if task == "classification" else "B"
        if task == "regression":
            model.meta["target_unit"] = "minutes"
        return model

    def assert_state_equal(self, left, right):
        self.assertEqual(left.meta, right.meta)
        for group in ("params", "first", "second"):
            for key in getattr(left, group):
                np.testing.assert_array_equal(getattr(left, group)[key], getattr(right, group)[key])
        np.testing.assert_array_equal(left.mean, right.mean)
        np.testing.assert_array_equal(left.scale, right.scale)

    def bundle(self):
        model = self.model()
        train = dataset()
        model.fit(train.x, train.y, 2)
        parent = self.root / "parent.npz"
        model.save(parent)
        return SimpleNamespace(task="B", model_class=self.Model, train=train,
                               validation=dataset(role="validation"), checkpoint=parent)

    def synthetic_archive(self, *, corrupt_digest=False, wrong_model_task=False):
        model = self.model("regression" if wrong_model_task else "classification")
        model.meta["profile"] = "A"
        checkpoint = self.root / "fixture-checkpoint.npz"
        model.save(checkpoint)
        members = {"checkpoint.npz": checkpoint.read_bytes(), "baseline-info.json": b"{}",
                   "baselines/model.py": self.vendor_code,
                   "baselines/data.py": b"", "baselines/train.py": b"", "baselines/predict.py": b"",
                   "baselines/configs/A.json": b"{}", "baselines/configs/B.json": b"{}"}
        for role in ("train", "validation"):
            rows = [{"id": role + str(i), "inputs": {"a": float(i), "b": float(i + 1)},
                     "answer": float(i + 1) if wrong_model_task else ["zero", "one", "two"][i],
                     "meta": {"source_batch": role + "-batch"}} for i in range(3)]
            members["public/" + role + ".jsonl"] = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        report = {"profile": "A", "checkpoint_sha256": digest_bytes(members["checkpoint.npz"]),
                  "train_sha256": digest_bytes(members["public/train.jsonl"]),
                  "validation_sha256": digest_bytes(members["public/validation.jsonl"]),
                  "source_sha256": {"model.py": digest_bytes(self.vendor_code)}}
        if corrupt_digest:
            report["train_sha256"] = "sha256:" + "0" * 64
        members["training.json"] = json.dumps(report).encode()
        path = self.root / "fixture.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        return path

    def test_plain_epoch_matches_original_fit_exactly(self):
        for task in ("classification", "regression"):
            with self.subTest(task=task):
                data = dataset(task)
                expected = self.model(task)
                expected.fit(data.x, data.y, 2)
                actual = copy.deepcopy(expected)
                losses = expected.fit(data.x, data.y, 3)
                observed = [train_epoch(actual, data, strategy()) for _ in range(3)]
                np.testing.assert_array_equal(observed, losses)
                self.assert_state_equal(actual, expected)
                self.assertEqual(actual.meta["steps"], 15)
                self.assertEqual(actual.meta["epochs"], 5)

    def test_validation_role_rejected_without_any_updates(self):
        model = self.model()
        before = copy.deepcopy(model)
        with self.assertRaisesRegex(ValueError, "train"):
            train_epoch(model, dataset(role="validation"), strategy())
        self.assert_state_equal(model, before)

    def test_original_scalers_frozen_under_custom_training(self):
        cases = [("classification", strategy(noise_std=.07, balanced_classes=True, l2=.002)),
                 ("regression", strategy(noise_std=.07, balanced_groups=True, l2=.002))]
        for task, action in cases:
            with self.subTest(task=task):
                model = self.model(task)
                before = copy.deepcopy(model)
                data = dataset(task)
                shifted = replace(data, x=data.x * 17 + 100, y=data.y if task == "classification" else data.y * 4 + 80)
                for _ in range(2):
                    train_epoch(model, shifted, action)
                np.testing.assert_array_equal(model.mean, before.mean)
                np.testing.assert_array_equal(model.scale, before.scale)
                for key in ("target_mean", "target_scale"):
                    self.assertEqual(model.meta[key], before.meta[key])
                self.assertTrue(any(not np.array_equal(model.params[k], before.params[k]) for k in model.params))

    def test_custom_training_deterministic_across_save_resume(self):
        for task in ("classification", "regression"):
            with self.subTest(task=task):
                data = dataset(task)
                action = strategy(noise_std=.05, balanced_classes=task == "classification",
                                  balanced_groups=task == "regression", l2=.001)
                continuous, resumed = self.model(task), self.model(task)
                for epoch in range(3):
                    self.assertEqual(train_epoch(continuous, data, action), train_epoch(resumed, data, action))
                    path = self.root / (task + str(epoch) + ".npz")
                    resumed.save(path)
                    resumed = self.Model.load(path)
                self.assert_state_equal(continuous, resumed)

    def test_zero_epoch_budget_prevents_updates(self):
        model = self.model()
        before = copy.deepcopy(model)
        with self.assertRaises(BudgetExhausted):
            train_epoch(model, dataset(), strategy(), max_seconds=0)
        self.assert_state_equal(model, before)

    def test_exhausted_candidate_budget_creates_no_checkpoint(self):
        bundle = self.bundle()
        before = digest(bundle.checkpoint)
        ledger = CostLedger(1)
        ledger.training_seconds = 1
        with mock.patch.object(ledger, "evaluate") as evaluate:
            result = run_candidate(bundle, bundle.checkpoint, strategy(), self.root / "empty", ledger, lambda *args: None)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["gate"]["eligible"])
        self.assertIsNone(result["best_checkpoint"])
        self.assertEqual(result["cost"]["steps"], 0)
        evaluate.assert_not_called()
        self.assertEqual(digest(bundle.checkpoint), before)

    def test_candidate_restores_best_epoch_and_continues_loaded_adam(self):
        bundle = self.bundle()
        before = digest(bundle.checkpoint)
        expected = self.Model.load(bundle.checkpoint)
        self.assertTrue(any(np.any(value) for value in expected.first.values()))
        train_epoch(expected, bundle.train, strategy())
        ledger = CostLedger(60)
        with mock.patch.object(ledger, "evaluate", side_effect=[{"rmse": 1.}, {"rmse": 4.}, {"rmse": 3.}]):
            result = run_candidate(bundle, bundle.checkpoint, strategy(), self.root / "best", ledger, lambda *args: None, patience=3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["trainer_engine"], "RuleDrivenHarnessTrainer")
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(len(result["diagnostic_trace"]), 3)
        self.assertEqual(result["restored_best_epoch"], 3)
        self.assertEqual(result["selected_steps"], 9)
        self.assertEqual(result["cost"]["steps"], 9)
        self.assertEqual(result["metrics"], {"rmse": 1.})
        self.assert_state_equal(self.Model.load(result["best_checkpoint"]), expected)
        self.assertEqual(digest(bundle.checkpoint), before)
        self.assertEqual(result["parent_checkpoint_sha256"], before)
        self.assertEqual(result["checkpoint_sha256"], digest(result["best_checkpoint"]))

    def test_candidate_failure_after_valid_epoch_is_ineligible(self):
        bundle = self.bundle()
        ledger = CostLedger(60)
        before = digest(bundle.checkpoint)
        with mock.patch.object(ledger, "evaluate", side_effect=[{"rmse": 1.}, RuntimeError("受控评估失败")]):
            result = run_candidate(bundle, bundle.checkpoint, strategy(), self.root / "failed", ledger, lambda *args: None)
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["history"]), 1)
        self.assertFalse(result["gate"]["eligible"])
        self.assertIsNone(result["best_checkpoint"])
        self.assertIsNone(select_candidate("B", {"rmse": 10.}, [result]))
        self.assertEqual(digest(bundle.checkpoint), before)

    def test_gate_rejects_regression_degradation(self):
        for task in ("B", "C"):
            with self.subTest(task=task):
                result = candidate(11.)
                self.assertIsNone(select_candidate(task, {"rmse": 10.}, [result]))
                self.assertFalse(result["gate"]["passes_primary_threshold"])

    def test_gate_rejects_classification_degradation(self):
        result = candidate(.4, "A")
        self.assertIsNone(select_candidate("A", {"macro_f1": .5}, [result]))
        self.assertLess(result["gate"]["improvement"], 0)

    def test_gate_requires_strict_threshold(self):
        for task, baseline, value, threshold in (("B", {"rmse": 10.}, 9., 1.),
                                                  ("A", {"macro_f1": .5}, .75, .25),
                                                  ("B", {"rmse": 10.}, 10., 0.)):
            with self.subTest(task=task, threshold=threshold):
                self.assertIsNone(select_candidate(task, baseline, [candidate(value, task)], threshold))

    def test_gate_selects_best_and_keeps_first_on_ties(self):
        for task, baseline, scores in (("B", {"rmse": 10.}, [9., 7., 7., 8.]),
                                       ("A", {"macro_f1": .25}, [.5, .875, .875, .75])):
            with self.subTest(task=task):
                options = [candidate(value, task) for value in scores]
                self.assertIs(select_candidate(task, baseline, options), options[1])

    def test_gate_rejects_ineligible_and_missing_checkpoints(self):
        options = [candidate(0., gate={"eligible": False}), candidate(0., best_checkpoint=None),
                   candidate(0., metrics={})]
        self.assertIsNone(select_candidate("B", {"rmse": 10.}, options))

    def test_gate_error_status_cannot_be_selected_even_with_stale_gate(self):
        failed = candidate(0., status="error")
        self.assertIsNone(select_candidate("B", {"rmse": 10.}, [failed]))

    def test_gate_rejects_nonfinite_gains(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(select_candidate("B", {"rmse": 10.}, [candidate(value)]))

    def test_prepare_wrong_profile_does_not_change_source(self):
        archive = self.synthetic_archive()
        before = archive.read_bytes()
        destination = self.root / "wrong-profile"
        with self.assertRaises(ValueError):
            prepare_bundle(archive, "B", destination)
        self.assertEqual(archive.read_bytes(), before)
        self.assertFalse(destination.exists())

    def test_prepare_wrong_model_task_does_not_change_source(self):
        archive = self.synthetic_archive(wrong_model_task=True)
        before = archive.read_bytes()
        with self.assertRaisesRegex(ValueError, "task"):
            prepare_bundle(archive, "A", self.root / "wrong-task")
        self.assertEqual(archive.read_bytes(), before)

    def test_prepare_incorrect_digest_fails_before_staging(self):
        archive = self.synthetic_archive(corrupt_digest=True)
        before = archive.read_bytes()
        destination = self.root / "bad-digest"
        with self.assertRaisesRegex(ValueError, "integrity"):
            prepare_bundle(archive, "A", destination)
        self.assertEqual(archive.read_bytes(), before)
        self.assertFalse(destination.exists())

    def test_prepare_valid_bundle_has_isolated_readonly_splits(self):
        archive = self.synthetic_archive()
        before = archive.read_bytes()
        bundle = prepare_bundle(archive, "A", self.root / "valid")
        self.assertEqual(bundle.train.role, "train")
        self.assertEqual(bundle.validation.role, "validation")
        self.assertFalse(set(bundle.train.ids) & set(bundle.validation.ids))
        self.assertFalse(set(bundle.train.groups) & set(bundle.validation.groups))
        for split in (bundle.train, bundle.validation):
            self.assertFalse(split.x.flags.writeable)
            self.assertFalse(split.y.flags.writeable)
            np.testing.assert_array_equal(split.y, [0, 1, 2])
            np.testing.assert_array_equal(split.x, [[0., 1.], [1., 2.], [2., 3.]])
        self.assertFalse(np.shares_memory(bundle.train.x, bundle.validation.x))
        self.assertEqual(digest(bundle.checkpoint), bundle.source["checkpoint_sha256"])
        self.assertEqual(archive.read_bytes(), before)

    def test_diagnosis_is_read_only_and_validation_is_not_fitted(self):
        model = self.model()
        before = copy.deepcopy(model)
        train = dataset()
        validation = dataset(role="validation")
        validation = replace(validation, x=validation.x * 30 + 1000, y=validation.y + 100)
        report = diagnose(model, train, validation, "B")
        self.assert_state_equal(model, before)
        self.assertEqual(report["validation_metrics"], metric_values(model, validation))
        reference = report["details"]["constant_training_mean_reference"]
        self.assertEqual(reference["value"], float(train.y.mean()))
        self.assertEqual(reference["fit_role"], "train")
        self.assertAlmostEqual(reference["validation_rmse"], float(np.sqrt(np.mean((validation.y - train.y.mean()) ** 2))))

    def test_rule_driven_trainer_asymmetric_rul_and_probe(self):
        model = self.model("regression")
        data = dataset("regression")
        trainer = RuleDrivenHarnessTrainer(patience=2)
        norms = []
        loss = trainer.execute_epoch(
            model,
            data,
            strategy(asymmetric_rul_alpha=0.25, l2=0.001),
            grad_norms_out=norms,
        )
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(len(norms), 0)
        probe = DiagnosticProbe(task="B")
        state = probe.record_epoch(
            epoch=model.meta["epochs"],
            train_loss=loss,
            validation_metrics={"rmse": 5.0, "overestimate_fraction": 0.6},
            grad_norms=norms,
            plateau_count=0,
        )
        vec = state.to_vector()
        self.assertEqual(vec.shape, (12,))
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_strategy_recommender_mlp_and_rule_select_candidates(self):
        specs = get_default_task_specs()
        self.assertEqual(set(specs.keys()), {"A", "B", "C"})
        model_a = self.model("classification")
        diag_a = diagnose(model_a, dataset("classification"), dataset("classification", role="validation"), "A")
        rule_cands = rule_select_candidates("A", model_a, diag_a, epochs=3, max_candidates=3)
        self.assertGreaterEqual(len(rule_cands), 2)

        buf = ExperienceBuffer()
        vec_a = diagnosis_to_state_vector("A", model_a, diag_a).tolist()
        buf.add(TrajectoryRecord("A", 1, "02-class_weighted", "class_weighted", vec_a, 0.02, True, True))
        buf.add(TrajectoryRecord("B", 1, "03-regularized", "regularized", [0.5] * 12, 20.0, True, True))
        buf.add(TrajectoryRecord("C", 1, "01-conservative", "conservative", [1.0] * 12, 1.1, True, True))

        sm = StrategyRecommenderMLP(seed=42)
        stats = sm.fit_from_buffer(buf, epochs=60)
        self.assertEqual(stats["trained_samples"], 3)
        save_path = self.root / "sm.npz"
        sm.save(save_path)
        reloaded = StrategyRecommenderMLP.load(save_path)
        sm_cands = choose_strategies("A", model_a, diag_a, epochs=3, max_candidates=3, decision_mode="small_model", strategy_model=reloaded)
        self.assertEqual(len(sm_cands), 3)
        self.assertTrue(all(c.id not in ("device_balanced", "penalize_late_rul") for c in sm_cands))


if __name__ == "__main__":
    unittest.main()
