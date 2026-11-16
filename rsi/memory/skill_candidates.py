"""Versioned skill candidates that cannot affect the active library in-place."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
from pathlib import Path

from .skills import SkillStore


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path) -> dict[str, str]:
    return {f"{name}/SKILL.md": _file_hash(root / name / "SKILL.md")
            for name in sorted(SkillStore(root).names())}


class SkillCandidateStore:
    """Stage edits against a frozen active version and require a gated eval.

    The agent's ``create/patch/delete`` tools call this store. Reads continue to
    use the active ``SkillStore``. Promotion is deliberately an evaluator-side
    API and requires two or more independent seeds plus a hash-matched report.
    """

    def __init__(self, active: SkillStore, root: str | Path, *, max_candidates: int = 3):
        self.active = active
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_candidates = max(1, int(max_candidates))

    def _manifests(self) -> list[Path]:
        return sorted(self.root.glob("candidate-*/manifest.json"))

    @property
    def candidate_ids(self) -> list[str]:
        return [path.parent.name for path in self._manifests()]

    def _stage(self, action: str, **kwargs) -> dict:
        manifests = self._manifests()
        if len(manifests) >= self.max_candidates:
            raise ValueError(f"skill candidate cap reached ({self.max_candidates})")
        source_candidate = None
        if manifests:
            previous = json.loads(manifests[-1].read_text(encoding="utf-8"))
            if previous.get("status") == "staged":
                source_candidate = previous["candidate_id"]
                source_root = self.root / source_candidate / "library"
            else:
                source_root = self.active.root
        else:
            source_root = self.active.root

        parent_hashes = _hashes(self.active.root)
        candidate_id = f"candidate-{len(manifests) + 1:03d}-{uuid.uuid4().hex[:8]}"
        candidate_root = self.root / candidate_id
        library = candidate_root / "library"
        shutil.copytree(source_root, library, ignore=shutil.ignore_patterns(".archive"))
        staged = SkillStore(library)
        result = getattr(staged, action)(**kwargs)
        candidate_hashes = _hashes(library)
        manifest = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "created_at": time.time(),
            "status": "staged",
            "action": action,
            "source_candidate_id": source_candidate,
            "parent_skill_hashes": parent_hashes,
            "candidate_skill_hashes": candidate_hashes,
            "candidate_digest": _hash_json(candidate_hashes),
            "result": result,
        }
        (candidate_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {**result, "candidate_id": candidate_id,
                "candidate_digest": manifest["candidate_digest"],
                "status": "staged", "parent_skill_hashes": parent_hashes,
                "candidate_skill_hashes": candidate_hashes}

    def create(self, name: str, text: str) -> dict:
        return self._stage("create", name=name, text=text)

    def patch(self, name: str, old_string: str, new_string: str) -> dict:
        return self._stage("patch", name=name, old_string=old_string, new_string=new_string)

    def delete(self, name: str) -> dict:
        return self._stage("delete", name=name)

    def promote(self, candidate_id: str, evaluation: dict) -> dict:
        manifest_path = self.root / candidate_id / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"unknown skill candidate {candidate_id!r}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "staged":
            raise ValueError("only staged skill candidates can be promoted")
        if _hashes(self.active.root) != manifest["parent_skill_hashes"]:
            raise ValueError("active skill parent changed after candidate creation")
        if _hashes(self.root / candidate_id / "library") != manifest["candidate_skill_hashes"]:
            raise ValueError("candidate skill content does not match its manifest")
        _validate_evaluation(evaluation, manifest)

        rollback = self.root / candidate_id / "rollback-library"
        if rollback.exists():
            raise ValueError("candidate already has a rollback snapshot")
        shutil.move(str(self.active.root), str(rollback))
        try:
            shutil.move(str(self.root / candidate_id / "library"), str(self.active.root))
        except BaseException:
            shutil.move(str(rollback), str(self.active.root))
            raise
        manifest["status"] = "promoted"
        manifest["evaluation"] = evaluation
        manifest["rollback_path"] = str(rollback)
        manifest["promoted_at"] = time.time()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        return {"candidate_id": candidate_id, "status": "promoted",
                "candidate_digest": manifest["candidate_digest"],
                "active_skill_hashes": _hashes(self.active.root)}

    def rollback(self, candidate_id: str) -> dict:
        manifest_path = self.root / candidate_id / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"unknown skill candidate {candidate_id!r}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "promoted":
            raise ValueError("only a promoted candidate can be rolled back")
        rollback = Path(manifest.get("rollback_path", ""))
        if not rollback.is_dir():
            raise ValueError("rollback snapshot is missing")
        rejected = self.root / candidate_id / "rejected-library"
        shutil.move(str(self.active.root), str(rejected))
        try:
            shutil.move(str(rollback), str(self.active.root))
        except BaseException:
            shutil.move(str(rejected), str(self.active.root))
            raise
        manifest["status"] = "rolled_back"
        manifest["rolled_back_at"] = time.time()
        manifest["active_skill_hashes_after_rollback"] = _hashes(self.active.root)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        return {"candidate_id": candidate_id, "status": "rolled_back",
                "active_skill_hashes": _hashes(self.active.root)}


def _hash_json(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_evaluation(evaluation: dict, manifest: dict) -> None:
    if not isinstance(evaluation, dict) or evaluation.get("independent") is not True:
        raise ValueError("candidate promotion requires an independent evaluation")
    if evaluation.get("candidate_id") != manifest["candidate_id"]:
        raise ValueError("evaluation candidate_id mismatch")
    if evaluation.get("parent_skill_hashes") != manifest["parent_skill_hashes"]:
        raise ValueError("evaluation parent skill hashes mismatch")
    if evaluation.get("candidate_skill_hashes") != manifest["candidate_skill_hashes"]:
        raise ValueError("evaluation candidate skill hashes mismatch")
    report_path = Path(str(evaluation.get("report_path") or ""))
    report_digest = evaluation.get("report_sha256")
    if not report_path.is_file() or not isinstance(report_digest, str):
        raise ValueError("evaluation must identify an existing immutable report file and hash")
    if _file_hash(report_path) != report_digest:
        raise ValueError("evaluation report hash does not match report content")
    seeds = evaluation.get("seeds")
    if not isinstance(seeds, list) or len(set(map(str, seeds))) < 2:
        raise ValueError("candidate promotion requires at least two independent seeds")
    rows = evaluation.get("seed_metrics")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("evaluation requires per-seed metrics")
    direction = evaluation.get("direction")
    minimum_delta = float(evaluation.get("minimum_delta", 0.0))
    max_regression = float(evaluation.get("max_group_regression", 0.0))
    if direction not in {"higher", "lower"} or minimum_delta < 0 or max_regression < 0:
        raise ValueError("evaluation gate parameters are invalid")
    by_seed = {str(row.get("seed")): row for row in rows if isinstance(row, dict)}
    if any(str(seed) not in by_seed for seed in seeds):
        raise ValueError("per-seed evaluation rows do not match the declared seeds")
    for seed in seeds:
        row = by_seed[str(seed)]
        baseline = float(row["baseline"])
        candidate = float(row["candidate"])
        group_delta = float(row.get("worst_group_delta", 0.0))
        if not all(math.isfinite(value) for value in (baseline, candidate, group_delta)):
            raise ValueError("evaluation metrics must be finite")
        improvement = candidate - baseline if direction == "higher" else baseline - candidate
        if improvement < minimum_delta:
            raise ValueError(f"seed {seed} did not pass the required improvement margin")
        if group_delta < -max_regression:
            raise ValueError(f"seed {seed} exceeded the worst-group regression guardrail")
