"""Protocol-neutral, restartable batch lifecycle for streaming RSI.

The platform's ``contest.harness.workspace.v2`` adapter is intentionally kept
separate: its method signatures are not present in the published SDK material
available to this repository. This module implements the ordering and replay
invariants independently, so a future adapter can map the official callbacks
without guessing their signatures.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


class BatchProtocolError(ValueError):
    """A batch transition violates ordering, identity, or idempotency rules."""


_BATCH_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_DECISIONS = {"promote", "retain", "rollback"}


def _canonical(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BatchProtocolError(f"batch data must be finite JSON values: {exc}") from exc


def _sha(value) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_write(path: Path, payload: dict, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".batch-state.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class BatchSession:
    """Persist a strict predict -> decide -> reveal -> complete batch stream.

    Repeating an operation with the same payload is idempotent. Reusing a batch
    identity with different input or output is rejected. Raw labels live in a
    separate mode-0600 file and are returned only by ``reveal_labels`` after a
    decision has been committed.
    """

    SCHEMA_VERSION = 1

    def __init__(self, state_dir: str | Path):
        self.root = Path(state_dir)
        self.labels_dir = self.root / "label-state"
        self.state_path = self.root / "batch-state.json"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.labels_dir, 0o700)
        if self.state_path.is_file():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self._state.get("schema_version") != self.SCHEMA_VERSION:
                raise BatchProtocolError("unsupported batch state schema")
        else:
            self._state = {"schema_version": self.SCHEMA_VERSION,
                           "batches": {}, "order": [], "events": []}
            self._save()

    def _save(self) -> None:
        _atomic_write(self.state_path, self._state)

    @staticmethod
    def _copy(value):
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

    def _get(self, batch_id: str) -> dict:
        try:
            return self._state["batches"][batch_id]
        except KeyError as exc:
            raise BatchProtocolError(f"unknown batch {batch_id!r}") from exc

    def _event(self, batch: dict, transition: str, digest: str | None = None) -> None:
        self._state["events"].append({"batch_id": batch["batch_id"],
                                       "transition": transition,
                                       "digest": digest})
        self._save()

    def start_batch(self, batch_id: str, sequence: int, inputs,
                    champion_sha256: str, fallback_sha256: str | None = None) -> dict:
        if not isinstance(batch_id, str) or not _BATCH_ID.fullmatch(batch_id):
            raise BatchProtocolError("invalid batch_id")
        if not isinstance(inputs, list):
            raise BatchProtocolError("batch inputs must be a JSON array")
        input_rows = len(inputs)
        digest = _sha(inputs)
        existing = self._state["batches"].get(batch_id)
        if existing is not None:
            if (existing["input_sha256"] != digest
                    or existing["sequence"] != int(sequence)
                    or existing["champion_sha256"] != champion_sha256
                    or existing["fallback_sha256"] != (fallback_sha256 or champion_sha256)):
                raise BatchProtocolError("duplicate batch identity has conflicting input, sequence, or model")
            return self._copy(existing)
        incomplete = next((self._state["batches"][item]
                           for item in reversed(self._state["order"])
                           if self._state["batches"][item]["phase"] != "complete"), None)
        if incomplete is not None:
            raise BatchProtocolError(f"batch {incomplete['batch_id']} is not complete")
        expected = len(self._state["order"])
        if int(sequence) != expected:
            raise BatchProtocolError(f"expected sequence {expected}, got {sequence}")
        if not isinstance(champion_sha256, str) or not champion_sha256:
            raise BatchProtocolError("champion_sha256 is required")
        batch = {
            "batch_id": batch_id,
            "sequence": int(sequence),
            "input_sha256": digest,
            "input_rows": input_rows,
            "phase": "awaiting_prediction",
            "champion_sha256": champion_sha256,
            "fallback_sha256": fallback_sha256 or champion_sha256,
            "prediction_sha256": None,
            "prediction_rows": None,
            "decision": None,
            "selected_sha256": None,
            "labels_sha256": None,
            "labels_available": False,
            "final_champion_sha256": None,
        }
        self._state["batches"][batch_id] = batch
        self._state["order"].append(batch_id)
        self._event(batch, "batch_started", digest)
        return self._copy(batch)

    def record_predictions(self, batch_id: str, predictions,
                           model_sha256: str | None = None) -> dict:
        batch = self._get(batch_id)
        digest = _sha(predictions)
        if not isinstance(predictions, list):
            raise BatchProtocolError("predictions must be a JSON array")
        if batch["phase"] != "awaiting_prediction":
            if batch["prediction_sha256"] == digest and batch.get("prediction_model_sha256") == model_sha256:
                return self._copy(batch)
            raise BatchProtocolError("predictions are already committed or batch is in the wrong phase")
        if batch["input_rows"] is not None and len(predictions) != batch["input_rows"]:
            raise BatchProtocolError("prediction count must match batch input count")
        batch["prediction_sha256"] = digest
        batch["prediction_model_sha256"] = model_sha256 or batch["champion_sha256"]
        batch["prediction_rows"] = len(predictions)
        batch["phase"] = "awaiting_decision"
        self._event(batch, "predictions_recorded", digest)
        return self._copy(batch)

    def commit_decision(self, batch_id: str, decision: str,
                        candidate_sha256: str | None = None) -> dict:
        batch = self._get(batch_id)
        if batch["phase"] != "awaiting_decision":
            if batch["decision"] == decision and batch.get("candidate_sha256") == candidate_sha256:
                return self._copy(batch)
            raise BatchProtocolError("decision requires committed predictions and may only be committed once")
        if decision not in _DECISIONS:
            raise BatchProtocolError(f"unsupported decision {decision!r}")
        if decision == "promote" and not candidate_sha256:
            raise BatchProtocolError("promote requires candidate_sha256")
        selected = (candidate_sha256 if decision == "promote" else
                    batch["fallback_sha256"] if decision == "rollback" else
                    batch["champion_sha256"])
        batch["decision"] = decision
        batch["candidate_sha256"] = candidate_sha256
        batch["selected_sha256"] = selected
        batch["phase"] = "awaiting_labels"
        self._event(batch, "decision_committed", _sha({"decision": decision,
                                                         "selected": selected}))
        return self._copy(batch)

    def record_labels(self, batch_id: str, labels) -> dict:
        batch = self._get(batch_id)
        if not isinstance(labels, list):
            raise BatchProtocolError("labels must be a JSON array")
        if batch["phase"] != "awaiting_labels":
            if batch["labels_sha256"] == _sha(labels) and batch["labels_available"]:
                return self._copy(batch)
            raise BatchProtocolError("labels may only be revealed after decision commit")
        if batch["prediction_rows"] is not None and len(labels) != batch["prediction_rows"]:
            raise BatchProtocolError("label count must match committed predictions")
        digest = _sha(labels)
        label_path = self._label_path(batch_id)
        _atomic_write(label_path, {"batch_id": batch_id, "labels_sha256": digest,
                                   "labels": labels})
        batch["labels_sha256"] = digest
        batch["labels_available"] = True
        batch["phase"] = "awaiting_update"
        self._event(batch, "labels_revealed", digest)
        return self._copy(batch)

    def reveal_labels(self, batch_id: str):
        batch = self._get(batch_id)
        if not batch["labels_available"] or batch["phase"] not in {"awaiting_update", "complete"}:
            raise BatchProtocolError("labels are not available before the committed decision")
        payload = json.loads(self._label_path(batch_id).read_text(encoding="utf-8"))
        if payload.get("labels_sha256") != batch["labels_sha256"]:
            raise BatchProtocolError("persisted label digest mismatch")
        return payload["labels"]

    def complete_batch(self, batch_id: str, final_champion_sha256: str | None = None) -> dict:
        batch = self._get(batch_id)
        final_sha = final_champion_sha256 or batch.get("selected_sha256")
        if batch["phase"] == "complete":
            if batch["final_champion_sha256"] == final_sha:
                return self._copy(batch)
            raise BatchProtocolError("completed batch cannot change its final champion")
        if batch["phase"] != "awaiting_update" or not batch["labels_available"]:
            raise BatchProtocolError("batch completion requires revealed labels")
        if not final_sha:
            raise BatchProtocolError("final champion hash is required")
        batch["final_champion_sha256"] = final_sha
        batch["phase"] = "complete"
        self._event(batch, "batch_completed", final_sha)
        return self._copy(batch)

    def agent_view(self, batch_id: str) -> dict:
        """Return batch metadata safe for pre-label agent context."""
        batch = self._get(batch_id)
        safe = {key: value for key, value in batch.items()
                if key not in {"labels_sha256", "labels_available"}}
        safe["labels_revealed"] = False
        return self._copy(safe)

    def _label_path(self, batch_id: str) -> Path:
        digest = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()
        return self.labels_dir / f"{digest}.json"
