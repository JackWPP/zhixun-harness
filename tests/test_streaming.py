from __future__ import annotations

import json

import pytest

from rsi.streaming import BatchProtocolError, BatchSession


def test_batch_lifecycle_orders_prediction_decision_and_label_reveal(tmp_path):
    session = BatchSession(tmp_path)
    with pytest.raises(BatchProtocolError, match="unknown batch"):
        session.reveal_labels("batch-0")

    session.start_batch("batch-0", 0, [{"x": 1}, {"x": 2}], "parent-sha")
    with pytest.raises(BatchProtocolError, match="before the committed decision"):
        session.reveal_labels("batch-0")

    session.record_predictions("batch-0", [1, 2])
    with pytest.raises(BatchProtocolError, match="decision"):
        session.record_labels("batch-0", [1, 2])
    session.commit_decision("batch-0", "retain")

    with pytest.raises(BatchProtocolError, match="not complete"):
        session.start_batch("batch-1", 1, [{"x": 3}], "parent-sha")
    labels = session.record_labels("batch-0", [1, 2])
    assert labels["labels_available"] is True
    assert session.reveal_labels("batch-0") == [1, 2]
    session.complete_batch("batch-0", "parent-sha")

    with pytest.raises(BatchProtocolError, match="expected sequence 1"):
        session.start_batch("batch-1", 2, [{"x": 3}], "parent-sha")
    session.start_batch("batch-1", 1, [{"x": 3}], "parent-sha")
    with pytest.raises(BatchProtocolError, match="conflicting"):
        session.start_batch("batch-1", 1, [{"x": 999}], "parent-sha")


def test_batch_replay_and_restart_are_idempotent(tmp_path):
    session = BatchSession(tmp_path)
    batch = session.start_batch("stream:0", 0, [{"a": 1}], "parent")
    duplicate = session.start_batch("stream:0", 0, [{"a": 1}], "parent")
    assert duplicate["input_sha256"] == batch["input_sha256"]

    session.record_predictions("stream:0", [{"value": 3.0}], "parent")
    session = BatchSession(tmp_path)
    replayed = session.record_predictions("stream:0", [{"value": 3.0}], "parent")
    assert replayed["phase"] == "awaiting_decision"
    session.commit_decision("stream:0", "promote", "candidate")
    session = BatchSession(tmp_path)
    with pytest.raises(BatchProtocolError, match="not available"):
        session.reveal_labels("stream:0")
    session.record_labels("stream:0", [5.0])
    session = BatchSession(tmp_path)
    assert session.reveal_labels("stream:0") == [5.0]
    session.complete_batch("stream:0")
    assert session.complete_batch("stream:0")["phase"] == "complete"


def test_batch_replay_rejects_changed_model_or_non_array_payloads(tmp_path):
    session = BatchSession(tmp_path)
    session.start_batch("b0", 0, [{"x": 1}], "champion")
    with pytest.raises(BatchProtocolError, match="conflicting"):
        session.start_batch("b0", 0, [{"x": 1}], "other-champion")
    with pytest.raises(BatchProtocolError, match="JSON array"):
        session.record_predictions("b0", {"value": 1})


def test_rollback_selects_known_fallback_and_agent_view_hides_label_metadata(tmp_path):
    session = BatchSession(tmp_path)
    session.start_batch("b0", 0, [{"x": 1}], "champion", fallback_sha256="fallback")
    session.record_predictions("b0", [0])
    committed = session.commit_decision("b0", "rollback", "bad-candidate")
    assert committed["selected_sha256"] == "fallback"
    session.record_labels("b0", [0])
    safe = session.agent_view("b0")
    assert "labels_sha256" not in safe
    assert "labels_available" not in safe
    assert safe["labels_revealed"] is False
    assert json.loads((tmp_path / "batch-state.json").read_text())["batches"]["b0"]["labels_sha256"]
