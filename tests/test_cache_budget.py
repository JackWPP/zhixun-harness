"""Bound cross-round cache storage without deleting unrelated state."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from rsi.model import ExtendedModel
from rsi.platform.experiment_cache import ExperimentCache


def scenario(tmp_path):
    x = np.arange(24, dtype=float).reshape(12, 2) / 10
    y = 10 + np.arange(12, dtype=float)
    model = ExtendedModel("regression", [json.dumps([["key", key]]) for key in ("x", "z")],
                          [], x, y, hidden=4, seed=13)
    parent = tmp_path / "parent.npz"
    model.save(parent)
    calls = []

    def evaluate(cache, variant):
        def fit():
            calls.append(variant)
            return model.clone()
        return cache.get_or_fit(parent_path=parent, task_key="B", x_train=x, y_train=y,
                                configuration={"variant": variant}, fit=fit,
                                code_revision="fixed-test-source")
    return evaluate, calls


def owned_entries(cache):
    if not cache.directory.exists():
        return []
    return [row for path in cache.directory.iterdir()
            if (row := cache._owned_entry(path)) is not None]


def test_many_rounds_stay_within_default_file_and_byte_budget(tmp_path):
    evaluate, _ = scenario(tmp_path)
    directory = tmp_path / "cache"
    last = None
    for variant in range(20):
        cache = ExperimentCache(directory)
        _, last = evaluate(cache, variant)
        entries = owned_entries(cache)
        assert len(entries) <= 8
        assert sum(row[1] for row in entries) <= 16 * 1024 * 1024
        assert len([p for p in directory.rglob("*") if p.is_file()]) <= 16
    _, hit = evaluate(ExperimentCache(directory), 19)
    assert hit["cache_hit"] and hit["cache_source"] == "state"
    assert hit["cache_key"] == last["cache_key"]


def test_byte_budget_evicts_oldest_entries(tmp_path):
    evaluate, _ = scenario(tmp_path)
    directory = tmp_path / "cache"
    initial = ExperimentCache(directory)
    _, first = evaluate(initial, 1)
    entry_bytes = owned_entries(initial)[0][1]
    # Each similarly sized model fits alone, but two cannot fit together.
    budget = entry_bytes + 100
    cache = ExperimentCache(directory, max_disk_bytes=budget)
    _, second = evaluate(cache, 2)
    entries = owned_entries(cache)
    assert len(entries) == 1 and entries[0][2].name == second["cache_key"]
    assert sum(row[1] for row in entries) <= budget
    assert not (directory / first["cache_key"]).exists()


def test_disk_hit_refreshes_recency_before_eviction(tmp_path):
    evaluate, _ = scenario(tmp_path)
    directory = tmp_path / "cache"
    cache = ExperimentCache(directory, max_disk_entries=2)
    _, first = evaluate(cache, 1)
    _, second = evaluate(cache, 2)
    os.utime(directory / first["cache_key"], ns=(1, 1))
    os.utime(directory / second["cache_key"], ns=(2, 2))
    fresh = ExperimentCache(directory, max_disk_entries=2)
    _, reused = evaluate(fresh, 1)
    assert reused["cache_source"] == "state"
    _, third = evaluate(fresh, 3)
    assert (directory / first["cache_key"]).is_dir()
    assert (directory / third["cache_key"]).is_dir()
    assert not (directory / second["cache_key"]).exists()


def test_single_oversized_entry_stays_in_memory_only(tmp_path):
    evaluate, calls = scenario(tmp_path)
    cache = ExperimentCache(tmp_path / "cache", max_disk_bytes=1)
    first, _ = evaluate(cache, 1)
    second, hit = evaluate(cache, 1)
    assert len(calls) == 1 and hit["cache_source"] == "memory"
    np.testing.assert_array_equal(first.params["w1"], second.params["w1"])
    assert list(cache.directory.iterdir()) == []


def test_unknown_files_entries_and_symlink_targets_are_preserved(tmp_path):
    evaluate, _ = scenario(tmp_path)
    directory = tmp_path / "cache"
    directory.mkdir()
    note = directory / "notes.txt"
    note.write_text("user state")
    unknown = directory / ("c" * 64)
    unknown.mkdir()
    (unknown / "manifest.json").write_text('{"unknown": true}')
    (unknown / "checkpoint.npz").write_text("user checkpoint")
    target = tmp_path / "external"
    target.mkdir()
    (target / "keep.txt").write_text("outside cache")
    link = directory / ("d" * 64)
    link.symlink_to(target, target_is_directory=True)
    cache = ExperimentCache(directory, max_disk_entries=1)
    _, first = evaluate(cache, 1)
    # A recognized entry with an extra file ceases to be cache-owned for deletion.
    extra = directory / first["cache_key"] / "keep.txt"
    extra.write_text("do not remove")
    evaluate(cache, 2)
    evaluate(cache, 3)
    assert note.read_text() == "user state"
    assert (unknown / "checkpoint.npz").read_text() == "user checkpoint"
    assert link.is_symlink() and (target / "keep.txt").read_text() == "outside cache"
    assert extra.read_text() == "do not remove"
    assert len(owned_entries(cache)) <= 1


def test_new_round_prunes_preexisting_owned_entries(tmp_path):
    evaluate, _ = scenario(tmp_path)
    directory = tmp_path / "cache"
    large = ExperimentCache(directory, max_disk_entries=20)
    for variant in range(12):
        evaluate(large, variant)
    assert len(owned_entries(large)) == 12
    bounded = ExperimentCache(directory)
    assert len(owned_entries(bounded)) == 8
