"""Reuse identical public-training experiments, never validation decisions.

The caller explicitly chooses a writable state directory for cross-round reuse.
Without a directory this is a per-context, in-memory cache. Keys include the
actual parent bytes, ordered training arrays, training configuration, source and
runtime fingerprints. Cached models are copied on every return; validation is
always the caller's responsibility and is intentionally absent from this cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np

from rsi.model import ExtendedModel, MAX_CHECKPOINT_BYTES


SCHEMA = 1


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_sha256(x_train, y_train) -> str:
    """Hash values, dtype, shape and row order, not merely sample identifiers."""
    digest = hashlib.sha256()
    for name, values in (("x", x_train), ("y", y_train)):
        array = np.ascontiguousarray(values)
        if array.dtype.hasobject:
            raise ValueError("object arrays are not valid experiment-cache inputs")
        header = _json_bytes({"name": name, "dtype": array.dtype.str,
                              "shape": list(array.shape), "bytes": array.nbytes})
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def source_sha256() -> str:
    """Fingerprint shipped Python training code, including this cache itself."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime() -> dict:
    return {
        "python": list(sys.version_info[:3]), "numpy": np.__version__,
        "system": platform.system(), "machine": platform.machine(),
        "threads": {name: os.environ.get(name) for name in (
            "BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }


class ExperimentCache:
    """Bounded model cache; missing/corrupt/unreadable disk entries are misses.

    Share one instance across contexts within a round. For a new process/round,
    instantiate with the corresponding copied ``rnd.state_workspace`` directory.
    ``RSI_V2_EXPERIMENT_CACHE=0`` disables reuse for controlled comparisons.
    Disk storage is capped at eight entries (16 files) and 16 MiB by default,
    leaving room under the SDK's 64-file state upload limit. Unknown state and
    symlinks are never removed. The byte budget covers cache-owned files only.
    """

    def __init__(self, directory: str | Path | None = None, *, enabled: bool | None = None,
                 max_memory_entries: int = 8, max_disk_entries: int = 8,
                 max_disk_bytes: int = 16 * 1024 * 1024):
        self.directory = Path(directory) if directory is not None else None
        self.enabled = (os.environ.get("RSI_V2_EXPERIMENT_CACHE", "1").lower()
                        not in {"0", "off", "false", "no"}) if enabled is None else enabled
        self.max_memory_entries = max(1, int(max_memory_entries))
        self.max_disk_entries = max(0, int(max_disk_entries))
        self.max_disk_bytes = max(0, int(max_disk_bytes))
        self._models: dict[str, object] = {}
        if self.enabled:
            self._prune_disk()

    def get_or_fit(self, *, parent_path: Path, task_key: str, x_train, y_train,
                   configuration: dict, fit, code_revision: str | None = None):
        if code_revision is None:
            code_revision = source_sha256()
        identity = {
            "schema": SCHEMA, "task_key": str(task_key),
            "parent_sha256": file_sha256(parent_path),
            "train_sha256": training_sha256(x_train, y_train),
            "configuration": configuration, "code_sha256": code_revision,
            "runtime": _runtime(),
        }
        key = hashlib.sha256(_json_bytes(identity)).hexdigest()
        metadata = {"cache_hit": False, "cache_key": key,
                    "parent_sha256": identity["parent_sha256"],
                    "train_sha256": identity["train_sha256"],
                    "code_sha256": identity["code_sha256"],
                    "cache_enabled": bool(self.enabled), "cache_source": "fit",
                    "disk_budget": {"entries": self.max_disk_entries, "bytes": self.max_disk_bytes}}
        if self.enabled:
            if key in self._models:
                self._touch(key)
                metadata.update(cache_hit=True, cache_source="memory")
                return copy.deepcopy(self._models[key]), metadata
            model = self._load(key, identity)
            if model is not None:
                self._touch(key)
                self._remember(key, model)
                metadata.update(cache_hit=True, cache_source="state")
                return copy.deepcopy(model), metadata
        model = fit()
        model.validate()
        if self.enabled:
            self._remember(key, model)
            self._save(key, identity, model)
        return model, metadata

    def _remember(self, key, model):
        if len(self._models) >= self.max_memory_entries:
            self._models.pop(next(iter(self._models)))
        self._models[key] = copy.deepcopy(model)

    def _owned_entry(self, entry: Path):
        """Recognize only our two-file, hash-named entries without following links."""
        try:
            if (self.directory is None or self.directory.is_symlink()
                    or re.fullmatch(r"[0-9a-f]{64}", entry.name) is None
                    or entry.is_symlink() or not entry.is_dir()):
                return None
            children = list(entry.iterdir())
            if {path.name for path in children} != {"manifest.json", "checkpoint.npz"}:
                return None
            if any(path.is_symlink() or not path.is_file() for path in children):
                return None
            manifest = entry / "manifest.json"
            if manifest.stat().st_size > 65536:
                return None
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            identity = payload.get("identity") if isinstance(payload, dict) else None
            if (not isinstance(identity, dict) or identity.get("schema") != SCHEMA
                    or payload.get("cache_key") != entry.name
                    or hashlib.sha256(_json_bytes(identity)).hexdigest() != entry.name
                    or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("checkpoint_sha256", "")))):
                return None
            return (entry.stat().st_mtime_ns, sum(path.stat().st_size for path in children), entry)
        except (OSError, ValueError, TypeError):
            return None

    def _touch(self, key):
        if self.directory is None:
            return
        entry = self.directory / key
        if self._owned_entry(entry) is not None:
            try:
                os.utime(entry, follow_symlinks=False)
            except OSError:
                pass

    def _remove_owned_entry(self, entry):
        if self._owned_entry(entry) is None:
            return False
        descriptor = None
        try:
            # Deletion is relative to a directory opened without following a
            # symlink; only the two known files can be removed, never extra data.
            descriptor = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            if set(os.listdir(descriptor)) != {"manifest.json", "checkpoint.npz"}:
                return False
            for name in ("manifest.json", "checkpoint.npz"):
                if not stat.S_ISREG(os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode):
                    return False
            for name in ("manifest.json", "checkpoint.npz"):
                os.unlink(name, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = None
            entry.rmdir()
            return True
        except (OSError, AttributeError, NotImplementedError):
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _prune_disk(self):
        if self.directory is None or self.directory.is_symlink() or not self.directory.is_dir():
            return
        try:
            entries = [record for path in self.directory.iterdir()
                       if (record := self._owned_entry(path)) is not None]
            entries.sort(key=lambda record: (record[0], record[2].name))
            count, total = len(entries), sum(record[1] for record in entries)
            for _, size, path in entries:
                if count <= self.max_disk_entries and total <= self.max_disk_bytes:
                    break
                if self._remove_owned_entry(path):
                    count -= 1
                    total -= size
        except OSError:
            pass  # An inaccessible cache must not affect the trained model.

    def _load(self, key: str, identity: dict):
        if self.directory is None:
            return None
        entry = self.directory / key
        manifest = entry / "manifest.json"
        checkpoint = entry / "checkpoint.npz"
        try:
            if (self.directory.is_symlink() or entry.is_symlink()
                    or manifest.is_symlink() or checkpoint.is_symlink()
                    or not manifest.is_file() or manifest.stat().st_size > 65536
                    or not checkpoint.is_file()
                    or checkpoint.stat().st_size > MAX_CHECKPOINT_BYTES):
                return None
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if (not isinstance(payload, dict) or payload.get("identity") != identity
                    or payload.get("cache_key") != key
                    or payload.get("checkpoint_sha256") != file_sha256(checkpoint)):
                return None
            return ExtendedModel.from_checkpoint(checkpoint)
        except Exception:
            # A copied state may be malformed at any layer (JSON metadata,
            # archive, dtype or model schema). It must never become authoritative.
            return None

    def _save(self, key: str, identity: dict, model):
        if self.directory is None or self.max_disk_entries == 0 or self.max_disk_bytes == 0:
            return
        temporary = None
        try:
            if self.directory.is_symlink():
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            entry = self.directory / key
            if entry.exists() or entry.is_symlink():
                return  # Another writer, or malformed state: never overwrite it.
            temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.directory))
            checkpoint = temporary / "checkpoint.npz"
            model.save(checkpoint)
            payload = {"identity": identity, "cache_key": key,
                       "checkpoint_sha256": file_sha256(checkpoint)}
            (temporary / "manifest.json").write_bytes(_json_bytes(payload))
            total = checkpoint.stat().st_size + (temporary / "manifest.json").stat().st_size
            if total > self.max_disk_bytes:
                return  # Keep the result in memory; never publish an oversized entry.
            os.rename(temporary, entry)
            temporary = None
            self._prune_disk()
        except (OSError, ValueError, TypeError):
            pass  # Cache persistence is optional, training output remains valid.
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
