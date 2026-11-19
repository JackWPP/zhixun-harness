"""Host side of the contest-harness chunked upload protocol (SDK >= 1.0.3).

SDK 1.0.3+ no longer inlines candidate/state bytes in ``candidate_request`` /
``state_request``.  Instead it performs::

    upload_begin_request   {kind, files:[{path,size_bytes,sha256}]}
    upload_chunk_request   {upload_id, path, offset, data_base64}
    upload_finish_request  {upload_id}

and then references the finished upload by ``upload_id`` in the candidate or
state request.  The host must support this (the SDK refuses to fall back to the
old inline protocol), which is what :class:`UploadStore` implements.
"""
from __future__ import annotations

import base64
import hashlib

MAX_FILES = 64


class UploadError(RuntimeError):
    """Raised when the participant sends an invalid chunked upload."""


def _safe_relpath(value) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        return False
    parts = value.split("/")
    return all(part not in ("", ".", "..") for part in parts)


class UploadStore:
    """Collect chunked uploads and materialise them on finish."""

    def __init__(self, max_bytes: int = 512 * 1024 * 1024, max_files: int = MAX_FILES):
        self.max_bytes = int(max_bytes)
        self.max_files = int(max_files)
        self._uploads: dict[str, dict] = {}
        self._next_id = 1

    # -- dispatch ---------------------------------------------------------
    def handle(self, reply: dict):
        """Return a response frame for upload RPCs, or ``None`` otherwise."""
        kind = reply.get("type")
        if kind == "upload_begin_request":
            return self._begin(reply)
        if kind == "upload_chunk_request":
            return self._chunk(reply)
        if kind == "upload_finish_request":
            return self._finish(reply)
        return None

    # -- protocol ---------------------------------------------------------
    def _begin(self, reply: dict) -> dict:
        manifest = reply.get("files")
        if not isinstance(manifest, list) or not manifest or len(manifest) > self.max_files:
            raise UploadError("upload manifest invalid")
        files: dict[str, dict] = {}
        total = 0
        for item in manifest:
            if not isinstance(item, dict):
                raise UploadError("upload manifest invalid")
            path = item.get("path")
            size = item.get("size_bytes")
            digest = item.get("sha256")
            if not _safe_relpath(path) or path in files:
                raise UploadError("upload path invalid: %r" % (path,))
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise UploadError("upload size invalid: %r" % (path,))
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise UploadError("upload digest invalid: %r" % (path,))
            total += size
            files[path] = {"size": size, "sha256": digest, "data": bytearray()}
        if total > self.max_bytes:
            raise UploadError("upload exceeds %d bytes" % self.max_bytes)
        upload_id = "upload-%d" % self._next_id
        self._next_id += 1
        kind = reply.get("kind")
        if kind not in ("candidate", "state"):
            raise UploadError("upload kind invalid: %r" % (kind,))
        self._uploads[upload_id] = {
            "kind": kind,
            "files": files,
            "received": 0,
            "done": False,
            "materialized": None,
        }
        return {"type": "upload_begin_response", "id": reply["id"], "upload_id": upload_id}

    def _chunk(self, reply: dict) -> dict:
        upload_id = reply.get("upload_id")
        upload = self._uploads.get(upload_id)
        if upload is None or upload["done"]:
            raise UploadError("unknown upload id: %r" % (upload_id,))
        path = reply.get("path")
        item = upload["files"].get(path)
        if item is None:
            raise UploadError("upload path not declared: %r" % (path,))
        offset = reply.get("offset")
        if not isinstance(offset, int) or offset != len(item["data"]):
            raise UploadError("upload offset mismatch: %r" % (offset,))
        try:
            chunk = base64.b64decode(reply.get("data_base64", ""), validate=True)
        except (ValueError, TypeError) as exc:
            raise UploadError("upload chunk is not valid base64") from exc
        if len(item["data"]) + len(chunk) > item["size"]:
            raise UploadError("upload chunk exceeds declared size: %r" % (path,))
        item["data"].extend(chunk)
        upload["received"] += len(chunk)
        if upload["received"] > self.max_bytes:
            raise UploadError("upload exceeds %d bytes" % self.max_bytes)
        return {"type": "upload_chunk_response", "id": reply["id"], "offset": len(item["data"])}

    def _finish(self, reply: dict) -> dict:
        upload_id = reply.get("upload_id")
        upload = self._uploads.get(upload_id)
        if upload is None or upload["done"]:
            raise UploadError("unknown upload id: %r" % (upload_id,))
        materialized: dict[str, bytes] = {}
        for path, item in upload["files"].items():
            raw = bytes(item["data"])
            if len(raw) != item["size"]:
                raise UploadError("upload is incomplete: %r" % (path,))
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            if digest != item["sha256"]:
                raise UploadError("upload digest mismatch: %r" % (path,))
            materialized[path] = raw
        upload["done"] = True
        upload["materialized"] = materialized
        return {"type": "upload_finish_response", "id": reply["id"], "upload_id": upload_id}

    # -- consumer ---------------------------------------------------------
    def take(self, upload_id, expected_kind: str) -> dict[str, bytes]:
        """Return and forget a finished upload (validated by content)."""
        upload = self._uploads.get(upload_id)
        if upload is None or not upload["done"] or upload["materialized"] is None:
            raise UploadError("upload not finished: %r" % (upload_id,))
        if upload["kind"] != expected_kind:
            raise UploadError("upload kind mismatch: %r != %r" % (upload["kind"], expected_kind))
        files = upload["materialized"]
        del self._uploads[upload_id]
        return dict(files)
