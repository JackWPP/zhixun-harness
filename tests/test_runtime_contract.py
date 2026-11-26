"""Check the actual entrypoint and both image delivery paths."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

import pytest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "docker/app"
SDK = REPO / "vendor/contest_harness-1.0.5"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BASELINE_CPU_THREADS")
spec = importlib.util.spec_from_file_location("image_builder", REPO / "scripts/build_image_tar.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def clean_environment(*paths):
    environment = dict(os.environ)
    for key in THREADS:
        environment.pop(key, None)
    environment["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_direct_python_entry_sets_threads_before_sdk_import(tmp_path):
    fake_sdk = tmp_path / "contest_harness"
    fake_sdk.mkdir()
    (fake_sdk / "__init__.py").write_text("")
    (fake_sdk / "cli.py").write_text(
        "import json, os, sys\n"
        "assert 'numpy' not in sys.modules\n"
        f"before = {{k: os.environ.get(k) for k in {THREADS!r}}}\n"
        "def main(argv):\n"
        "    print(json.dumps(before))\n"
        "    return 0\n")
    environment = clean_environment(tmp_path)
    environment["MKL_NUM_THREADS"] = "2"
    result = subprocess.run([sys.executable, str(APP / "harness.py"), "describe"],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {key: "2" if key == "MKL_NUM_THREADS" else "1"
                                         for key in THREADS}


def test_direct_entry_describes_verified_sdk():
    result = subprocess.run([sys.executable, str(APP / "harness.py"), "describe"],
                            env=clean_environment(REPO, SDK, APP), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["protocol"] == "contest.harness.workspace.v2"
    assert response["sdk_version"] == "1.0.5"
    assert response["tasks"] == ["classification", "regression"]


def test_image_build_defaults_match_dockerfile():
    text = (REPO / "docker/Dockerfile").read_text()
    block = re.search(r"^ENV (.*?)(?=\n\n)", text, re.M | re.S)[1]
    values = dict(item.split("=", 1) for item in shlex.split(block.replace("\\\n", " ")))
    assert values == builder.DEFAULT_ENV
    assert values["RSI_V2_PROBE_RETAIN"] == ""
    assert values["RSI_V2_SCRATCH_ARM"] == "off"
    assert "contest_harness-1.0.5" in text
    assert "contest_harness-1.0.5" in (REPO / "scripts/stage_image.sh").read_text()


def stage_minimal_image(root):
    app = root / "opt/zhixun/app"
    (app / "rsi").mkdir(parents=True)
    (app / "rsi/__init__.py").write_text('__version__ = "0.15.0.dev1"\n')
    (app / "contest_harness").mkdir()
    shutil.copyfile(SDK / "contest_harness/protocol.py", app / "contest_harness/protocol.py")
    (app / "pyproject.toml").write_text('[project]\nversion = "0.15.0.dev1"\n')
    return app


def test_manifest_tracks_staged_source_and_excludes_credentials(tmp_path):
    app = stage_minimal_image(tmp_path)
    environment = {**builder.DEFAULT_ENV, "OPENAI_API_KEY": "DO_NOT_EXPORT",
                   "SWANLAB_API_KEY": "DO_NOT_EXPORT", "RSI_V2_B_EPOCHS": "30"}
    manifest = builder.write_build_manifest(tmp_path, environment, "a" * 40, "true")
    assert manifest["git_commit"] == "a" * 40
    assert manifest["git_dirty"] is True
    assert manifest["sdk"]["version"] == "1.0.5"
    assert manifest["environment"]["RSI_V2_B_EPOCHS"] == "30"
    assert "DO_NOT_EXPORT" not in json.dumps(manifest)
    assert builder.source_digest(tmp_path)[0] == manifest["source_sha256"]
    assert builder.write_build_manifest(tmp_path, environment, "a" * 40, "true") == manifest
    (app / "rsi/__init__.py").write_text("# changed source\n")
    assert builder.source_digest(tmp_path)[0] != manifest["source_sha256"]


def test_manifest_rejects_wrong_sdk_and_invalid_commit(tmp_path):
    app = stage_minimal_image(tmp_path)
    with pytest.raises(ValueError, match="commit hash"):
        builder.write_build_manifest(tmp_path, git_sha="main")
    (app / "contest_harness/protocol.py").write_text('SDK_VERSION = "1.0.4"\n')
    with pytest.raises(ValueError, match="SDK version"):
        builder.write_build_manifest(tmp_path, git_sha="", git_dirty="unknown")


def test_archive_identity_matches_its_real_environment_and_source(tmp_path):
    base = tmp_path / "base"
    (base / "usr/local/bin").mkdir(parents=True)
    (base / "usr/local/bin/python3").write_text("test fixture, not executable\n")
    overlay = tmp_path / "overlay"
    stage_minimal_image(overlay)
    output = tmp_path / "harness.tar"
    assert builder.main(["--rootfs", str(base), "--overlay", str(overlay), "--output", str(output),
                         "--git-sha", "b" * 40, "--git-dirty", "false",
                         "--extra-env", "RSI_V2_B_EPOCHS=30"]) == 0
    with tarfile.open(output) as archive:
        entry = json.load(archive.extractfile("manifest.json"))[0]
        config = json.load(archive.extractfile(entry["Config"]))
        environment = dict(value.split("=", 1) for value in config["config"]["Env"])
        assert environment["RSI_V2_B_EPOCHS"] == "30"
        assert len(environment) == len(config["config"]["Env"])
        with tarfile.open(fileobj=io.BytesIO(archive.extractfile(entry["Layers"][1]).read())) as layer:
            identity = json.load(layer.extractfile(builder.MANIFEST_PATH.as_posix()))
    assert identity["git_commit"] == "b" * 40
    assert identity["git_dirty"] is False
    assert identity["environment"]["RSI_V2_B_EPOCHS"] == "30"
    assert identity["source_sha256"] == builder.source_digest(overlay)[0]
