#!/usr/bin/env python3
"""Build a `docker load` compatible image tar without a Docker daemon.

The competition accepts exactly one artifact: a single uncompressed
``linux/amd64`` image tar produced by ``docker save``.  This script assembles
the same classic docker-save layout from a staged root filesystem, which is what
we need on the delivery machine where rootful Docker is unavailable.

    # 1. export a base rootfs (once):  crane export python:3.12-slim | tar -x -C rootfs
    # 2. stage dependencies:            uv pip install --target overlay/opt/zhixun/site-packages -r docker/requirements.txt
    # 3. build:                         python scripts/build_image_tar.py \
    #        --rootfs .scratch/image/rootfs --overlay .scratch/image/overlay \
    #        --output .scratch/image/harness.tar --tag zhixun-rsi-harness:0.3.0

The produced tar is validated by ``docker load`` in CI/delivery and by
``tar -tf`` plus config/diff_id checks here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

FIXED_MTIME = 1735689600  # 2025-01-01T00:00:00Z, keeps rebuilds byte-stable
SOURCE_DIRS = ("rsi", "baselines", "config", "skills", "scripts", "docs", "extra", "datasets")
SOURCE_FILES = ("pyproject.toml", "README.md", "Makefile")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".DS_Store"}
SDK_VERSION = "1.0.5"
SDK_ARCHIVE_SHA256 = "bb5c616081de975d8f932a20cc00fd489e701e7585b69fdb52982bd669a305e6"
MANIFEST_PATH = Path("opt/zhixun/app/build-manifest.json")
DEFAULT_ENV = {
    "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/opt/zhixun/site-packages:/opt/zhixun/app",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "BASELINE_CPU_THREADS": "1",
    "RSI_BUILD_MANIFEST": "/opt/zhixun/app/build-manifest.json",
    "RSI_DATA_ROOT": "/opt/zhixun/app/datasets",
    "RSI_GAS_ROOT": "/opt/zhixun/app/extra/uci-gas/Dataset",
    "RSI_STATE_DIR": "/work/rsi-runs", "RSI_TASKS": "A,B,C", "RSI_ROUNDS": "10",
    "RSI_LLM": "auto", "RSI_V2_CONTROLLER": "auto", "RSI_V2_MAX_TURNS": "10",
    "RSI_V2_CAPACITY_ARM": "ls10", "RSI_V2_NATIVE_TOOLS": "1",
    "RSI_V2_KERNEL_THINKING": "off", "RSI_V2_MAX_OUTPUT": "256",
    "RSI_V2_KERNEL_TOOLS": "core", "RSI_V2_WARMUP_ROUNDS": "0",
    "RSI_V2_B_RETAIN": "0", "RSI_V2_B_STATIC": "0", "RSI_V2_B_EPOCHS": "15",
    "RSI_V2_B_LR": "0.001", "RSI_V2_EXPERIMENT_CACHE": "1",
    "RSI_V2_SCRATCH_ARM": "off", "RSI_V2_C_EFFICIENT": "1",
    "RSI_V2_C_MAX_ALPHA": "0.25", "RSI_V2_PROBE_RETAIN": "",
}
# Hash executable source/configuration; data has separate per-experiment hashes.
# Neither arbitrary host files nor host environment variables enter the record.
IDENTITY_PATHS = (
    "opt/zhixun/app/rsi", "opt/zhixun/app/scripts", "opt/zhixun/app/skills",
    "opt/zhixun/app/contest_harness", "opt/zhixun/app/pyproject.toml",
    "opt/zhixun/requirements.txt", "opt/zhixun/bin/entrypoint.sh",
    "opt/zhixun/bin/rsi-predict", "app/harness.py", "app/contest_submission.py",
)


def source_digest(image_root: Path) -> tuple[str, int]:
    root = Path(image_root).resolve()
    files = set()
    for relative in IDENTITY_PATHS:
        path = root / relative
        candidates = path.rglob("*") if path.is_dir() else [path]
        for item in candidates:
            if item.is_file() and not any(part in SKIP_PARTS for part in item.parts) \
                    and item.suffix != ".pyc":
                if item.is_symlink():
                    raise ValueError(f"source identity requires regular files: {item}")
                files.add(item)
    if not files:
        raise ValueError("no staged source files for build identity")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest(), len(files)


def git_identity(repo: Path) -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo,
            stderr=subprocess.DEVNULL, text=True)
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def write_build_manifest(image_root: Path, environment: dict[str, str] | None = None,
                         git_sha: str | None = None, git_dirty: str | bool | None = None) -> dict:
    root = Path(image_root).resolve()
    if git_sha is None:
        git_sha, detected_dirty = git_identity(Path(__file__).resolve().parents[1])
        if git_dirty is None:
            git_dirty = detected_dirty
    if git_sha and re.fullmatch(r"[0-9a-fA-F]{40}", git_sha) is None:
        raise ValueError("git SHA must be a full 40-character commit hash")
    if isinstance(git_dirty, str):
        if git_dirty not in {"true", "false", "unknown"}:
            raise ValueError("git dirty status must be true, false or unknown")
        git_dirty = {"true": True, "false": False, "unknown": None}[git_dirty]
    protocol = root / "opt/zhixun/app/contest_harness/protocol.py"
    sdk_match = re.search(r'^SDK_VERSION\s*=\s*["\']([^"\']+)', protocol.read_text(), re.M)
    if sdk_match is None or sdk_match[1] != SDK_VERSION:
        raise ValueError("staged SDK version does not match verified SDK 1.0.5")
    project = (root / "opt/zhixun/app/pyproject.toml").read_text()
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.M)
    digest, count = source_digest(root)
    supplied = DEFAULT_ENV if environment is None else environment
    manifest = {
        "schema": "zhixun.build.v1",
        "version": version_match[1] if version_match else None,
        "git_commit": git_sha or None,
        "git_dirty": git_dirty,
        "source_sha256": digest,
        "source_file_count": count,
        "sdk": {"version": SDK_VERSION, "archive_sha256": SDK_ARCHIVE_SHA256},
        "environment": {key: str(supplied.get(key, default))
                        for key, default in DEFAULT_ENV.items()},
    }
    destination = root / MANIFEST_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def image_environment(extra_env: list[str]) -> dict[str, str]:
    values = {"PATH": "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
              **DEFAULT_ENV}
    for item in extra_env:
        key, equals, value = item.partition("=")
        if not equals or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("extra image environment must use KEY=VALUE")
        values[key] = value
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = FIXED_MTIME
    return info


def make_layer_tar(root: Path, output: Path, skip_names: set[str] | None = None) -> str:
    """Tar the contents of *root* (paths relative to root) and return its sha256."""
    skip_names = skip_names or set()
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("wb") as raw, tarfile.open(fileobj=raw, mode="w") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in SKIP_PARTS or part in skip_names for part in relative.parts):
                continue
            info = tarfile.TarInfo(str(relative))
            _normalise(info)
            if path.is_symlink():
                info.type = tarfile.SYMTYPE
                info.linkname = str(path.readlink())
                archive.addfile(info)
                continue
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
                continue
            info.size = path.stat().st_size
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    # hash the whole layer file so diff_id matches its content
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rootfs", help="exported base image rootfs")
    parser.add_argument("--overlay", help="staging dir -> image rootfs")
    parser.add_argument("--output", help="target harness.tar")
    parser.add_argument("--tag", default="zhixun-rsi-harness:0.15.0.dev1")
    parser.add_argument("--write-build-manifest", metavar="IMAGE_ROOT",
                        help="only write source/SDK/configuration identity in a staged image")
    parser.add_argument("--git-sha", default=None, help="source commit (empty means unknown)")
    parser.add_argument("--git-dirty", choices=["true", "false", "unknown"], default=None)
    parser.add_argument("--workdir", default="/opt/zhixun/app")
    parser.add_argument("--extra-env", action="append", default=[],
                        help="extra ENV KEY=VALUE pairs written into the image config")
    args = parser.parse_args(argv)
    environment = image_environment(args.extra_env)
    if args.write_build_manifest:
        # Docker supplies its effective ENV; staging uses canonical defaults.
        if Path(args.write_build_manifest).resolve() == Path("/"):
            environment.update({key: os.environ[key] for key in DEFAULT_ENV if key in os.environ})
        manifest = write_build_manifest(Path(args.write_build_manifest), environment,
                                        args.git_sha, args.git_dirty)
        print(json.dumps({key: manifest[key] for key in
                          ("git_commit", "git_dirty", "source_sha256", "version")}))
        return 0
    if not all((args.rootfs, args.overlay, args.output)):
        parser.error("--rootfs, --overlay and --output are required for an image build")

    rootfs = Path(args.rootfs).resolve()
    overlay = Path(args.overlay).resolve()
    output = Path(args.output).resolve()
    if not (rootfs / "usr" / "local" / "bin" / "python3").exists():
        return sys.exit(f"not a python base rootfs: {rootfs}")
    if not (overlay / "opt" / "zhixun" / "app" / "rsi").is_dir():
        return sys.exit(f"overlay missing /opt/zhixun/app/rsi: {overlay}")

    repository, _, tag = args.tag.partition(":")
    repository, tag = repository or "zhixun-rsi-harness", tag or "0.15.0.dev1"
    write_build_manifest(overlay, environment, args.git_sha, args.git_dirty)

    build = output.parent / f".build-{output.stem}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    print("[1/4] base layer")
    base_tar = build / "base-layer.tar"
    base_diff = make_layer_tar(rootfs, base_tar, skip_names={".dockerenv"})
    print(f"      {base_tar.stat().st_size / 1e6:.1f} MB  diff_id={base_diff[:16]}...")

    print("[2/4] application layer")
    app_tar = build / "app-layer.tar"
    app_diff = make_layer_tar(overlay, app_tar)
    print(f"      {app_tar.stat().st_size / 1e6:.1f} MB  diff_id={app_diff[:16]}...")

    print("[3/4] image config")
    env = [f"{key}={value}" for key, value in environment.items()]
    config = {
        "architecture": "amd64",
        "os": "linux",
        "created": "2025-01-01T00:00:00Z",
        "config": {
            "Env": env,
            "Entrypoint": ["/opt/zhixun/bin/entrypoint.sh"],
            "Cmd": [],
            "WorkingDir": args.workdir,
            "ExposedPorts": {"7861/tcp": {}},
            "Labels": {
                "org.opencontainers.image.title": "zhixun-rsi-harness",
                "org.opencontainers.image.description":
                    "Controlled recursive self-improvement harness for drifting industrial models",
                "org.opencontainers.image.version": tag,
                "org.opencontainers.image.licenses": "Proprietary",
            },
        },
        "rootfs": {"type": "layers", "diff_ids": [f"sha256:{base_diff}",
                                                  f"sha256:{app_diff}"]},
        "history": [
            {"created": "2025-01-01T00:00:00Z",
             "created_by": "python:3.12-slim rootfs (crane export)"},
            {"created": "2025-01-01T00:00:00Z",
             "created_by": "zhixun RSI harness application layer"},
        ],
    }
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config_name = f"{config_sha}.json"
    (build / config_name).write_bytes(config_bytes)

    layer_ids = [f"{base_diff[:32]}{base_diff[:32]}", f"{app_diff[:32]}{app_diff[:32]}"]
    manifest = [{
        "Config": config_name,
        "RepoTags": [f"{repository}:{tag}", f"{repository}:latest"],
        "Layers": [f"{layer_ids[0]}/layer.tar", f"{layer_ids[1]}/layer.tar"],
    }]
    (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (build / "repositories").write_text(json.dumps(
        {repository: {tag: config_sha, "latest": config_sha}}, indent=2) + "\n")
    for layer_id, source, diff in ((layer_ids[0], base_tar, base_diff),
                                   (layer_ids[1], app_tar, app_diff)):
        layer_dir = build / layer_id
        layer_dir.mkdir()
        shutil.move(str(source), layer_dir / "layer.tar")
        (layer_dir / "VERSION").write_text("1.0")
        (layer_dir / "json").write_text(json.dumps(
            {"id": layer_id, "created": "2025-01-01T00:00:00Z",
             "container_config": {"Cmd": ["/bin/sh"]}}) + "\n")

    print("[4/4] assembling docker save tar")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as archive:
        for name in sorted(p.name for p in build.iterdir()):
            path = build / name
            info = archive.gettarinfo(str(path), arcname=name)
            _normalise(info)
            if path.is_dir():
                archive.addfile(info)
                for child in sorted(path.iterdir()):
                    child_info = archive.gettarinfo(str(child), arcname=f"{name}/{child.name}")
                    _normalise(child_info)
                    with child.open("rb") as handle:
                        archive.addfile(child_info, handle)
            else:
                with path.open("rb") as handle:
                    archive.addfile(info, handle)

    with tarfile.open(output) as archive:
        names = archive.getnames()
    required = {"manifest.json", "repositories", config_name,
                f"{layer_ids[0]}/layer.tar", f"{layer_ids[1]}/layer.tar"}
    missing = required - set(names)
    if missing:
        return sys.exit(f"invalid tar, missing: {sorted(missing)}")
    print(f"OK  {output}  {output.stat().st_size / 1e6:.1f} MB")
    print(f"    config: {config_name}")
    print(f"    layers: {layer_ids[0][:16]}..., {layer_ids[1][:16]}...")
    print(f"    tags:   {repository}:{tag}, {repository}:latest")
    shutil.rmtree(build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
