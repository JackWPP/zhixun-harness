#!/bin/bash
# Verify the docker-save tar without a Docker daemon:
#   1) unpack config/manifest/layers into a merged rootfs
#   2) run the image entrypoint inside bwrap: selftest + a short offline run
#   3) assert artifacts exist and contain the expected fields
#
#   bash scripts/verify_image.sh [TAR] [WORKDIR]
#
# Requires: bwrap (rootless user namespaces).
set -euo pipefail
cd "$(dirname "$0")/.."

TAR=${1:-.scratch/image/harness.tar}
BASE=.scratch/image/verify
SAVE=$BASE/save
ROOTFS=$BASE/rootfs
WORK=$BASE/work

echo "[verify] tar: $TAR"
rm -rf "$BASE"
mkdir -p "$SAVE" "$ROOTFS" "$WORK"

tar -xf "$TAR" -C "$SAVE"
python3 - "$SAVE" "$ROOTFS" <<'PY'
import importlib.util, json, sys, tarfile
from pathlib import Path
save, rootfs = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads((save / "manifest.json").read_text())
entry = manifest[0]
print(f"[verify] image config={entry['Config']} layers={len(entry['Layers'])}")
for layer in entry["Layers"]:
    with tarfile.open(save / layer) as tf:
        tf.extractall(rootfs, filter="tar")
print(f"[verify] merged rootfs: {rootfs}")
config = json.loads((save / entry["Config"]).read_text())
(save / "image-env.json").write_text(json.dumps(config["config"]["Env"]))
spec = importlib.util.spec_from_file_location("image_builder", "scripts/build_image_tar.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
identity = json.loads((rootfs / builder.MANIFEST_PATH).read_text())
digest, count = builder.source_digest(rootfs)
assert identity["source_sha256"] == digest, "build source digest mismatch"
assert identity["source_file_count"] == count, "build source file count mismatch"
assert identity["sdk"]["version"] == builder.SDK_VERSION, "build SDK mismatch"
environment = dict(item.split("=", 1) for item in config["config"]["Env"])
assert identity["environment"] == {key: environment[key] for key in builder.DEFAULT_ENV}
print(f"[verify] source={digest[:16]} sdk={identity['sdk']['version']} "
      f"commit={identity['git_commit'] or 'unknown'} dirty={identity['git_dirty']}")
PY

# bwrap does not interpret Docker config ENV. Apply it explicitly so the
# verification run exercises the same defaults as the submitted image.
IMAGE_ENV=()
while IFS= read -r -d '' item; do IMAGE_ENV+=("$item"); done < <(
    python3 - "$SAVE/image-env.json" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1])):
    key, value = item.split("=", 1)
    sys.stdout.buffer.write(b"--setenv\0" + key.encode() + b"\0" + value.encode() + b"\0")
PY
)

ENTRY=$ROOTFS/opt/zhixun/bin/entrypoint.sh
test -x "$ENTRY" || { echo "[verify] missing entrypoint $ENTRY"; exit 1; }

run() {
    bwrap --clearenv --bind "$ROOTFS" / \
          --dev /dev --proc /proc --tmpfs /tmp \
          --bind "$WORK" /work \
          "${IMAGE_ENV[@]}" \
          --setenv RSI_STATE_DIR /work/rsi-runs \
          --chdir /opt/zhixun/app "$@"
}

echo "[verify] selftest A"
run /opt/zhixun/bin/entrypoint.sh selftest A | tail -n 6

echo "[verify] workspace v2 describe (entrypoint dispatch)"
DESCRIBE=$(run /opt/zhixun/bin/entrypoint.sh describe)
echo "$DESCRIBE" | head -n 2
echo "$DESCRIBE" | grep -q '"protocol":"contest.harness.workspace.v2"' || { echo "[verify] describe protocol mismatch"; exit 1; }
echo "$DESCRIBE" | grep -q '"tasks":\["classification","regression"\]' || { echo "[verify] describe tasks mismatch"; exit 1; }
echo "$DESCRIBE" | grep -q '"sdk_version":"1.0.5"' || { echo "[verify] describe SDK mismatch"; exit 1; }
DIRECT_DESCRIBE=$(run python3 /app/harness.py describe)
test "$DIRECT_DESCRIBE" = "$DESCRIBE" || { echo "[verify] direct Python entry differs"; exit 1; }
test -f "$ROOTFS/app/harness.py" || { echo "[verify] missing /app/harness.py"; exit 1; }
test -f "$ROOTFS/app/contest_submission.py" || { echo "[verify] missing /app/contest_submission.py"; exit 1; }
test -d "$ROOTFS/opt/zhixun/app/contest_harness" || { echo "[verify] missing vendored contest_harness"; exit 1; }

echo "[verify] workspace v2 fake-host round (A, train+predict)"
run python3 /opt/zhixun/app/scripts/emulate_workspace_v2.py --task A --rounds 1 2>&1 | tail -n 6

echo "[verify] offline run A (2 rounds)"
run /opt/zhixun/bin/entrypoint.sh run --tasks A --rounds 2 --llm off | tail -n 8

REPORT=$(ls -1t "$WORK"/rsi-runs/*/workspace_report.json 2>/dev/null | head -n1)
SUMMARY=$(ls -1t "$WORK"/rsi-runs/*/summary.json 2>/dev/null | head -n1)
test -f "$REPORT" || { echo "[verify] missing $REPORT"; exit 1; }
test -f "$SUMMARY" || { echo "[verify] missing $SUMMARY"; exit 1; }

python3 - "$REPORT" "$SUMMARY" "$WORK" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
summary = json.loads(Path(sys.argv[2]).read_text())
assert report.get("datasets"), "workspace_report.datasets missing"
ds = report["datasets"][0]
assert ds.get("rounds"), "dataset rounds missing"
assert ds["rounds"][0].get("decision") in {"promote", "retain", "rollback"}
assert ds["rounds"][0].get("parent_checkpoint_sha256")
assert Path(sys.argv[3], "workspace_report.json").exists()
print(f"[verify] workspace_report ok: task={ds.get('key')} rounds={len(ds['rounds'])} "
      f"metric={ds.get('metrics', {}).get('primary')}")
print(f"[verify] summary keys: {sorted(summary)[:8]}")
PY

echo "[verify] PASS  artifacts: $(dirname "$REPORT")"
