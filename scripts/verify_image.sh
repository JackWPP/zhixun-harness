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
import json, sys, tarfile
from pathlib import Path
save, rootfs = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads((save / "manifest.json").read_text())
entry = manifest[0]
print(f"[verify] image config={entry['Config']} layers={len(entry['Layers'])}")
for layer in entry["Layers"]:
    with tarfile.open(save / layer) as tf:
        tf.extractall(rootfs, filter="tar")
print(f"[verify] merged rootfs: {rootfs}")
PY

ENTRY=$ROOTFS/opt/zhixun/bin/entrypoint.sh
test -x "$ENTRY" || { echo "[verify] missing entrypoint $ENTRY"; exit 1; }

run() {
    bwrap --bind "$ROOTFS" / \
          --dev /dev --proc /proc --tmpfs /tmp \
          --bind "$WORK" /work \
          --setenv RSI_STATE_DIR /work/rsi-runs \
          --chdir / "$@"
}

echo "[verify] selftest A"
run /opt/zhixun/bin/entrypoint.sh selftest A | tail -n 6

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
