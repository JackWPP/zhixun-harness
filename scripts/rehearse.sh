#!/usr/bin/env bash
# End-to-end rehearsal: build the image tar, verify it under bwrap, optionally
# run the full 3x10 loop inside the extracted image.
#
#   bash scripts/rehearse.sh            # build + extract + selftest + 2-round run
#   bash scripts/rehearse.sh --full     # additionally A,B,C x 10 rounds (walls: ~20 min)
#   bash scripts/rehearse.sh --skip-build   # reuse .scratch/image/harness.tar
set -euo pipefail
cd "$(dirname "$0")/.."

FULL=0
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

TAR=.scratch/image/harness.tar
if [ "$SKIP_BUILD" = "0" ]; then
    echo "== stage overlay =="
    bash scripts/stage_image.sh
    echo "== build docker-save tar =="
    ./.venv/bin/python scripts/build_image_tar.py \
        --rootfs .scratch/image/rootfs \
        --overlay .scratch/image/overlay \
        --output "$TAR" \
        --tag zhixun-rsi-harness:0.4.0
fi
[ -f "$TAR" ] || { echo "missing $TAR (run without --skip-build)" >&2; exit 1; }
echo "== verify extracted image (selftest + 2-round run) =="
bash scripts/verify_image.sh "$TAR"

if [ "$FULL" = "1" ]; then
    echo "== full rehearsal: A,B,C x 10 rounds inside the extracted image =="
    ROOTFS=.scratch/image/verify/rootfs
    WORK=.scratch/image/verify/work-full
    mkdir -p "$WORK"
    ./.venv/bin/python - "$ROOTFS" "$WORK" <<'EOF'
import os, subprocess, sys
rootfs, work = sys.argv[1], sys.argv[2]
cmd = ["bwrap", "--bind", rootfs, "/", "--dev", "/dev", "--proc", "/proc",
       "--bind", work, "/work", "--chdir", "/opt/zhixun/app",
       "--setenv", "HOME", "/work",
       "/opt/zhixun/bin/entrypoint.sh", "run"]
env = dict(os.environ, RSI_TASKS="A,B,C", RSI_ROUNDS="10", RSI_LLM="off",
           RSI_STATE_DIR="/work/rsi-runs", RSI_SEED="42")
raise SystemExit(subprocess.call(cmd, env=env))
EOF
    echo "== key artifacts =="
    find "$WORK" -maxdepth 3 -name 'workspace_report.json' -o -maxdepth 3 -name 'summary.json' | sort
fi

echo "== done: image tar =="
ls -lh "$TAR"
