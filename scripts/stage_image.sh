#!/bin/bash
# Stage the image overlay (application code + dependencies + entrypoints).
#
#   bash scripts/stage_image.sh [OVERLAY_DIR]
#
# Requires the base rootfs already exported to .scratch/image/rootfs, e.g.
#   crane export python:3.12-slim | tar -x --no-same-owner -C .scratch/image/rootfs
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=${RSI_IMAGE_STAGE:-.scratch/image}
OVERLAY=${1:-$ROOT/overlay}
APP=$OVERLAY/opt/zhixun/app
SITE=$OVERLAY/opt/zhixun/site-packages
BIN=$OVERLAY/opt/zhixun/bin

echo "[stage] overlay: $OVERLAY"
rm -rf "$OVERLAY"
mkdir -p "$APP" "$SITE" "$BIN"

# 1. harness code and official data packages
for item in rsi baselines config skills knowledge scripts docs extra datasets tasks; do
    if [ -d "$item" ]; then
        mkdir -p "$APP/$item"
        rsync -a --exclude '__pycache__' --exclude '*.pyc' "$item/" "$APP/$item/"
    fi
done
for item in pyproject.toml README.md Makefile; do
    [ -f "$item" ] && cp -a "$item" "$APP/"
done

# 2. extended-validation data (UCI Gas Sensor Array Drift, CC BY 4.0)
GAS_SRC=${RSI_GAS_SOURCE:-}
if [ -z "$GAS_SRC" ]; then
    if [ -d "extra/uci-gas/Dataset" ]; then
        GAS_SRC=extra/uci-gas/Dataset
    else
        GAS_SRC=.scratch/gas/Dataset
    fi
fi
if [ -d "$GAS_SRC" ]; then
    mkdir -p "$APP/extra/uci-gas/Dataset"
    cp -a "$GAS_SRC"/batch*.dat "$APP/extra/uci-gas/Dataset/"
else
    echo "[stage] WARNING: $GAS_SRC missing; extended validation data not bundled" >&2
fi

# 3. entrypoints
cp docker/entrypoint.sh "$BIN/entrypoint.sh"
cp docker/predict.sh "$BIN/rsi-predict"
chmod +x "$BIN/entrypoint.sh" "$BIN/rsi-predict"

# 4. python dependencies (wheels for the same 3.12 ABI as the base image)
UV=${UV_BIN:-.scratch/uv/bin/uv}
if [ -x "$UV" ]; then
    export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/.scratch/uv/cache}
    export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$PWD/.scratch/uv/python}
    export UV_TOOL_DIR=${UV_TOOL_DIR:-$PWD/.scratch/uv/tools}
    "$UV" pip install --target "$SITE" -r docker/requirements.txt
else
    echo "[stage] ERROR: uv not found at $UV" >&2
    exit 1
fi

du -sh "$APP" "$SITE" "$OVERLAY"
echo "[stage] done"
