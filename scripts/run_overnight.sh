#!/usr/bin/env bash
# 整夜编排脚本（本地开发用；不是评测入口）。
# 用途：按顺序跑完 严格 Track S → kernel 验证 → 外部测试床 → 契约自检 → 镜像重建，
# 每阶段独立日志；失败不阻塞后续可选项，但会在日志与最终 summary 中标注。
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./.venv/bin/python}
NIGHT=${NIGHT:-.scratch/night}
mkdir -p "$NIGHT"
LOG="$NIGHT/overnight-$(date +%Y%m%d-%H%M%S).log"
echo "== overnight start $(date -Is) log=$LOG" | tee -a "$LOG"

stage() {
  local name="$1"; shift
  echo "== [$(date +%H:%M:%S)] stage: $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "== [$(date +%H:%M:%S)] stage $name exit=$rc" | tee -a "$LOG"
  return $rc
}

SKIP_STRICT=${SKIP_STRICT:-0}
SKIP_KERNEL=${SKIP_KERNEL:-0}
SKIP_TESTBED=${SKIP_TESTBED:-0}
SKIP_CONTRACT=${SKIP_CONTRACT:-0}
SKIP_IMAGE=${SKIP_IMAGE:-0}

if [ "$SKIP_STRICT" = 0 ]; then
  stage strict-3x10 env RSI_ALLOW_EXTENDED_DATA=0 \
    "$PY" -m rsi.cli run --tasks A,B,C --rounds 10 --llm off --controller pipeline \
    --state-dir "$NIGHT/strict" || true
fi

if [ "$SKIP_KERNEL" = 0 ] && [ -f .scratch/llm.env ]; then
  stage kernel-a-3 env RSI_ALLOW_EXTENDED_DATA=0 \
    "$PY" -m rsi.cli run --tasks A --rounds 3 --llm on --llm-env .scratch/llm.env \
    --controller kernel --state-dir "$NIGHT/kernel" --wall-cap 5400 \
    --experiments-cap 8 --round-turns 16 || true
fi

if [ "$SKIP_TESTBED" = 0 ]; then
  stage testbed-x-4 env RSI_ALLOW_EXTENDED_DATA=1 \
    "$PY" -m rsi.cli stream --tasks A --source external-a --batches 4 \
    --external-root extra/uci-gas/Dataset --holdout 0.5 --experiments-cap 2 \
    --llm off --state-dir "$NIGHT/testbed-x" || true
fi

if [ "$SKIP_CONTRACT" = 0 ] && [ -f scripts/contract_check.py ]; then
  stage contract-check "$PY" scripts/contract_check.py --json-out "$NIGHT/contract.json" || true
fi

if [ "$SKIP_IMAGE" = 0 ]; then
  stage image-build bash -c 'bash scripts/stage_image.sh && '"$PY"' scripts/build_image_tar.py && bash scripts/verify_image.sh' || true
fi

echo "== overnight done $(date -Is)" | tee -a "$LOG"
