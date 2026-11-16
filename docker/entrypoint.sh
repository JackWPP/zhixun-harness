#!/bin/sh
# Container entrypoint for the Zhixun RSI harness.
#
#   docker run --rm -v /host/work:/work zhixun-rsi-harness:0.3.0
#   docker run --rm -v /host/work:/work zhixun-rsi-harness:0.3.0 run --tasks A --rounds 3 --llm off
#   docker run --rm -i zhixun-rsi-harness:0.3.0 predict --checkpoint ... < inputs.jsonl
#
# The evaluation platform only requires the container to run unattended and to
# leave reproducible artifacts under $RSI_STATE_DIR; every knob below has a
# deterministic default, so `docker run <image>` is a complete run.
set -eu

APP_HOME=${RSI_HOME:-/opt/zhixun/app}
STATE_DIR=${RSI_STATE_DIR:-/work/rsi-runs}
export PYTHONPATH="/opt/zhixun/site-packages:$APP_HOME${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export RSI_DATA_ROOT=${RSI_DATA_ROOT:-$APP_HOME/datasets}
export RSI_GAS_ROOT=${RSI_GAS_ROOT:-$APP_HOME/extra/uci-gas/Dataset}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# Optional endpoint configuration file mounted by the operator.  It is never
# baked into the image; values are read with KEY=VALUE semantics.
if [ -n "${RSI_LLM_ENV:-}" ] && [ -f "${RSI_LLM_ENV}" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${RSI_LLM_ENV}"
    set +a
fi

mkdir -p "$STATE_DIR"

COMMAND=${1:-run}
if [ "$#" -gt 0 ]; then shift; fi

case "$COMMAND" in
    run)
        TASKS=${RSI_TASKS:-A,B,C}
        ROUNDS=${RSI_ROUNDS:-10}
        LLM=${RSI_LLM:-auto}
        SEED=${RSI_SEED:-42}
        EXTRA_ARGS=""
        if [ -n "${RSI_LLM_ENV:-}" ] && [ -f "${RSI_LLM_ENV}" ]; then
            EXTRA_ARGS="--llm-env $RSI_LLM_ENV"
        fi
        if [ "${RSI_SWANLAB:-0}" = "1" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --swanlab"
        fi
        if [ "${RSI_VERIFY_ASSETS:-1}" = "1" ]; then
            echo "[harness] self-check: datasets, checkpoints, skills, predict entry"
            python "$APP_HOME/scripts/verify_assets.py" --json-out "$STATE_DIR/assets-check.json"
        fi
        CONTROLLER=${RSI_CONTROLLER:-auto}
        # Strict compliance default: scored runs fit only the official train
        # split. The image still bundles the public UCI gas archive for the
        # opt-in ablation / extended-data validation (`RSI_ALLOW_EXTENDED_DATA=1`,
        # documented in docs/11_COMPLIANCE.md §11.9 and docs/15_DATA.md).
        export RSI_ALLOW_EXTENDED_DATA="${RSI_ALLOW_EXTENDED_DATA:-0}"
        TOKEN_CAP=${RSI_TOKEN_CAP:-}
        [ -n "$TOKEN_CAP" ] && EXTRA_ARGS="$EXTRA_ARGS --token-cap $TOKEN_CAP"
        echo "[harness] tasks=$TASKS rounds=$ROUNDS llm=$LLM controller=$CONTROLLER state=$STATE_DIR"
        python -m rsi.cli run --tasks "$TASKS" --rounds "$ROUNDS" --llm "$LLM" \
            --controller "$CONTROLLER" --seed "$SEED" --state-dir "$STATE_DIR" $EXTRA_ARGS "$@"
        # Publish the latest run in a fixed location for automated collection.
        LATEST=$(ls -dt "$STATE_DIR"/*/ 2>/dev/null | head -n 1 || true)
        if [ -n "$LATEST" ]; then
            # Relative link stays valid when the state dir is bind-mounted on the host.
            ln -sfn "$(basename "$LATEST")" "$STATE_DIR/latest"
            if [ -f "$LATEST/workspace_report.json" ]; then
                cp "$LATEST/workspace_report.json" /work/workspace_report.json
            fi
            if [ -f "$LATEST/summary.json" ]; then
                cp "$LATEST/summary.json" /work/summary.json
            fi
            echo "[harness] artifacts: $LATEST"
        fi
        ;;
    stream)
        # Batch-stream mode: every incoming batch is trained/evaluated/decided in
        # turn, then predictions + decision are written for the platform to pick up.
        SOURCE=${RSI_STREAM_SOURCE:-official}
        BATCHES=${RSI_STREAM_BATCHES:-10}
        TASKS=${RSI_TASKS:-A,B,C}
        HOLDOUT=${RSI_STREAM_HOLDOUT:-0.3}
        STREAM_DIR=${RSI_STREAM_DIR:-$STATE_DIR/stream}
        EXTRA_ARGS=""
        [ -n "${RSI_LLM_ENV:-}" ] && [ -f "${RSI_LLM_ENV}" ] && EXTRA_ARGS="--llm-env $RSI_LLM_ENV"
        # History adaptation: continue-train on the cumulative labelled history of
        # earlier batches before scoring the current one (no current-batch labels).
        ADAPT_ARGS=""
        if [ "${RSI_STREAM_ADAPT:-1}" = "1" ]; then
            ADAPT_ARGS="--adapt-history --adapt-epochs ${RSI_STREAM_ADAPT_EPOCHS:-30} --adapt-lr ${RSI_STREAM_ADAPT_LR:-0.0005} --adapt-weight ${RSI_STREAM_ADAPT_WEIGHT:-0.5}"
        fi
        if [ "${RSI_STREAM_PAIR:-1}" = "1" ]; then
            ADAPT_ARGS="$ADAPT_ARGS --pair ${RSI_STREAM_PAIR_CLASSES:-0,3}"
        else
            ADAPT_ARGS="$ADAPT_ARGS --no-pair-calibration"
        fi
        echo "[harness] stream source=$SOURCE batches=$BATCHES tasks=$TASKS state=$STREAM_DIR adapt=${RSI_STREAM_ADAPT:-1}"
        python -m rsi.cli stream --source "$SOURCE" --tasks "$TASKS" --batches "$BATCHES" \
            --holdout "$HOLDOUT" --llm "${RSI_LLM:-off}" --controller "${RSI_CONTROLLER:-auto}" \
            --state-dir "$STREAM_DIR" $ADAPT_ARGS $EXTRA_ARGS "$@"
        ;;
    predict)
        python -m rsi.predict "$@"
        ;;
    selftest)
        python -m rsi.cli selftest --task "${1:-A}"
        ;;
    serve)
        exec python -m rsi.dashboard --state "${1:?state.json required}" \
            --port "${2:-7861}" --host 0.0.0.0
        ;;
    shell)
        exec /bin/sh
        ;;
    *)
        exec "$COMMAND" "$@"
        ;;
esac
