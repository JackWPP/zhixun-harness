# Zhixun RSI Harness — evaluation image
#
# Build (if a Docker daemon is available):
#     docker build -f docker/Dockerfile --build-arg RSI_GIT_SHA=<commit> \
#       --build-arg RSI_GIT_DIRTY=false -t zhixun-rsi-harness:0.15.0.dev1 .
#
# The competition only accepts a `docker save` tar, and the delivery machine used
# during development has no rootful Docker.  `scripts/build_image_tar.py` builds
# the exact same filesystem layout and emits a docker-load compatible
# `harness.tar` without a daemon; this file is kept as the canonical recipe.
FROM python:3.12-slim
ARG RSI_GIT_SHA=""
ARG RSI_GIT_DIRTY=unknown

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/zhixun/site-packages:/opt/zhixun/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    BASELINE_CPU_THREADS=1 \
    RSI_BUILD_MANIFEST=/opt/zhixun/app/build-manifest.json \
    RSI_DATA_ROOT=/opt/zhixun/app/datasets \
    RSI_GAS_ROOT=/opt/zhixun/app/extra/uci-gas/Dataset \
    RSI_STATE_DIR=/work/rsi-runs \
    RSI_TASKS=A,B,C \
    RSI_ROUNDS=10 \
    RSI_LLM=auto \
    RSI_V2_CONTROLLER=auto \
    RSI_V2_MAX_TURNS=10 \
    RSI_V2_CAPACITY_ARM=ls10 \
    RSI_V2_NATIVE_TOOLS=1 \
    RSI_V2_KERNEL_THINKING=off \
    RSI_V2_MAX_OUTPUT=256 \
    RSI_V2_KERNEL_TOOLS=core \
    RSI_V2_WARMUP_ROUNDS=0 \
    RSI_V2_B_RETAIN=0 \
    RSI_V2_B_STATIC=0 \
    RSI_V2_B_EPOCHS=15 \
    RSI_V2_B_LR=0.001 \
    RSI_V2_EXPERIMENT_CACHE=1 \
    RSI_V2_SCRATCH_ARM=off \
    RSI_V2_C_EFFICIENT=1 \
    RSI_V2_C_MAX_ALPHA=0.25 \
    RSI_V2_PROBE_RETAIN=""

WORKDIR /opt/zhixun/app

COPY docker/requirements.txt /opt/zhixun/requirements.txt
RUN pip install --no-cache-dir --target /opt/zhixun/site-packages -r /opt/zhixun/requirements.txt

COPY rsi /opt/zhixun/app/rsi
# The runtime uses rsi.model/data_official. Reference baselines are retained
# inside the task packages under datasets/, not as top-level source folders.
COPY skills /opt/zhixun/app/skills
COPY scripts /opt/zhixun/app/scripts
COPY docs /opt/zhixun/app/docs
COPY extra /opt/zhixun/app/extra
COPY datasets /opt/zhixun/app/datasets
COPY tasks /opt/zhixun/app/tasks
COPY pyproject.toml README.md Makefile /opt/zhixun/app/

# Workspace v2 participant runtime (platform-provided SDK, vendored unpinned-free)
COPY vendor/contest_harness-1.0.5/contest_harness /opt/zhixun/app/contest_harness
COPY docker/app/harness.py docker/app/contest_submission.py /app/

COPY docker/entrypoint.sh /opt/zhixun/bin/entrypoint.sh
COPY docker/predict.sh /opt/zhixun/bin/rsi-predict
RUN chmod +x /opt/zhixun/bin/entrypoint.sh /opt/zhixun/bin/rsi-predict
RUN python3 scripts/build_image_tar.py --write-build-manifest / \
    --git-sha "$RSI_GIT_SHA" --git-dirty "$RSI_GIT_DIRTY"

# Work directory for experiment state.  The evaluation platform may mount its
# own hidden data and read the artifacts from here.
RUN mkdir -p /work/rsi-runs
VOLUME ["/work"]
EXPOSE 7861

ENTRYPOINT ["/opt/zhixun/bin/entrypoint.sh"]
CMD []
