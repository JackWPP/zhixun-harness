# Zhixun RSI Harness — evaluation image
#
# Build (if a Docker daemon is available):
#     docker build -f docker/Dockerfile -t zhixun-rsi-harness:0.3.0 .
#
# The competition only accepts a `docker save` tar, and the delivery machine used
# during development has no rootful Docker.  `scripts/build_image_tar.py` builds
# the exact same filesystem layout and emits a docker-load compatible
# `harness.tar` without a daemon; this file is kept as the canonical recipe.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/zhixun/site-packages:/opt/zhixun/app \
    RSI_DATA_ROOT=/opt/zhixun/app/datasets \
    RSI_GAS_ROOT=/opt/zhixun/app/extra/uci-gas/Dataset \
    RSI_STATE_DIR=/work/rsi-runs \
    RSI_TASKS=A,B,C \
    RSI_ROUNDS=10 \
    RSI_LLM=auto

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

COPY docker/entrypoint.sh /opt/zhixun/bin/entrypoint.sh
COPY docker/predict.sh /opt/zhixun/bin/rsi-predict
RUN chmod +x /opt/zhixun/bin/entrypoint.sh /opt/zhixun/bin/rsi-predict

# Work directory for experiment state.  The evaluation platform may mount its
# own hidden data and read the artifacts from here.
RUN mkdir -p /work/rsi-runs
VOLUME ["/work"]
EXPOSE 7861

ENTRYPOINT ["/opt/zhixun/bin/entrypoint.sh"]
CMD []
