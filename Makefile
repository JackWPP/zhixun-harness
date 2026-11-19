# ============================================================================
#  智训引擎 RSI Harness —— 工程入口（v0.12.1）
#  新 harness 在 rsi/；harness/ 与 audit/ 是旧原型的审计留档（legacy-* 目标）。
# ============================================================================
SHELL := /bin/bash
PY    ?= ./.venv/bin/python
UV    ?= ./.venv/bin/uv
VENV  ?= .venv
STATE ?= .scratch/rsi-runs
TASKS ?= A,B,C
ROUNDS ?= 10

.PHONY: help venv test test-all selftest run run-offline run-llm run-kernel predict \
        dashboard probe synth external assets submission rehearse \
        image image-stage image-tar image-verify clean distclean \
        legacy-test legacy-run audit

help:                     ## 显示所有命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:                     ## 用 uv 创建 .venv（Python 3.12）并装依赖
	curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$$PWD/.scratch/uv/bin sh
	UV_CACHE_DIR=$$PWD/.scratch/uv/cache UV_PYTHON_INSTALL_DIR=$$PWD/.scratch/uv/python \
		.scratch/uv/bin/uv python install 3.12
	UV_CACHE_DIR=$$PWD/.scratch/uv/cache .scratch/uv/bin/uv venv $(VENV) \
		--python .scratch/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12
	VIRTUAL_ENV=$$PWD/$(VENV) UV_CACHE_DIR=$$PWD/.scratch/uv/cache \
		.scratch/uv/bin/uv pip install -r docker/requirements.txt pytest

test:                     ## 新 harness 单元测试
	$(PY) -m pytest tests -q

test-all: test            ## 新 harness + 旧原型测试
	$(PY) -m unittest discover -s harness/tests -q || true

selftest:                 ## 加载官方 checkpoint，复算三任务主指标
	@for t in A B C; do $(PY) -m rsi.cli selftest --task $$t; done

run:                      ## 完整闭环（默认 3 任务 × 10 轮，LLM=auto）
	$(PY) -m rsi.cli run --tasks $(TASKS) --rounds $(ROUNDS) --state-dir $(STATE)

run-offline:              ## 完全确定性离线闭环（无 LLM）
	$(PY) -m rsi.cli run --tasks $(TASKS) --rounds $(ROUNDS) --llm off --state-dir $(STATE)

run-llm:                  ## 接真实 LLM 端点（LLM_ENV 默认 .scratch/llm.env）
	$(PY) -m rsi.cli run --tasks $(TASKS) --rounds $(ROUNDS) --llm on \
		--llm-env $${LLM_ENV:-.scratch/llm.env} --state-dir $(STATE)

run-kernel:               ## 强制 Agent Kernel 控制器（需可用 LLM 端点）
	$(PY) -m rsi.cli run --tasks $(TASKS) --rounds $(ROUNDS) --llm on \
		--llm-env $${LLM_ENV:-.scratch/llm.env} --controller kernel --state-dir $(STATE)

predict:                  ## 官方兼容预测入口（CHECKPOINT=... POSTPROCESS=...）
	$(PY) -m rsi.predict --checkpoint $${CHECKPOINT:?set CHECKPOINT} \
		$${POSTPROCESS:+--postprocess $$POSTPROCESS} --mode $${MODE:-batch}

probe:                    ## 私有/隐藏风格 JSONL 外部评测（TASK=A PATH=... [CKPT=...]）
	$(PY) -m rsi.cli probe --task $${TASK:-A} --path $${PATH:?set PATH} \
		$${CKPT:+--checkpoint $$CKPT} $${OUTPUT:+--output $$OUTPUT}

synth:                    ## 生成合成漂移探针（TASK=A BATCHES=4 OUT=...）
	$(PY) -m rsi.cli synth --task $${TASK:-A} --batches $${BATCHES:-4} \
		--output $${OUT:?set OUT}

external:                 ## UCI 气体漂移逐批外部验证（CKPT=... [SUMMARY=...]）
	$(PY) scripts/external_validation.py --checkpoint $${CKPT:?set CKPT} \
		$${SUMMARY:+--postprocess $$SUMMARY} --output docs/evidence/uci_gas_drift.md

assets:                   ## 交付前资产自检（数据/checkpoint/技能/入口/DB）
	$(PY) scripts/verify_assets.py --json-out .scratch/assets.json

submission:               ## 从完成的 run 生成 <=64MiB 提交包（RUN_DIR=...）
	$(PY) scripts/make_submission.py --run-dir $${RUN_DIR:?set RUN_DIR}

dashboard:                ## 启动实时看板（STATE_JSON=... 或自动取最新 run）
	$(PY) -m rsi.dashboard --state $${STATE_JSON:-$$(ls -dt $(STATE)/*/state.json | head -n1)} \
		--port $${PORT:-7861}

image: image-stage image-tar   ## 一键构建 harness.tar（需先导出 rootfs）
image-stage:              ## 组装镜像 overlay（代码 + 数据 + 依赖）
	bash scripts/stage_image.sh

image-tar:                ## 生成 docker save 兼容 tar
	$(PY) scripts/build_image_tar.py --rootfs .scratch/image/rootfs \
		--overlay .scratch/image/overlay --output .scratch/image/harness.tar \
		--tag zhixun-rsi-harness:0.12.1

image-verify:             ## 解包 tar 并做镜像内 selftest（bwrap）
	bash scripts/verify_image.sh

rehearse:                 ## 完整彩排：构建镜像 + 解包 + 镜像内 2 轮闭环
	bash scripts/rehearse.sh

clean:                    ## 清理运行产物（保留 .venv 与 datasets）
	rm -rf runs/ .scratch/rsi-runs/ __pycache__ 2>/dev/null || true
	find rsi tests -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean          ## 清理 + 删除 virtualenv 与镜像中间产物
	rm -rf $(VENV) .scratch/image .scratch/uv

legacy-test:              ## 旧原型单元测试（审计留档，不参与交付）
	$(PY) -m unittest discover -s harness/tests -v

legacy-run:               ## 旧原型演示运行（审计留档）
	$(PY) -m harness run --task A --archive archives/cpu-a-guide-20261110-v1.zip \
		--output runs/A-$$(date +%Y%m%d-%H%M%S) --epochs 5 --max-candidates 3 --rounds 2

audit:                    ## 复现旧原型审计脚本的全部结论
	$(PY) audit/scripts/run_all.py
