# ============================================================================
#  智训引擎 Harness —— 工程入口
#  所有命令都可以从仓库根目录直接跑；不依赖任何全局状态。
# ============================================================================
SHELL      := /bin/bash
PY         ?= python3
VENV       ?= .venv
ARCHIVE_A  := archives/cpu-a-guide-20261110-v1.zip
ARCHIVE_B  := archives/cpu-b-rul-20261110-v1.zip
ARCHIVE_C  := archives/cpu-c-guide-20261110-v1.zip
export HARNESS_TEST_VENDOR_ARCHIVE := $(CURDIR)/$(ARCHIVE_A)

.PHONY: help setup unpack test verify run-a run-b run-c run-all audit clean nuke ci

help:                     ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:                    ## 创建虚拟环境并安装 harness（可编辑模式）
	$(PY) -m venv $(VENV)
	$(VENV)/bin/$(PY) -m pip install -q --upgrade pip
	$(VENV)/bin/$(PY) -m pip install -q -e .
	@echo "✅ 完成。使用前先执行: source $(VENV)/bin/activate"

unpack:                   ## 从 archives/ 重新解包三个数据包到 datasets/
	@for z in $(ARCHIVE_A) $(ARCHIVE_B) $(ARCHIVE_C); do \
		name=$$(basename $$z .zip); \
		echo "→ $$name"; \
		rm -rf datasets/$$name; \
		unzip -q -o $$z -d datasets/; \
	done
	@echo "✅ datasets/ 已重建"

test:                     ## 运行 harness 单元测试（30 项）
	$(PY) -m unittest discover -s harness/tests -v

verify:                   ## 校验原始压缩包摘要 + 复算 demo 交付指标
	$(PY) audit/scripts/00_verify_bundle.py

run-a:                    ## 在任务 A（气体识别）上跑一轮 harness
	$(PY) -m harness run --task A --archive $(ARCHIVE_A) \
		--output runs/A-$$(date +%Y%m%d-%H%M%S) --epochs 5 --max-candidates 3 --rounds 2

run-b:                    ## 在任务 B（轴承 RUL）上跑一轮 harness
	$(PY) -m harness run --task B --archive $(ARCHIVE_B) \
		--output runs/B-$$(date +%Y%m%d-%H%M%S) --epochs 5 --max-candidates 3 --rounds 2

run-c:                    ## 在任务 C（发动机 RUL）上跑一轮 harness
	$(PY) -m harness run --task C --archive $(ARCHIVE_C) \
		--output runs/C-$$(date +%Y%m%d-%H%M%S) --epochs 5 --max-candidates 3 --rounds 2 --max-train-seconds 200

run-all: run-a run-b run-c  ## 三个任务各跑一轮

audit:                    ## 复现审计脚本的全部结论
	$(PY) audit/scripts/run_all.py

ci: test verify           ## CI 入口

clean:                    ## 清理运行时产物
	rm -rf runs/ audit/.tmp/
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

nuke: clean               ## 清理 + 删除虚拟环境与解包数据
	rm -rf $(VENV) datasets/cpu-*
