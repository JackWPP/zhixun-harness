# 07 · 移交与上手（服务器/新同学必读）

> 目标：一个从没碰过这个项目的人，**30 分钟内能跑出全部数字**。

## 1. 环境

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.10 – 3.12**（推荐 3.11） | 本地验证用的是 3.9.6 也能跑，但评委环境是 3.10+ |
| numpy | **1.26.4 ≤ v < 3** | 2.0.2 已验证；不要用 numpy 1.24 以下的版本 |
| 内存 | ≥ 2 GB | C 任务诊断时要对 113,089×24 做一次前向 |
| 磁盘 | ≥ 500 MB | 仓库 + 解包数据 |
| GPU | **不需要** | 全程纯 NumPy CPU |
| 网络 | **运行不需要** | 只有 clone 和 pip install 需要 |

**没有其他依赖。** 没有 pandas / sklearn / torch / scipy。这是刻意的设计选择，
也是这个项目最容易被低估的优点：**它能在一台没有 GPU 的评测机上跑完三个任务，总共 27 秒。**

## 2. 三步跑起来

```bash
git clone <repo-url> zhixun-harness && cd zhixun-harness

# ① 装依赖（二选一）
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
#    或者： pip install "numpy>=1.26.4,<3"

# ② 重建数据集
make unpack

# ③ 验证一切正常
make ci        # = make test（30 项单元测试）+ make verify（基线/交付物逐位复算）
```

如果 `make ci` 全绿，环境就对了。

## 3. 常用命令

```bash
make help          # 列出所有目标
make test          # 30 项单元测试（会自动设置 HARNESS_TEST_VENDOR_ARCHIVE）
make verify        # 复算基线分数 + demo 分数 + 校验 5 个 zip 的 sha256
make audit         # 跑全部 13 个审计脚本，产出 docs 里所有数字（约 3 分钟）
make run-a         # 在 A 任务上跑一次 harness（约 1.4 s）
make run-b         # B（约 2.2 s）
make run-c         # C（约 23 s）
make run-all       # 三个都跑
make clean         # 清临时产物
```

**输出位置**：`make run-*` 写到 `runs/<TASK>-<时间戳>/`；`make audit` 打印到 stdout。

## 4. 目录导航

```
.
├── archives/            5 个原始 zip（只读，444）+ SHA256SUMS
├── datasets/            3 个解包后的赛题包（58 MB）
│   ├── cpu-a-guide-20261110-v1/     任务 A：气体传感器
│   ├── cpu-b-rul-20261110-v1/       任务 B：轴承 RUL
│   └── cpu-c-guide-20261110-v1/     任务 C：发动机 RUL
├── harness/             自进化原型源码（13 文件，v0.2）
│   ├── strategies.py    6 个配方 + 规则路由 + 策略小模型
│   ├── training.py      受控续训内核
│   ├── runner.py        两轮控制循环 + 门控
│   ├── diagnostics.py   阈值诊断
│   ├── data.py          压缩包导入 + 角色隔离
│   ├── reporting.py     静态 HTML 报告
│   └── tests/           30 项单元测试
├── demo/                团队原始交付物（3 次运行的完整轨迹）
│   └── harness-runs/{A,B,C}-v1/
├── audit/
│   ├── scripts/         13 个可复现脚本（00–09 + run_all）
│   └── results/         独立重跑的输出
├── docs/                本套文档（01–09）
├── Makefile
└── pyproject.toml
```

## 5. 七个坑（踩过一次就不用再踩）

### 坑 1：`BASELINE_CPU_THREADS` 必须在 import numpy 之前设置
`datasets/*/baselines/model.py` 在模块顶层读环境变量并设置
`OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS`。
**如果 numpy 已经 import 了，这个设置无效。** 所以：
```bash
BASELINE_CPU_THREADS=1 python3 ...
```
不要在多线程代码里动态改。默认值 1（可复现性优先），允许范围 1–8。

### 坑 2：测试需要 `HARNESS_TEST_VENDOR_ARCHIVE`
`harness/tests/test_harness.py:37` 和 `test_runner.py:19` 里写着原作者机器的路径
`C:/Users/zhao jiahao/Downloads/cpu-a-guide-20261110-v1.zip`。
**不要改代码**，设环境变量即可（Makefile 已自动做）：
```bash
export HARNESS_TEST_VENDOR_ARCHIVE="$(pwd)/archives/cpu-a-guide-20261110-v1.zip"
```

### 坑 3：`Model.save()` 用 `"xb"` 模式，目标文件已存在会抛异常
自己写脚本复现实验时，保存前先 `path.unlink(missing_ok=True)`。
`audit/scripts/04_seed_noise.py` 里有示例。

### 坑 4：`execute_candidate()` 会自己建目录
`harness/training.py:470` 里 `directory.mkdir()`，
**不要预先创建候选目录**，否则 `FileExistsError`。

### 坑 5：`prepare_bundle()` 的 `source` 目录必须是全新的
它对每个 seed 轮次都要重新调用一次，且要求 `output` 目录不存在。

### 坑 6：`meta["paths"]` 必须是 `list[str]`，每个元素是 JSON 字符串
`Model.validate()` 会检查。手工造 checkpoint 时最容易在这里翻车。

### 坑 7：macOS 的 grep 不支持 `-P`
Makefile 和脚本里全部用 `grep -E`。Linux 服务器上两者都能跑。

## 6. 数据重建

`datasets/` 是从 `archives/` 解出来的。如果数据坏了或者想验证完整性：

```bash
# 校验 5 个 zip 的 sha256
shasum -a 256 -c archives/SHA256SUMS

# 重建 datasets/（会先删除现有目录）
make unpack
```

`archives/` 里的 zip 权限是 444（只读），**不要 chmod 改掉**——
这是为了确保任何人都不会"顺手改一下原始数据"。

## 7. 在服务器上长期跑

```bash
git clone <repo-url> && cd zhixun-harness
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e .
make unpack
make ci
nohup python3 audit/scripts/run_all.py > audit.log 2>&1 &
```

三个任务的完整审计约 **3 分钟**（C 的 bootstrap 和容量扫描是大头）。
如果只要快速冒烟测试：
```bash
python3 audit/scripts/run_all.py 00 01 08
```

## 8. 移交检查单

- [ ] `make ci` 全绿（30 测试 + verify）
- [ ] `make run-all` 三个任务都能跑完
- [ ] `make audit` 的数字与 `docs/05_findings.md` 一致
- [ ] 仓库能 clone 到一个干净目录并跑通（**在另一台机器上验证**）
- [ ] 明确仓库是 private 还是 public，以及数据能否外传
- [ ] 指定一位同学负责 `docs/08_roadmap.md` 的第一项
