#!/usr/bin/env python3
"""run_all — 依次执行全部审计脚本，汇总通过/失败。

用法：
    python3 audit/scripts/run_all.py          # 全部跑一遍
    python3 audit/scripts/run_all.py 04 05    # 只跑指定编号
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "00_verify_bundle.py",
    "01_data_quality.py",
    "02_baseline_metrics.py",
    "03_constant_control.py",
    "04_seed_noise.py",
    "05_capacity_sweep.py",
    "06_drift_probe.py",
    "07_rul_cap_probe.py",
    "08_demo_forensics.py",
]


def main(argv: list) -> int:
    selected = [s for s in SCRIPTS if not argv or any(s.startswith(a) for a in argv)]
    if not selected:
        print(f"没有匹配的脚本：{argv}")
        return 2

    failures = []
    total = time.perf_counter()
    for name in selected:
        path = HERE / name
        print(f"\n{'#' * 78}\n### {name}\n{'#' * 78}")
        started = time.perf_counter()
        result = subprocess.run([sys.executable, str(path)], cwd=str(HERE.parents[1]))
        elapsed = time.perf_counter() - started
        status = "✅" if result.returncode == 0 else f"❌ (exit {result.returncode})"
        print(f"\n### {name} {status}  {elapsed:.1f}s")
        if result.returncode != 0:
            failures.append(name)

    print("\n" + "=" * 78)
    print(f"总计 {len(selected)} 个脚本，耗时 {time.perf_counter() - total:.1f}s")
    if failures:
        print(f"❌ 失败：{failures}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
