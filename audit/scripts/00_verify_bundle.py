#!/usr/bin/env python3
"""00 — 校验原始压缩包完整性，并独立复算 demo 交付指标。

做两件事：
  1. 对 archives/ 下每个 zip 计算 sha256，写入/比对 archives/SHA256SUMS。
  2. 用本仓库自己的评估代码重新计算 demo/harness-runs/*/selected/checkpoint.npz
     在公开验证集上的指标，与 demo 自己 summary.json 里记录的值对比。

用途：证明"移交到服务器后拿到的数据没变、交付物里的数字是真的"。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C

SUMS = C.ARCHIVES / "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives() -> bool:
    C.header("1. 原始压缩包完整性")
    records = []
    for path in sorted(C.ARCHIVES.glob("*.zip")):
        records.append((sha256(path), path.stat().st_size, path.name))
        print(f"  {path.name:<34} {records[-1][1]:>10,} B  {records[-1][0]}")

    if SUMS.exists():
        expected = {}
        for line in SUMS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                expected[name.lstrip("*").strip()] = digest
        bad = [(n, d) for d, _, n in records if expected.get(n) != d]
        if bad:
            print(f"\n  ❌ 摘要不匹配: {[n for n, _ in bad]}")
            return False
        print(f"\n  ✅ 与 archives/SHA256SUMS 中记录的 {len(expected)} 条摘要一致")
        return True

    SUMS.write_text(
        "".join(f"{d}  {n}\n" for d, _, n in records), encoding="utf-8"
    )
    print(f"\n  ✅ 首次运行，已写入 {SUMS.relative_to(C.ROOT)}")
    return True


def verify_demo() -> bool:
    C.header("2. demo 交付指标独立复算")
    ok = True
    for task in "ABC":
        run = C.DEMO / "harness-runs" / f"{task}-v1"
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        rows = C.read_split(task, "validation")
        x, y = C.matrix(rows), C.labels(rows, task)
        model = C.vendor()[0].Model.load(run / "selected" / "checkpoint.npz")
        recomputed = C.evaluate(task, model, x, y)

        recorded = summary["selected"]["metrics"]
        key = C.TASKS[task]["primary"]
        same = abs(recomputed[key] - recorded[key]) < 1e-9
        ok = ok and same
        print(
            f"  {task}  selected {key}: 记录 {recorded[key]:.10f} | "
            f"复算 {recomputed[key]:.10f}  {'✅' if same else '❌'}"
        )

        base = summary["baseline"]["metrics"]
        baseline_model = C.load_model(task)
        recomputed_base = C.evaluate(task, baseline_model, x, y)
        same_base = abs(recomputed_base[key] - base[key]) < 1e-9
        ok = ok and same_base
        print(
            f"     baseline {key}: 记录 {base[key]:.10f} | "
            f"复算 {recomputed_base[key]:.10f}  {'✅' if same_base else '❌'}"
        )
    return ok


def main() -> int:
    archives_ok = verify_archives()
    demo_ok = verify_demo()
    print()
    if archives_ok and demo_ok:
        print("✅ 全部校验通过")
        return 0
    print("❌ 存在校验失败项")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
