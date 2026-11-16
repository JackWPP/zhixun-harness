"""Render the private-sim evidence tables (matrix + sensitivity) to markdown."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(".scratch/ps-results")
OUT = Path("docs/evidence/private_sim.md")


def _fmt(values):
    return " ".join(f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in values)


def main() -> int:
    matrix = json.loads((ROOT / "matrix.json").read_text()) if (ROOT / "matrix.json").exists() else {"rows": {}}
    sens = json.loads((ROOT / "sensitivity.json").read_text()) if (ROOT / "sensitivity.json").exists() else {"rows": {}}
    lines = ["# Private-Sim 证据（诚实批次协议）", ""]
    lines.append("协议：每批只把**已揭示历史标签**放进 `train`/`dev`；当前批输入无标签，预测后才用 `answers` 评分；")
    lines.append("hidden 批永不揭示、不进历史。`decision.json` / `predictions.jsonl` 不依赖当前批标签。")
    lines.append("")
    lines.append("## E1/E2 主对照（arms × seeds）")
    lines.append("")
    lines.append("| task | arm | seed | arrival 逐批 primary | hidden 逐批 primary | hidden 均值 | wall(s) |")
    lines.append("|---|---|---|---|---|---|---|")
    rows = matrix.get("rows", {})
    for tag in sorted(rows):
        row = rows[tag]
        hidden = row.get("hidden") or []
        hidden_values = [value for _, value in hidden]
        lines.append(f"| {row['task']} | {row['arm']} | {row['seed']} | {_fmt(row.get('arrival') or [])} | "
                     f"{_fmt(hidden_values)} | {_fmt([row.get('hidden_mean')])} | {row.get('elapsed')} |")
    lines.append("")
    lines.append("## E3–E6/E9 敏感性（A）")
    lines.append("")
    lines.append("| case | arm | arrival 逐批 | hidden 逐批 | hidden 均值 | wall(s) |")
    lines.append("|---|---|---|---|---|---|")
    srows = sens.get("rows", {})
    for tag in sorted(srows):
        row = srows[tag]
        hidden = row.get("hidden") or []
        lines.append(f"| {tag} | {row['arm']} | {_fmt(row.get('arrival') or [])} | "
                     f"{_fmt([value for _, value in hidden])} | {_fmt([row.get('hidden_mean')])} | {row.get('elapsed')} |")

    gated = json.loads((ROOT / "gated.json").read_text()) if (ROOT / "gated.json").exists() else {"rows": {}}
    lines.append("")
    lines.append("## E2b 门控版 adapt/full（dev 验证后回滚；诚实协议）")
    lines.append("")
    lines.append("| tag | arm | arrival 逐批 | hidden 逐批 | hidden 均值 | gate kept | wall(s) |")
    lines.append("|---|---|---|---|---|---|---|")
    grows = gated.get("rows", {})
    for tag in sorted(grows):
        row = grows[tag]
        hidden = row.get("hidden") or []
        lines.append(f"| {tag} | {row['arm']} | {_fmt(row.get('arrival') or [])} | "
                     f"{_fmt([value for _, value in hidden])} | {_fmt([row.get('hidden_mean')])} | "
                     f"{row.get('gate_kept')}/{row.get('gate_attempted')} | {row.get('elapsed')} |")
    e0_path = ROOT / "e0_invariance.json"
    if e0_path.exists():
        e0 = json.loads(e0_path.read_text())
        lines.append("")
        lines.append("## E0 协议不变性（打乱当前批 answers 后 decision/predictions 不变）")
        lines.append("")
        lines.append(f"- 结果：**{e0.get('verdict')}**；batch 数 {e0.get('batches')}；差异 {e0.get('differences')}")


    auto_root = Path(".scratch/ps-auto")
    if auto_root.exists():
        lines.append("")
        lines.append("## E1b 默认 auto 路由（按任务/来源选择 arm）")
        lines.append("")
        lines.append("| task/source | 解析 arm | 首末 arrival primary | hidden 均值 | errors |")
        lines.append("|---|---|---|---|---|")
        for report_path in sorted(auto_root.glob("*/stream_report.json")):
            report = json.loads(report_path.read_text())
            for key, data in report.get("tasks", {}).items():
                final = data.get("final") or {}
                hidden = final.get("hidden_mean_primary")
                arm = (data.get("batches") or [{}])[0].get("arm", "?")
                lines.append(f"| {key}/{report.get('source')} | {arm} | "
                             f"{final.get('first_primary')} -> {final.get('final_primary')} | "
                             f"{hidden} | {len(data.get('errors', []))} |")
    e7 = Path(".scratch/ps-e7/stream_report.json")
    if e7.exists():
        report = json.loads(e7.read_text())
        data = (report.get("tasks") or {}).get("D") or {}
        final = data.get("final") or {}
        lines.append("")
        lines.append("## E7 未知任务零代码（合成 task.json，4 批官方流）")
        lines.append("")
        lines.append(f"- batches={len(data.get('batches', []))} errors={len(data.get('errors', []))} "
                     f"first={final.get('first_primary')} final={final.get('final_primary')}")
    e8 = Path(".scratch/ps-e8/stream_report.json")
    if e8.exists():
        report = json.loads(e8.read_text())
        data = (report.get("tasks") or {}).get("A") or {}
        final = data.get("final") or {}
        lines.append("")
        lines.append("## E8 kernel LLM 批次协议（2 arrival + 1 hidden）")
        lines.append("")
        decisions = [b.get("decision") for b in data.get("batches", [])]
        lines.append(f"- arrival decisions={decisions} hidden={len(data.get('hidden', []))} "
                     f"errors={len(data.get('errors', []))} wall={final.get('wall_seconds')}s")

    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHONPATH=. ./.venv/bin/python scripts/research/run_private_sim_matrix.py")
    lines.append("PYTHONPATH=. ./.venv/bin/python scripts/research/run_private_sim_sensitivity.py")
    lines.append("PYTHONPATH=. ./.venv/bin/python scripts/research/run_private_sim_gated.py")
    lines.append("PYTHONPATH=. ./.venv/bin/python scripts/research/run_e0_invariance.py")
    lines.append("PYTHONPATH=. ./.venv/bin/python scripts/research/summarize_private_sim.py")
    lines.append("```")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
