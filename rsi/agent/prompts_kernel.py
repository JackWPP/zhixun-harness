"""Kernel prompts: the autonomous agent's system prompt and per-round brief.

The system prompt is *frozen* for a whole task (Hermes-style prefix-cache
protection): goal, rules and tool usage live there, while each round only sends
a compact briefing as a normal user message.
"""
from __future__ import annotations

import json

from .views import agent_view

KERNEL_SYSTEM = """你是"工业模型自进化系统"的自主实验 Agent（受控 RSI 内核）。你不是脚本执行器，
而是一个有目标、有工具、有记忆的闭环控制器：观察漂移 → 提出可证伪假设 → 选择并执行实验 →
评估公开验证指标 → 依据证据做 promote/retain/rollback 决策 → 把经验写进记忆。

【不可违反的边界（否则工具会直接拒绝）】
1. 只能使用官方训练数据 train 拟合。selection 标签只供公开验证与候选选择；
   report/probe/隐藏评测结果不向你开放，不可据此调整策略或写入记忆。
2. 只能使用策略目录中的策略和参数范围；每轮最多 3 个实验，每任务最多 40 个实验。
3. 决策必须依据工具返回的公开 selection 指标与确定性门控；你不能强行提升一个未过门槛的候选。
   分类任务除官方 macro-F1 外，还要检查每类 selection 支持数；确定性门控会要求支持数至少为
   max(10, selection 行数的 5%) 的类别宏平均不下降。不要把只有少数样本类别的偶然变化当成稳健改进，
   也不要通过反复微调同一参数追逐小样本类别。
4. 不要尝试执行 shell、读取文件、访问网络或修改代码；你没有这些工具。
5. 不要编造数字。所有结论必须引用工具返回的指标、sha 或历史记录。

【工作循环（每一轮）】
① get_run_state 看当前状态；必要时 get_history/get_lessons/search_records 回忆。
② propose_experiment 提出 1-3 个实验（每个实验一个假设、一个策略、一个后处理方案）。
③ run_experiment(plan_id) 执行；可对比多个候选（compare_candidates）。
④ 根据 selection 指标与门槛，调用 decide 选定 promote/retain/rollback 并写理由。
⑤ 如出现新经验，用 memory_add / skill_manage 写入；最后调用 finish_round 提交本轮摘要。
卡住时可 delegate 给诊断/实验/评审角色，或 convene_panel 召专家组（每任务最多 2 次）。

【决策标准】promote 只在候选的公开 selection 主指标越过门槛（分类提升≥margin，回归下降≥margin），
且分类任务支持充分类别宏平均不下降时选择；否则 retain（保留父模型）；候选显著变差时 rollback。
对小样本类别提升不稳健的轮次，下一轮要换机制或改进诊断，不能从 report/probe 寻找参数。
【输出语言】中文；工具参数中的 JSON 必须是合法 JSON。
"""


def _clip(value, limit: int = 2600) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def kernel_round_prompt(brief: dict) -> str:
    """Compact per-round briefing (target: < 4K tokens, no hidden data)."""
    brief = agent_view(brief)
    task = brief.get("task", {})
    goal = brief.get("goal", {})
    metrics = brief.get("metrics", {})
    budget = brief.get("budget", {})
    lines = [
        "=== 本轮任务简报 ===",
        (f"任务 {task.get('key')} · {task.get('name_en')} · 类型={task.get('task')} · "
         f"主指标={task.get('primary_metric')}({task.get('direction')}) · "
         f"官方基线={task.get('baseline_primary')} · 晋升门槛={task.get('margin')}"),
        f"目标：{goal.get('goal', '')}",
        (f"目标进度：round {brief.get('round_index')}/{brief.get('rounds_total')}，"
         f"turns={goal.get('turns_used')}/{goal.get('max_turns')}，status={goal.get('status')}"),
        f"当前父模型指标：{_clip(metrics, 1400)}",
        f"预算：{_clip(budget, 600)}",
        "记忆快照（frozen，写入后立即生效于下一轮）：\n" + str(brief.get("memory", ""))[:2000],
        "历史经验：\n" + str(brief.get("lessons", ""))[:1000],
        "策略胜率：\n" + str(brief.get("stats", ""))[:800],
        "最近历史轮次：" + _clip(brief.get("history", []), 2000),
        "漂移报告：" + _clip(brief.get("drift", {}), 1800),
        "可用策略目录：\n" + str(brief.get("catalog", ""))[:2200],
        "技能索引：\n" + str(brief.get("skills", ""))[:1200],
        "",
        "请开始本轮：先观察，再提出实验；本轮必须至少执行 1 个实验或给出明确 retain 理由，"
        "并调用 decide 与 finish_round。",
    ]
    return "\n".join(lines)
