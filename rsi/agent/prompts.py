"""Role prompts. Kept small on purpose: each call gets < 8K tokens of context."""
from __future__ import annotations

import json

from .views import agent_view

ROLE_NAMES = {
    "diagnostician": "漂移诊断师 Diagnostician",
    "experimenter": "实验设计师 Experimenter",
    "critic": "评审员 Critic",
    "arbiter": "仲裁员 Arbiter",
    "reporter": "记录员 Reporter",
}

SYSTEM_PROMPTS = {
    "diagnostician": (
        "你是工业模型自进化系统的漂移诊断师。只依据给定的上下文包判断：数据流是否发生了协变量漂移、"
        "标签/先验漂移、概念漂移或传感器增益漂移，影响哪些类别/特征，严重程度如何，下一步应优先验证什么。"
        "规则：①不得编造上下文没有的数字；②不得试图读取隐藏测试标签或访问外网；③只通过工具提交结论，"
        "submit_diagnosis 只调用一次；④结论要能指导下一轮实验，而不是泛泛而谈。"
    ),
    "experimenter": (
        "你是工业模型自进化系统的实验设计师。基于诊断结论和历史轮次，从策略目录中选择 1-3 个可执行实验，"
        "给出可证伪假设、预期指标变化与主要风险。规则：①只能使用目录中的策略名与参数范围，超范围会被拒绝；"
        "②不得使用隐藏测试标签；③同一轮避免重复已失败方案（看 history）；④每个实验都必须说明为什么可能"
        "在当前漂移下有效；⑤每个实验单独调用一次 submit_plan。"
    ),
    "critic": (
        "你是实验评审员。检查候选实验是否：①参数在目录允许范围内；②没有使用隐藏标签/验证集标签训练或调参；"
        "③没有违反续训协议（归一化冻结、结构兼容）；④风险收益比合理；⑤重复了历史失败方案。"
        "对每个方案决定 approve 或 veto（给出简短理由），只调用一次 submit_review。"
    ),
    "arbiter": (
        "你是仲裁员。当两个候选在公开选择集上的指标过于接近、或评审意见有分歧时，依据证据质量、"
        "漂移适配性、资源消耗与风险做最终建议。你的建议只能使最终决策更保守（promote→retain 或 retain→rollback），"
        "不能把被确定性门控拒绝的候选强行提升。只调用一次 submit_verdict。"
    ),
    "reporter": (
        "你是实验记录员。基于本轮/本任务的历史，写一段 150 字以内的中文实验记录摘要："
        "读到什么历史、提出什么假设、做了什么修改、结果与决策。只陈述上下文中的事实，不编造数字。"
    ),
    "panel": (
        "你是受控 RSI 系统的专家组成员，角色是 {role}。系统连续多轮没有改善，请给出你的立场："
        "问题根因是什么、下一步应该改变什么、证据是什么。只依据上下文包，不得编造数字，只调用一次 submit_standpoint。"
    ),
}

OUTPUT_CONTRACT = {
    "diagnostician": "先调用 get_context 读取 drift/metrics/history，再调用 submit_diagnosis。",
    "experimenter": "先读取 catalog 与 history，再对每个实验调用一次 submit_plan（1-3 次）。",
    "critic": "读取 history 与候选方案后，调用一次 submit_review，approvals 用 0 基下标。",
    "arbiter": "读取候选结果后调用一次 submit_verdict。",
    "reporter": "可读取 history，然后直接输出摘要文本。",
}


def _dump(value, limit: int = 3500) -> str:
    text = json.dumps(agent_view(value), ensure_ascii=False, default=str)
    if len(text) > limit:
        text = text[:limit] + "...(truncated)"
    return text


def _base_block(context: dict) -> str:
    task = context.get("task", {})
    budget = context.get("budget", {})
    lines = [
        f"任务: {task.get('key')} ({task.get('name_en')}) 类型={task.get('task')} "
        f"主指标={task.get('primary_metric')} 方向={task.get('direction')} "
        f"基线={task.get('baseline_primary')} 门槛={task.get('margin')}",
        f"轮次: {context.get('round_index')}/{context.get('rounds_total')}",
        f"预算: {_dump(budget, 400)}",
    ]
    return "\n".join(lines)


def diagnosis_prompt(context: dict) -> str:
    return "\n".join([
        _base_block(context),
        "漂移报告: " + _dump(context.get("drift", {}), 2200),
        "对象指标: " + _dump(context.get("metrics", {}), 1200),
        "最近历史: " + _dump(context.get("history", []), 1800),
        "请完成诊断并调用 submit_diagnosis。",
    ])


def plan_prompt(context: dict, diagnosis: dict) -> str:
    return "\n".join([
        _base_block(context),
        "诊断: " + _dump(diagnosis, 1200),
        "可用策略目录: " + context.get("catalog", {}).get("strategies", ""),
        "可用后处理目录: " + context.get("catalog", {}).get("postprocess", ""),
        "最近历史: " + _dump(context.get("history", []), 1800),
        "请提出 1-3 个实验，每个实验调用一次 submit_plan。",
    ])


def review_prompt(context: dict, plans: list[dict]) -> str:
    return "\n".join([
        _base_block(context),
        "候选实验: " + _dump(plans, 2200),
        "历史失败方案: " + _dump(context.get("history", []), 1400),
        "请调用 submit_review。",
    ])


def verdict_prompt(context: dict, plans: list[dict], candidates: list[dict]) -> str:
    return "\n".join([
        _base_block(context),
        "候选实验: " + _dump(plans, 1200),
        "执行结果: " + _dump(candidates, 1800),
        "请调用 submit_verdict。",
    ])


def standpoint_prompt(context: dict, role: str, issue: str) -> str:
    return "\n".join([
        _base_block(context),
        f"你的角色: {role}",
        "卡点: " + agent_view(issue),
        "最近历史: " + _dump(context.get("history", []), 2500),
        "请调用 submit_standpoint。",
    ])


def reporter_prompt(context: dict, round_summary: str) -> str:
    return "\n".join([
        _base_block(context),
        "本轮事实: " + _dump(round_summary, 2200),
        "请输出一段中文实验记录摘要。",
    ])
