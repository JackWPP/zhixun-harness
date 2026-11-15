"""运行结束后生成不依赖外部资源的中文 HTML 报告。"""

import html
import json
import math
from pathlib import Path


__all__ = ["write_report"]


def _escape(value):
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _items(value):
    return value if isinstance(value, (list, tuple)) else []


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    try:
        if not math.isfinite(value):
            return "—"
        return _escape(format(value, ".6g"))
    except (OverflowError, ValueError):
        return "—"


def _json(value):
    return _escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _digest(value):
    if not value:
        return "—"
    text = str(value)
    short = text if len(text) <= 20 else text[:12] + "…" + text[-8:]
    return '<code title="' + _escape(text) + '">' + _escape(short) + "</code>"


def _metrics(value):
    metrics = _mapping(value)
    if not metrics:
        return "—"
    labels = {"macro_f1": "Macro-F1", "accuracy": "Accuracy", "rmse": "RMSE", "mae": "MAE", "nasa_score": "NASA 分数（本地辅助）", "mean_signed_error": "平均有符号误差", "overestimate_fraction": "高估比例"}
    keys = [key for key in labels if key in metrics]
    return "<br>".join(labels[key] + ": " + _number(metrics[key]) for key in keys) or "—"


def _diagnosis(value):
    diagnosis = _mapping(value)
    summary = "".join("<li>" + _escape(item) + "</li>" for item in _items(diagnosis.get("summary")))
    findings = []
    for entry in _items(diagnosis.get("findings")):
        finding = _mapping(entry)
        findings.append(
            '<article class="finding"><h4>' + _escape(finding.get("title")) + "</h4>"
            + '<p class="muted">编号：' + _escape(finding.get("code"))
            + " · 严重程度：" + _escape(finding.get("severity")) + "</p>"
            + "<p>建议：" + _escape(finding.get("recommendation")) + "</p>"
            + "<details><summary>查看证据</summary><pre>" + _json(finding.get("evidence", {}))
            + "</pre></details></article>"
        )
    return (
        ("<ul>" + summary + "</ul>" if summary else '<p class="muted">未提供诊断摘要。</p>')
        + "".join(findings)
        + "<details><summary>诊断指标与详细信息</summary><pre>"
        + _json({key: diagnosis.get(key, {}) for key in ("train_metrics", "validation_metrics", "details")})
        + "</pre></details>"
    )


def _candidate(value):
    candidate = _mapping(value)
    strategy = _mapping(candidate.get("strategy"))
    cost = _mapping(candidate.get("cost"))
    gate = _mapping(candidate.get("gate"))
    eligible = gate.get("eligible")
    gate_label = "已接受" if gate.get("accepted") is True else "未接受" if gate.get("accepted") is False else "可参与比较" if eligible is True else "无效候选" if eligible is False else "未评估"
    status = {"completed": "已完成", "budget_exhausted": "预算耗尽", "error": "错误"}.get(
        candidate.get("status"), candidate.get("status", "未知")
    )
    error = candidate.get("error")
    return (
        "<tr><td><strong>" + _escape(candidate.get("id")) + "</strong><br>" + _escape(status)
        + ("<p class=\"error\">" + _escape(error) + "</p>" if error else "")
        + "</td><td>" + _escape(strategy.get("label", strategy.get("id")))
        + "<p>" + _escape(strategy.get("reason")) + "</p>"
        + "<details><summary>策略参数</summary><pre>" + _json(strategy) + "</pre></details>"
        + "</td><td>" + _metrics(candidate.get("metrics"))
        + "<p>最佳检查点：<code>" + _escape(candidate.get("best_checkpoint")) + "</code></p>"
        + _digest(candidate.get("checkpoint_sha256"))
        + "</td><td>训练 " + _number(cost.get("training_seconds")) + " 秒<br>总计 "
        + _number(cost.get("total_seconds")) + " 秒<br>轮次 " + _number(cost.get("epochs"))
        + " / 步数 " + _number(cost.get("steps"))
        + "</td><td>" + gate_label + "<p>" + _escape(gate.get("reason"))
        + "</p>改进量：" + _number(gate.get("improvement"))
        + "</td></tr>"
    )


def write_report(summary: dict, output_dir: Path) -> Path:
    """仅为已结束的运行写入 report.html，并返回该文件路径。"""
    if summary.get("status") not in {"completed", "budget_exhausted", "failed"} or not summary.get("finished_at"):
        raise ValueError("运行尚未结束，不能生成报告。")

    baseline = _mapping(summary.get("baseline"))
    selected = _mapping(summary.get("selected"))
    source = _mapping(summary.get("source"))
    cost = _mapping(summary.get("cost"))
    task = summary.get("task")
    key_metric = "macro_f1" if task == "A" else "rmse" if task in ("B", "C") else None
    direction = "越高越好" if key_metric == "macro_f1" else "越低越好" if key_metric == "rmse" else "未指定"
    metric_name = key_metric or "关键指标"
    status = {"completed": "已完成", "budget_exhausted": "预算耗尽", "failed": "失败"}[summary["status"]]
    rounds = []
    for entry in _items(summary.get("rounds")):
        round_data = _mapping(entry)
        decision = _mapping(round_data.get("decision"))
        candidates = "".join(_candidate(candidate) for candidate in _items(round_data.get("candidates")))
        rounds.append(
            '<section><h2>第 ' + _escape(round_data.get("round")) + " 轮</h2>"
            + "<p>父检查点摘要：" + _digest(round_data.get("parent_checkpoint_sha256")) + "</p>"
            + _diagnosis(round_data.get("diagnosis"))
            + '<div class="table-wrap"><table><thead><tr><th>候选 / 状态</th><th>策略</th>'
            + "<th>最佳验证指标 / 检查点</th><th>成本</th><th>准入判定</th></tr></thead><tbody>"
            + (candidates or '<tr><td colspan="5">无候选记录。</td></tr>') + "</tbody></table></div>"
            + '<div class="decision"><h3>本轮决策：'
            + ("接受候选" if decision.get("accepted") is True else "未接受候选")
            + "</h3><p>胜出候选：" + _escape(decision.get("winner_id"))
            + "</p><p>原因：" + _escape(decision.get("reason"))
            + "</p><p>变更前：" + _metrics(decision.get("metrics_before"))
            + "</p><p>变更后：" + _metrics(decision.get("metrics_after")) + "</p></div></section>"
        )
    warnings = "".join("<li>" + _escape(item) + "</li>" for item in _items(summary.get("warnings")))
    source_rows = "".join(
        "<tr><th>" + label + "</th><td>" + _digest(source.get(key)) + "</td></tr>"
        for key, label in (
            ("archive_sha256", "归档 SHA-256"), ("checkpoint_sha256", "初始检查点 SHA-256"),
            ("train_sha256", "训练集 SHA-256"), ("validation_sha256", "验证集 SHA-256"),
        )
    )
    document = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Harness 本地运行报告</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#243047;font:15px/1.65 system-ui,-apple-system,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:32px 22px}h1,h2,h3,h4{line-height:1.35;color:#16233a}h1{font-size:30px}h2{font-size:22px}h4{margin:0}section,header{background:white;border:1px solid #dfe5ee;border-radius:14px;padding:24px;margin-bottom:22px}.muted{color:#627089}.notice{background:#fff6df;border-left:4px solid #d49a22;padding:16px;border-radius:6px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:20px}.card{background:#f4f7fe;border:1px solid #e0e7f6;padding:18px;border-radius:10px}.value{display:block;font-size:30px;font-weight:700;color:#224d99}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:12px;border-bottom:1px solid #e3e8f0;vertical-align:top}th{background:#f5f7fb}td p{margin:6px 0}code,pre{font-family:ui-monospace,Consolas,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f5f7fb;padding:14px;border-radius:8px;font-size:13px}details{margin:12px 0}summary{cursor:pointer;color:#2855a0}a{color:#2158ad;text-underline-offset:3px}.links{display:flex;flex-wrap:wrap;gap:18px}.finding,.decision{border-left:3px solid #a8bce2;padding:14px 18px;margin:16px 0;background:#f8faff}.error{color:#a32929;overflow-wrap:anywhere}p,li,td{overflow-wrap:anywhere}@media(max-width:640px){main{padding:16px 10px}section,header{padding:18px 14px}.cards{grid-template-columns:1fr}h1{font-size:25px}th,td{min-width:140px;padding:10px}}
</style></head><body><main>'''
    document += (
        "<header><h1>Harness 本地运行报告</h1><p>任务 " + _escape(task)
        + " · " + _escape(status) + "</p><p>运行 ID：<code>" + _escape(summary.get("run_id"))
        + "</code></p><p class=\"muted\">开始：" + _escape(summary.get("started_at"))
        + "<br>结束：" + _escape(summary.get("finished_at")) + "</p>"
        + '<div class="notice"><strong>仅限开发验证，不代表官方或隐藏测试集成绩。</strong>'
        + "本报告不包含隐藏数据评估。验证集仅用于评估与候选选择，不参与优化器更新。"
        + "策略与诊断基于规则，不使用 LLM。</div>"
        + ('<p class="error">运行错误：' + _escape(summary.get("error")) + '</p>' if summary.get("error") else '')
        + "</header>"
        + "<section><h2>基线与最终选择</h2><p>关键指标：" + _escape(metric_name)
        + "（" + direction + "）；不推断或补造缺失指标及改进量。</p>"
        + '<div class="cards"><div class="card">基线<span class="value">'
        + _number(_mapping(baseline.get("metrics")).get(key_metric)) + "</span>"
        + _metrics(baseline.get("metrics")) + "</div><div class=\"card\">最终选择<span class=\"value\">"
        + _number(_mapping(selected.get("metrics")).get(key_metric)) + "</span>"
        + _metrics(selected.get("metrics")) + "</div></div>"
        + "<p>基线检查点：<code>" + _escape(baseline.get("checkpoint")) + "</code> "
        + _digest(baseline.get("checkpoint_sha256")) + "</p>"
        + "<p>最终检查点：<code>" + _escape(selected.get("checkpoint")) + "</code> "
        + _digest(selected.get("checkpoint_sha256")) + "</p>"
        + '<nav class="links"><a href="summary.json" download>下载运行摘要</a>'
        + '<a href="events.jsonl" download>下载事件日志</a>'
        + ('<a href="selected/checkpoint.npz" download>下载最终检查点</a>' if selected.get("checkpoint") else "")
        + "</nav></section><section><h2>基线诊断</h2>" + _diagnosis(baseline.get("diagnosis"))
        + "</section>" + ("".join(rounds) or "<section><h2>候选轮次</h2><p>无轮次记录。</p></section>")
        + "<section><h2>最终诊断</h2>" + _diagnosis(summary.get("diagnosis_final")) + "</section>"
        + "<section><h2>运行成本</h2><p>总耗时：" + _number(cost.get("total_seconds"))
        + " 秒 · 训练：" + _number(cost.get("training_seconds")) + " 秒 · 验证："
        + _number(cost.get("validation_seconds")) + " 秒 · 候选数：" + _number(cost.get("candidate_count"))
        + "</p><details><summary>运行配置</summary><pre>" + _json(summary.get("configuration", {}))
        + "</pre></details></section><section><h2>来源与可追溯性</h2><p>归档："
        + _escape(source.get("archive")) + "</p><p>官方评分接口就绪："
        + ("是（不表示已经执行官方评分）" if source.get("official_scoring_ready") is True else "否 / 未确认")
        + '</p><div class="table-wrap"><table>' + source_rows + "</table></div>"
        + '<p class="muted">悬停摘要可查看完整 SHA-256。</p></section>'
        + "<section><h2>警告</h2>" + ("<ul>" + warnings + "</ul>" if warnings else "<p>无额外警告。</p>")
        + "</section></main></body></html>"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
