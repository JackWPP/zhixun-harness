/* 智训引擎 · RSI 看板前端
 * 手写 vanilla JS，零依赖、零外部资源。
 * 数据来自同源 /api/state 与 /api/events（服务端每 2s 重读 state.json）。
 */
"use strict";

(() => {
  const POLL_MS = 2000;
  const EVENTS_LIMIT = 200;
  const EVENTS_RENDER_CAP = 120;

  const RUN_STATUS = {
    running: { label: "运行中", cls: "is-running" },
    finished: { label: "已完成", cls: "is-done" },
    failed: { label: "失败", cls: "is-failed" },
    stopped: { label: "已停止", cls: "is-stopped" },
  };
  const TASK_STATUS = {
    waiting: { label: "等待中", cls: "is-waiting" },
    running: { label: "运行中", cls: "is-running" },
    done: { label: "已完成", cls: "is-done" },
    failed: { label: "失败", cls: "is-failed" },
  };
  const DECISION = {
    promote: { label: "提升", en: "promote", cls: "is-promote" },
    retain: { label: "保留", en: "retain", cls: "is-retain" },
    rollback: { label: "回滚", en: "rollback", cls: "is-rollback" },
  };
  const DRIFT = {
    stable: { label: "稳定", level: 0 },
    mild: { label: "轻微", level: 1 },
    moderate: { label: "中度", level: 2 },
    severe: { label: "严重", level: 3 },
  };
  const LEVELS = {
    info: { label: "信息", cls: "is-info" },
    warn: { label: "警告", cls: "is-warn" },
    error: { label: "错误", cls: "is-error" },
  };
  const ROLES = {
    diagnostician: { label: "诊断者", cls: "is-diagnostician" },
    experimenter: { label: "实验者", cls: "is-experimenter" },
    critic: { label: "评审者", cls: "is-critic" },
    arbiter: { label: "仲裁者", cls: "is-arbiter" },
  };
  const TASK_UNIT = { A: "", B: "分钟", C: "循环" };

  const ui = {
    payload: null,
    events: [],
    eventsTotal: 0,
    expanded: new Set(["A"]),
    panels: new Set(),
    filters: { level: "all", task: "all" },
    drawer: null,
    drawerReturnFocus: null,
    fetchAt: 0,
    signature: "",
    pollOk: false,
    timer: null,
    lastError: "",
  };

  const $ = (id) => document.getElementById(id);
  const icon = { arrowUp: "▲", arrowDown: "▼", dash: "◆", chevron: "▾" };

  /* ------------------------------------------------------------------ */
  /* 基础工具                                                            */
  /* ------------------------------------------------------------------ */
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[ch]));
  }

  function num(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function groupFixed(n, digits) {
    const fixed = Number(n).toFixed(digits);
    const negative = fixed.startsWith("-");
    const [intPart, fracPart] = (negative ? fixed.slice(1) : fixed).split(".");
    const grouped = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    return `${negative ? "-" : ""}${grouped}${fracPart ? "." + fracPart : ""}`;
  }

  function fmtInt(value) {
    const n = num(value);
    return n === null ? "—" : groupFixed(Math.round(n), 0);
  }

  function fmtCompact(value) {
    const n = num(value);
    if (n === null) return "—";
    const abs = Math.abs(n);
    if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return String(Math.round(n));
  }

  function fmtDuration(seconds) {
    const n = num(seconds);
    if (n === null || n < 0) return "—";
    const total = Math.floor(n);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function parseIso(value) {
    if (typeof value !== "string" || !value) return null;
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  function fmtClock(value) {
    const d = parseIso(value);
    if (!d) return "--:--:--";
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
  }

  function fmtDateTime(value) {
    const d = parseIso(value);
    if (!d) return "—";
    const date = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    return `${date} ${fmtClock(value)}`;
  }

  function relTime(value) {
    const d = parseIso(value);
    if (!d) return "";
    const diff = Math.max(0, (Date.now() - d.getTime()) / 1000);
    if (diff < 10) return "刚刚";
    if (diff < 60) return `${Math.floor(diff)} 秒前`;
    if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
    return `${Math.floor(diff / 86400)} 天前`;
  }

  function fmtMetric(metric, value) {
    const n = num(value);
    if (n === null) return "—";
    if (metric === "macro_f1" || metric === "accuracy") return n.toFixed(4);
    if (metric === "rmse" || metric === "mae" || metric === "nasa_score") {
      if (Math.abs(n) >= 100) return groupFixed(n, 1);
      return groupFixed(n, 2);
    }
    if (Math.abs(n) >= 1000) return groupFixed(n, 1);
    if (Math.abs(n) >= 1) return groupFixed(n, 2);
    if (Math.abs(n) >= 0.001) return n.toFixed(4);
    return n.toExponential(2);
  }

  function fmtDelta(metric, delta) {
    const n = num(delta);
    if (n === null) return "—";
    const sign = n > 0 ? "+" : n < 0 ? "-" : "";
    return sign + fmtMetric(metric, Math.abs(n));
  }

  function isoSortKey(value) {
    const d = parseIso(value);
    return d ? d.getTime() : 0;
  }

  function metricOf(task, metrics) {
    const metric = task.primary_metric || (task.task === "classification" ? "macro_f1" : "rmse");
    return { metric, value: metrics ? num(metrics[metric]) : null };
  }

  function improvement(task) {
    const current = num(task.current);
    const baseline = num(task.baseline);
    if (current === null || baseline === null) return null;
    const delta = current - baseline;
    const pct = Math.abs(baseline) > 1e-12 ? delta / Math.abs(baseline) : null;
    const good = delta === 0 ? null : (task.direction === "higher" ? delta > 0 : delta < 0);
    return { current, baseline, delta, pct, good };
  }

  /* ------------------------------------------------------------------ */
  /* 渲染：顶部运行状态                                                    */
  /* ------------------------------------------------------------------ */
  function renderHeader() {
    const data = ui.payload;
    const run = (data && data.run) || {};
    const status = RUN_STATUS[run.status] || RUN_STATUS.stopped;
    const llm = run.llm || {};
    const model = [llm.backend, llm.model].filter(Boolean).join(" / ") || "—";
    const calls = num(llm.calls);
    const tokensIn = num(llm.tokens_in);
    const tokensOut = num(llm.tokens_out);
    const backendHtml = `
      <div class="meta-item">
        <span class="meta-label">LLM 后端</span>
        <span class="meta-value mono">${esc(model)}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">LLM 调用</span>
        <span class="meta-value mono">${calls === null ? "—" : fmtInt(calls)} 次 · ${fmtCompact(tokensIn)} / ${fmtCompact(tokensOut)} tokens</span>
      </div>`;

    $("run-meta").innerHTML = `
      <div class="run-status ${status.cls}">
        <span class="status-light" aria-hidden="true"></span>
        <span class="run-status-label">${status.label}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">RUN ID</span>
        <span class="meta-value mono">${esc(run.run_id || "—")}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">已运行</span>
        <span class="meta-value mono" id="run-elapsed">${fmtDuration(run.elapsed_seconds)}</span>
      </div>
      ${backendHtml}
      <span class="poll-pill ${ui.pollOk ? "is-ok" : "is-bad"}" id="poll-indicator">
        <i aria-hidden="true"></i>${ui.pollOk ? "实时 · 2s" : "重连中…"}
      </span>`;
  }

  function updatePollPill() {
    const pill = $("poll-indicator");
    if (!pill) return;
    pill.className = `poll-pill ${ui.pollOk ? "is-ok" : "is-bad"}`;
    pill.innerHTML = `<i aria-hidden="true"></i>${ui.pollOk ? "实时 · 2s" : "重连中…"}`;
  }

  function currentElapsedSeconds() {
    const run = (ui.payload && ui.payload.run) || {};
    const base = num(run.elapsed_seconds);
    if (base === null) return null;
    if (run.status !== "running") return base;
    const drift = ui.fetchAt ? (Date.now() - ui.fetchAt) / 1000 : 0;
    return base + Math.max(0, drift);
  }

  /* ------------------------------------------------------------------ */
  /* 渲染：预算                                                          */
  /* ------------------------------------------------------------------ */
  function budgetItem({ label, en, used, cap, fmtUsed, fmtCap, fmtRemain }) {
    const u = num(used);
    const c = num(cap);
    const pct = c !== null && c > 0 && u !== null ? Math.max(0, Math.min(100, (u / c) * 100)) : null;
    const cls = pct === null ? "is-unknown" : pct >= 95 ? "is-danger" : pct >= 80 ? "is-warn" : "is-ok";
    return `
      <div class="budget-item ${cls}">
        <div class="budget-top">
          <span class="budget-label">${esc(label)} <i>${esc(en)}</i></span>
          <span class="budget-pct mono">${pct === null ? "—" : pct.toFixed(1) + "%"}</span>
        </div>
        <div class="budget-bar" role="img" aria-label="${esc(label)} 已使用 ${pct === null ? "未知" : pct.toFixed(0) + "%"}">
          <span style="width:${pct === null ? 0 : pct.toFixed(2)}%"></span>
        </div>
        <div class="budget-bottom">
          <span class="budget-used mono">${fmtUsed}</span>
          <span class="budget-remain">${fmtRemain}</span>
        </div>
      </div>`;
  }

  function renderBudget() {
    const data = ui.payload;
    const panel = $("budget-panel");
    if (!data || !data.run || !data.run.budget) {
      panel.innerHTML = '<p class="placeholder">预算数据不可用</p>';
      return;
    }
    const b = data.run.budget;
    const tokensUsed = num(b.tokens_used) || 0;
    const tokensCap = num(b.tokens_cap) || 0;
    const cpu = num(b.cpu_seconds) || 0;
    const cpuCap = num(b.cpu_cap) || 0;
    const wall = num(b.wall_seconds) || 0;
    const wallCap = num(b.wall_cap) || 0;
    const expUsed = num(b.experiments_used) || 0;
    const expCap = num(b.experiments_cap) || 0;

    const items = [
      budgetItem({
        label: "Token 预算", en: "TOKENS", used: tokensUsed, cap: tokensCap,
        fmtUsed: `${fmtCompact(tokensUsed)} / ${fmtCompact(tokensCap)}`,
        fmtRemain: `剩余 ${fmtCompact(Math.max(0, tokensCap - tokensUsed))}`,
      }),
      budgetItem({
        label: "CPU 时间", en: "CPU", used: cpu, cap: cpuCap,
        fmtUsed: `${fmtDuration(cpu)} / ${fmtDuration(cpuCap)}`,
        fmtRemain: `剩余 ${fmtDuration(Math.max(0, cpuCap - cpu))}`,
      }),
      budgetItem({
        label: "墙钟时间", en: "WALL", used: wall, cap: wallCap,
        fmtUsed: `${fmtDuration(wall)} / ${fmtDuration(wallCap)}`,
        fmtRemain: `剩余 ${fmtDuration(Math.max(0, wallCap - wall))}`,
      }),
      budgetItem({
        label: "实验次数", en: "EXPERIMENTS", used: expUsed, cap: expCap,
        fmtUsed: `${fmtInt(expUsed)} / ${fmtInt(expCap)} 次`,
        fmtRemain: `剩余 ${fmtInt(Math.max(0, expCap - expUsed))} 次`,
      }),
    ];
    panel.innerHTML = items.join("");
  }

  /* ------------------------------------------------------------------ */
  /* 渲染：任务卡片                                                      */
  /* ------------------------------------------------------------------ */
  function latestEventsForTask(key, limit) {
    const events = (ui.payload && Array.isArray(ui.payload.events)) ? ui.payload.events : [];
    return events
      .filter((e) => String(e.task || "").toUpperCase() === key)
      .slice()
      .sort((a, b) => isoSortKey(b.ts) - isoSortKey(a.ts))
      .slice(0, limit);
  }

  function statusChip(status) {
    const s = TASK_STATUS[status] || TASK_STATUS.waiting;
    return `<span class="status-chip ${s.cls}"><i aria-hidden="true"></i>${s.label}</span>`;
  }

  function decisionBadge(decision, fallbackRunning) {
    const d = DECISION[decision];
    if (d) return `<span class="badge ${d.cls}" title="${esc(d.en)}">${d.label}</span>`;
    if (fallbackRunning) return '<span class="badge is-running"><i aria-hidden="true"></i>进行中</span>';
    return '<span class="badge is-none">—</span>';
  }

  function ringHtml(done, total) {
    const safeTotal = total > 0 ? total : 10;
    const ratio = Math.min(1, Math.max(0, done / safeTotal));
    const r = 22;
    const circumference = 2 * Math.PI * r;
    const dash = circumference * ratio;
    return `
      <div class="ring" role="img" aria-label="已完成 ${done} / ${safeTotal} 轮">
        <svg viewBox="0 0 56 56" aria-hidden="true" focusable="false">
          <circle class="ring-track" cx="28" cy="28" r="${r}"></circle>
          <circle class="ring-fill" cx="28" cy="28" r="${r}"
            stroke-dasharray="${dash.toFixed(2)} ${(circumference - dash).toFixed(2)}"></circle>
        </svg>
        <span class="ring-label"><b>${done}</b><span>/${safeTotal}</span></span>
      </div>`;
  }

  function driftHtml(drift) {
    if (!drift || typeof drift !== "object") {
      return '<div class="drift drift-empty">暂无漂移诊断数据</div>';
    }
    const ordered = ["stable", "mild", "moderate", "severe"];
    const sev = DRIFT[drift.severity] || DRIFT.stable;
    const score = num(drift.score);
    const ks = num(drift.ks_max);
    const shift = num(drift.feature_shift);
    const causes = Array.isArray(drift.causes) ? drift.causes.filter(Boolean) : [];
    const evidence = String(drift.evidence || "");
    const tooltip = [...causes, evidence].filter(Boolean).join("；");
    const segments = ordered.map((name, i) => {
      const active = i === sev.level;
      const cls = `drift-seg sev-${name}${i <= sev.level ? " is-on" : ""}${active ? " is-active" : ""}`;
      return `<span class="${cls}" title="${esc(DRIFT[name].label + (active ? "（当前）" : ""))}"></span>`;
    }).join("");
    const chips = causes.slice(0, 2)
      .map((cause) => `<span class="chip" title="${esc(cause)}">${esc(cause)}</span>`)
      .join("");
    const more = causes.length > 2
      ? `<span class="chip chip-more" title="${esc(causes.slice(2).join("；"))}">+${causes.length - 2}</span>`
      : "";
    return `
      <div class="drift" title="${esc(tooltip)}">
        <div class="drift-head">
          <span class="drift-title">漂移严重度 <i>DRIFT</i></span>
          <span class="drift-sev sev-${DRIFT[drift.severity] ? drift.severity : "stable"}">${sev.label}${score === null ? "" : " · " + score.toFixed(2)}</span>
        </div>
        <div class="drift-gauge" role="img" aria-label="漂移严重度：${sev.label}，score ${score === null ? "未知" : score.toFixed(2)}">${segments}</div>
        <div class="drift-scale"><span>稳定</span><span>轻微</span><span>中度</span><span>严重</span></div>
        <div class="drift-stats mono">
          <span>KS max <b>${ks === null ? "—" : ks.toFixed(3)}</b></span>
          <span>特征偏移 <b>${shift === null ? "—" : shift.toFixed(2)}</b></span>
        </div>
        <div class="drift-causes">${chips}${more || (causes.length ? "" : '<span class="chip chip-more">无显著漂移因素</span>')}</div>
      </div>`;
  }

  function sparklineHtml(task) {
    const key = String(task.key || "?");
    const metric = task.primary_metric || (task.task === "classification" ? "macro_f1" : "rmse");
    const rounds = Array.isArray(task.rounds) ? task.rounds.filter((r) => r.status === "done") : [];
    const pairs = rounds
      .map((r) => ({ index: r.index, value: r.metrics ? num(r.metrics[metric]) : null }))
      .filter((p) => p.value !== null);
    const baseline = num(task.baseline);
    const W = 320;
    const H = 76;
    const PAD_X = 8;
    const PAD_Y = 12;
    if (!pairs.length) {
      return '<div class="spark-wrap"><div class="spark-head"><span>指标轨迹 <i>' + esc(metric) + "</i></span></div><div class=\"spark-empty\">指标轨迹待生成</div></div>";
    }
    const series = baseline === null ? pairs.map((p) => p.value) : [baseline, ...pairs.map((p) => p.value)];
    let min = Math.min(...series);
    let max = Math.max(...series);
    if (Math.abs(max - min) < 1e-12) {
      const pad = Math.max(Math.abs(max) * 0.05, 0.001);
      min -= pad;
      max += pad;
    }
    const pad = (max - min) * 0.2;
    min -= pad;
    max += pad;
    const yOf = (v) => PAD_Y + (H - 2 * PAD_Y) * (1 - (v - min) / (max - min));
    const xOf = (i) => (pairs.length === 1 ? W / 2 : PAD_X + (i * (W - 2 * PAD_X)) / (pairs.length - 1));
    const points = pairs.map((p, i) => [xOf(i), yOf(p.value)]);
    const line = points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const area = points.length > 1
      ? `${line} L${points[points.length - 1][0].toFixed(1)},${(H - PAD_Y).toFixed(1)} L${points[0][0].toFixed(1)},${(H - PAD_Y).toFixed(1)} Z`
      : "";
    const baseY = baseline === null ? null : yOf(baseline);
    const dots = points.map(([x, y], i) => {
      const last = i === points.length - 1;
      return `<circle class="spark-dot${last ? " is-last" : ""}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${last ? 3.4 : 2.2}"><title>第 ${esc(pairs[i].index)} 轮 · ${esc(fmtMetric(metric, pairs[i].value))}</title></circle>`;
    }).join("");
    return `
      <div class="spark-wrap">
        <div class="spark-head">
          <span>指标轨迹 <i>${esc(metric)}</i></span>
          <span class="spark-legend"><span class="legend-line" aria-hidden="true"></span>基线 ${esc(fmtMetric(metric, baseline))}</span>
        </div>
        <svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="指标轨迹，共 ${pairs.length} 个评估点">
          <defs>
            <linearGradient id="spark-grad-${esc(key)}" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" class="spark-stop-top"></stop>
              <stop offset="1" class="spark-stop-bottom"></stop>
            </linearGradient>
          </defs>
          ${baseY === null ? "" : `<line class="spark-baseline" x1="${PAD_X}" y1="${baseY.toFixed(1)}" x2="${W - PAD_X}" y2="${baseY.toFixed(1)}"></line>`}
          ${area ? `<path class="spark-area" d="${area}" fill="url(#spark-grad-${esc(key)})"></path>` : ""}
          ${points.length > 1 ? `<path class="spark-line" d="${line}"></path>` : ""}
          ${dots}
        </svg>
      </div>`;
  }

  function runningRoundOf(task) {
    return (Array.isArray(task.rounds) ? task.rounds : []).find((r) => r.status === "running") || null;
  }

  function currentRoundText(task) {
    const running = runningRoundOf(task);
    if (running) {
      const strategy = running.strategy ? ` · ${running.strategy}` : "";
      return `第 ${running.index} 轮 · 运行中${strategy}`;
    }
    const status = task.status;
    const index = num(task.round_index);
    const total = num(task.rounds_total) || 10;
    if (status === "done") {
      return `已收敛 · 共 ${num(index) || (Array.isArray(task.rounds) ? task.rounds.length : 0)} 轮`;
    }
    if (status === "failed") return "运行失败 · 等待人工检查";
    if (status === "waiting") return "等待调度";
    const next = Math.min((index || 0) + 1, total);
    return `第 ${next} 轮 · 诊断/规划中`;
  }

  function timelineHtml(task) {
    const rounds = Array.isArray(task.rounds) ? task.rounds : [];
    const metric = task.primary_metric || (task.task === "classification" ? "macro_f1" : "rmse");
    if (!rounds.length) return '<p class="empty">尚无轮次记录</p>';
    let accepted = num(task.baseline);
    const direction = task.direction === "higher" ? 1 : -1;
    const rows = rounds.map((round) => {
      const key = String(task.key);
      const roundId = String(round.id || round.index);
      const status = String(round.status || "done");
      const decision = DECISION[round.decision] ? DECISION[round.decision] : null;
      const metrics = round.metrics && typeof round.metrics === "object" ? round.metrics : {};
      const value = num(metrics[metric]);
      const delta = value !== null && accepted !== null ? value - accepted : null;
      const good = delta === null ? null : (delta === 0 ? null : delta * direction > 0);
      const sha = String(round.candidate_checkpoint_sha256 || "");
      const metricTitle = Object.keys(metrics).map((k) => `${k}=${fmtMetric(k, metrics[k])}`).join(" · ") || "指标待生成";
      const rowClass = decision ? decision.cls : "is-running";
      const deltaHtml = delta === null
        ? ""
        : `<span class="round-delta ${good === null ? "is-flat" : good ? "is-good" : "is-bad"}">${fmtDelta(metric, delta)}</span>`;
      const metricHtml = value === null
        ? '<span class="muted">—</span>'
        : `<b class="mono">${esc(fmtMetric(metric, value))}</b>${deltaHtml}`;
      const row = `
        <button type="button" class="round-row ${rowClass}" data-action="open-round"
          data-task="${esc(key)}" data-round="${esc(roundId)}"
          title="第 ${esc(round.index)} 轮 · ${esc(decision ? decision.label : "进行中")} · 点击查看证据详情">
          <span class="round-index mono">R${esc(round.index)}</span>
          ${decisionBadge(round.decision, status === "running")}
          <span class="round-body">
            <span class="round-hypo">${esc(round.hypothesis || "（未记录假设）")}</span>
            <span class="round-change">${esc(round.change_summary || "")}</span>
          </span>
          <span class="round-metric" title="${esc(metricTitle)}">${metricHtml}</span>
          <span class="round-sha mono" title="${esc(sha || "候选检查点待生成")}">${sha ? esc(sha.slice(0, 12)) + "…" : "待生成"}</span>
        </button>`;
      if (status === "done" && decision && decision.en === "promote" && value !== null) {
        accepted = value;
      }
      return row;
    }).join("");
    return `<div class="timeline-list">${rows}</div>
      <p class="timeline-hint">指标 Δ 为相对当前已接受检查点的变化；checkpoint 显示 sha256 前 12 位，悬停查看详情。</p>`;
  }

  function panelsHtml(task) {
    const panels = Array.isArray(task.panel) ? task.panel : [];
    if (!panels.length) return '<p class="empty">暂无 panel 讨论记录</p>';
    return panels.map((panel) => {
      const messages = Array.isArray(panel.messages) ? panel.messages : [];
      return `
        <div class="panel-block">
          <div class="panel-block-head">
            <span class="panel-trigger"><span class="mono">R${esc(panel.round_index)}</span> ${esc(panel.trigger || "")}</span>
            ${decisionBadge(panel.verdict, false)}
          </div>
          <div class="messages">
            ${messages.map((msg) => {
              const role = ROLES[msg.role] || ROLES.diagnostician;
              const confidence = num(msg.confidence);
              const pct = confidence === null ? null : Math.max(0, Math.min(1, confidence)) * 100;
              return `
                <div class="message ${role.cls}">
                  <div class="msg-head">
                    <span class="role-chip">${role.label}</span>
                    <span class="conf mono">置信度 ${pct === null ? "—" : pct.toFixed(0) + "%"}</span>
                  </div>
                  <div class="conf-bar" role="img" aria-label="置信度 ${pct === null ? "未知" : pct.toFixed(0) + "%"}">
                    <i style="width:${pct === null ? 0 : pct.toFixed(1)}%"></i>
                  </div>
                  <p class="standpoint">${esc(msg.standpoint || "")}</p>
                  <p class="evidence"><span>证据</span>${esc(msg.evidence || "")}</p>
                </div>`;
            }).join("")}
          </div>
        </div>`;
    }).join("");
  }

  function taskCardHtml(task) {
    const key = String(task.key || "?");
    const taskCls = /^[ABC]$/.test(key) ? `task-${key}` : "task-x";
    const metric = task.primary_metric || (task.task === "classification" ? "macro_f1" : "rmse");
    const unit = TASK_UNIT[key] || "";
    const directionText = task.direction === "higher" ? "越高越好 ↑" : "越低越好 ↓";
    const imp = improvement(task);
    const doneCount = Array.isArray(task.rounds) ? task.rounds.filter((r) => r.status === "done").length : 0;
    const total = num(task.rounds_total) || (Array.isArray(task.rounds) ? task.rounds.length : 0) || 10;
    const budget = task.budget || {};
    const recent = latestEventsForTask(key, 3);
    const running = runningRoundOf(task);
    const isExpanded = ui.expanded.has(key);
    const isPanelOpen = ui.panels.has(key);
    const roundCount = Array.isArray(task.rounds) ? task.rounds.length : 0;
    const panelCount = Array.isArray(task.panel) ? task.panel.length : 0;

    let deltaHtml = '<span class="delta-chip is-flat">基线数据缺失</span>';
    if (imp) {
      const cls = imp.delta === 0 ? "is-flat" : imp.good ? "is-good" : "is-bad";
      const arrow = imp.delta > 0 ? icon.arrowUp : imp.delta < 0 ? icon.arrowDown : icon.dash;
      const pct = imp.pct === null ? "" : ` (${imp.pct > 0 ? "+" : ""}${(imp.pct * 100).toFixed(2)}%)`;
      deltaHtml = `<span class="delta-chip ${cls}" title="相对基线 ${esc(fmtMetric(metric, imp.baseline))}">${arrow} ${fmtDelta(metric, imp.delta)}${pct}</span>`;
    }

    const taskEvents = recent.length
      ? recent.map((event) => {
        const level = LEVELS[event.level] || LEVELS.info;
        return `<div class="task-event">
          <span class="task-event-time mono">${fmtClock(event.ts)}</span>
          <span class="task-event-dot ${level.cls}" title="${level.label}"></span>
          <span class="task-event-msg" title="${esc(event.message)}">${esc(event.message)}</span>
        </div>`;
      }).join("")
      : '<div class="empty mini">暂无该任务事件</div>';

    return `
      <article class="task-card ${taskCls}" data-task="${esc(key)}">
        <div class="card-accent" aria-hidden="true"></div>
        <header class="card-head">
          <span class="task-badge mono" title="任务 ${esc(key)}">${esc(key)}</span>
          <div class="task-title">
            <h3>${esc(task.name || task.name_en || key)}</h3>
            <p>${esc(task.name_en || "")} · ${esc(task.task || "")} · ${esc(metric)} ${esc(directionText)}</p>
          </div>
          ${statusChip(task.status)}
        </header>

        <div class="metric-row">
          <div class="metric-main">
            <div class="metric-value mono ${imp && imp.delta !== 0 ? (imp.good ? "is-good" : "is-bad") : ""}">${esc(fmtMetric(metric, task.current))}</div>
            <div class="metric-label">${esc(metric)}${unit ? `<span> · ${esc(unit)}</span>` : ""}</div>
          </div>
          <div class="metric-side">
            ${deltaHtml}
            <div class="metric-ref">基线 <b class="mono">${esc(fmtMetric(metric, task.baseline))}</b></div>
            <div class="metric-ref">最优 <b class="mono">${esc(fmtMetric(metric, task.best))}</b></div>
          </div>
          ${ringHtml(doneCount, total)}
        </div>

        ${sparklineHtml(task)}
        ${driftHtml(task.drift)}

        <div class="round-now ${running ? "is-running" : ""}">
          <span class="round-now-dot" aria-hidden="true"></span>${esc(currentRoundText(task))}
        </div>

        <div class="card-stats">
          <span><i>实验</i><b class="mono">${fmtInt(budget.experiments)}</b> 次</span>
          <span><i>Tokens</i><b class="mono">${fmtCompact(budget.tokens_used)}</b></span>
          <span><i>CPU</i><b class="mono">${fmtDuration(budget.cpu_seconds)}</b></span>
        </div>

        <div class="task-events">
          <div class="task-events-head"><span>最近事件 <i>LAST 3</i></span><span class="mono">${doneCount}/${total} 轮</span></div>
          ${taskEvents}
        </div>

        <div class="card-actions">
          <button type="button" class="action-btn ${isExpanded ? "is-open" : ""}" data-action="toggle-timeline"
            data-task="${esc(key)}" aria-expanded="${isExpanded ? "true" : "false"}">
            <span>轮次时间线 <b class="mono">${roundCount}</b></span><span class="chev" aria-hidden="true">${icon.chevron}</span>
          </button>
          <button type="button" class="action-btn ${isPanelOpen ? "is-open" : ""}" data-action="toggle-panel"
            data-task="${esc(key)}" aria-expanded="${isPanelOpen ? "true" : "false"}">
            <span>Panel 讨论 <b class="mono">${panelCount}</b></span><span class="chev" aria-hidden="true">${icon.chevron}</span>
          </button>
        </div>

        ${isExpanded ? `<div class="timeline">${timelineHtml(task)}</div>` : ""}
        ${isPanelOpen ? `<div class="discussion">${panelsHtml(task)}</div>` : ""}
      </article>`;
  }

  function renderTasks() {
    const grid = $("task-grid");
    const data = ui.payload;
    if (!data || !Array.isArray(data.tasks) || !data.tasks.length) {
      grid.innerHTML = '<p class="placeholder">暂无任务数据</p>';
      return;
    }
    grid.innerHTML = data.tasks.map(taskCardHtml).join("");
  }

  /* ------------------------------------------------------------------ */
  /* 渲染：事件流                                                        */
  /* ------------------------------------------------------------------ */
  function filteredEvents() {
    const { level, task } = ui.filters;
    return ui.events.filter((event) => {
      if (level !== "all" && event.level !== level) return false;
      if (task !== "all" && String(event.task || "").toUpperCase() !== task) return false;
      return true;
    });
  }

  function syncFilters() {
    document.querySelectorAll("[data-filter-level]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.filterLevel === ui.filters.level);
    });
    document.querySelectorAll("[data-filter-task]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.filterTask === ui.filters.task);
    });
  }

  function renderEvents() {
    const list = $("event-list");
    const count = $("event-count");
    const events = filteredEvents();
    const meta = (ui.payload && ui.payload._meta) || {};
    const recordsNote = meta.records_available ? ` · records.jsonl 已合并 ${fmtInt(meta.records_events)} 条` : "";
    count.textContent = `共 ${fmtInt(ui.eventsTotal)} 条 · 当前筛选 ${fmtInt(events.length)} 条${recordsNote}`;
    if (!events.length) {
      list.innerHTML = '<p class="empty">当前筛选条件下没有事件</p>';
      return;
    }
    const shown = events.slice(0, EVENTS_RENDER_CAP);
    list.innerHTML = shown.map((event) => {
      const level = LEVELS[event.level] || LEVELS.info;
      const task = String(event.task || "");
      const source = event.source === "records.jsonl" ? '<span class="ev-source">jsonl</span>' : "";
      return `<div class="event-row ${level.cls}">
        <span class="ev-time mono">${fmtClock(event.ts)}</span>
        <span class="ev-level">${level.label}</span>
        <span class="ev-task mono">${task ? esc(task) : "全局"}</span>
        <span class="ev-msg" title="${esc(event.message)}">${esc(event.message)}${source}</span>
      </div>`;
    }).join("");
    if (events.length > shown.length) {
      list.insertAdjacentHTML("beforeend", `<p class="event-more">仅显示前 ${shown.length} 条（共 ${events.length} 条），可用级别 / 任务筛选缩小范围。</p>`);
    }
  }

  /* ------------------------------------------------------------------ */
  /* 渲染：决策摘要                                                      */
  /* ------------------------------------------------------------------ */
  function renderSummary() {
    const box = $("summary");
    const data = ui.payload;
    if (!data || !Array.isArray(data.tasks)) {
      box.innerHTML = '<p class="empty">等待数据…</p>';
      return;
    }
    const counts = { promote: 0, retain: 0, rollback: 0 };
    let decided = 0;
    data.tasks.forEach((task) => {
      (Array.isArray(task.rounds) ? task.rounds : []).forEach((round) => {
        if (round.status === "done" && DECISION[round.decision]) {
          counts[round.decision] += 1;
          decided += 1;
        }
      });
    });
    const total = decided || 1;
    const stack = ["promote", "retain", "rollback"].map((key) => {
      const pct = (counts[key] / total) * 100;
      return pct > 0
        ? `<span class="stack-seg ${DECISION[key].cls}" style="width:${pct.toFixed(2)}%" title="${DECISION[key].label} ${counts[key]} 次"></span>`
        : "";
    }).join("");
    const legend = ["promote", "retain", "rollback"].map((key) => `
      <div class="legend-row">
        <span class="legend-key"><i class="legend-dot ${DECISION[key].cls}"></i>${DECISION[key].label} <em>${DECISION[key].en}</em></span>
        <span class="legend-count mono">${counts[key]} 次</span>
      </div>`).join("");
    const taskProgress = data.tasks.map((task) => {
      const key = String(task.key || "?");
      const done = (Array.isArray(task.rounds) ? task.rounds : []).filter((r) => r.status === "done").length;
      const totalRounds = num(task.rounds_total) || 10;
      const metric = task.primary_metric || "rmse";
      const status = TASK_STATUS[task.status] || TASK_STATUS.waiting;
      return `
        <div class="progress-row">
          <span class="progress-key mono">${esc(key)}</span>
          <span class="progress-main">
            <span class="progress-title">${esc(task.name || key)}</span>
            <span class="progress-bar"><i style="width:${Math.min(100, (done / totalRounds) * 100).toFixed(1)}%"></i></span>
          </span>
          <span class="progress-value mono">${esc(fmtMetric(metric, task.current))}</span>
          <span class="status-chip mini ${status.cls}">${status.label}</span>
        </div>`;
    }).join("");
    const meta = data._meta || {};
    const stale = num(meta.stale_seconds);
    const metaRows = [
      ["数据源", esc(meta.state_path || "—")],
      ["事件源", meta.records_available ? "state.json + records.jsonl" : "state.json"],
      ["schema", `v${esc(meta.schema_version == null ? "—" : meta.schema_version)}`],
      ["数据延迟", stale === null ? "—" : `${stale.toFixed(0)}s`],
    ].map(([label, value]) => `<div class="meta-row"><span>${label}</span><b class="mono" title="${value}">${value}</b></div>`).join("");

    box.innerHTML = `
      <div class="summary-block">
        <div class="summary-block-head"><span>门控决策分布</span><span class="mono">${decided} 轮</span></div>
        <div class="stack-bar">${stack}</div>
        <div class="legend">${legend}</div>
      </div>
      <div class="summary-block">
        <div class="summary-block-head"><span>任务进度</span><span class="mono">${data.tasks.length} 个</span></div>
        <div class="progress-list">${taskProgress}</div>
      </div>
      <div class="summary-block">
        <div class="summary-block-head"><span>数据链路</span><span class="mono">LINK</span></div>
        <div class="meta-table">${metaRows}</div>
      </div>`;
  }

  /* ------------------------------------------------------------------ */
  /* 渲染：抽屉 / 页脚                                                   */
  /* ------------------------------------------------------------------ */
  function findDrawerRound() {
    if (!ui.drawer || !ui.payload || !Array.isArray(ui.payload.tasks)) return null;
    const task = ui.payload.tasks.find((t) => String(t.key) === String(ui.drawer.task));
    if (!task) return null;
    const round = (Array.isArray(task.rounds) ? task.rounds : []).find(
      (r) => String(r.id || r.index) === String(ui.drawer.roundId)
    );
    return round ? { task, round } : null;
  }

  function metricTable(metrics) {
    const keys = Object.keys(metrics || {});
    if (!keys.length) return '<p class="empty mini">该轮指标尚未产出（运行中或已失败）</p>';
    return `<div class="metric-table">${keys.map((key) => `
      <div class="metric-cell"><span>${esc(key)}</span><b class="mono">${esc(fmtMetric(key, metrics[key]))}</b></div>`).join("")}</div>`;
  }

  function renderDrawer() {
    const drawer = $("drawer");
    const backdrop = $("drawer-backdrop");
    const found = findDrawerRound();
    if (!found) {
      drawer.classList.add("hidden");
      backdrop.classList.add("hidden");
      document.body.classList.remove("drawer-open");
      if (ui.drawer) ui.drawer = null;
      return;
    }
    const { task, round } = found;
    const metric = task.primary_metric || "rmse";
    const metrics = round.metrics && typeof round.metrics === "object" ? round.metrics : {};
    const decision = DECISION[round.decision] || null;
    const panel = (Array.isArray(task.panel) ? task.panel : []).find((p) => Number(p.round_index) === Number(round.index));
    const panelVerdict = panel ? DECISION[panel.verdict] : null;
    const shaParent = String(round.parent_checkpoint_sha256 || "");
    const shaCandidate = String(round.candidate_checkpoint_sha256 || "");
    const statusLabel = round.status === "running" ? "进行中" : round.status === "failed" ? "失败" : "已完成";

    drawer.innerHTML = `
      <div class="drawer-head">
        <div>
          <p class="drawer-kicker">TASK ${esc(task.key)} · 轮次证据</p>
          <h3>R${esc(round.index)} · ${decision ? decision.label : statusLabel}</h3>
          <p class="drawer-sub">${esc(task.name || "")} · ${esc(metric)}</p>
        </div>
        <button type="button" class="icon-btn" data-action="close-drawer" aria-label="关闭详情抽屉">×</button>
      </div>
      <div class="drawer-body">
        <div class="drawer-chips">
          ${decisionBadge(round.decision, round.status === "running")}
          <span class="chip">策略 ${esc(round.strategy || "—")}</span>
          <span class="chip">后处理 ${esc(round.postprocess || "none")}</span>
          <span class="chip">${esc(statusLabel)}</span>
        </div>
        <section class="drawer-section">
          <h4>假设 <i>HYPOTHESIS</i></h4>
          <p>${esc(round.hypothesis || "（未记录）")}</p>
        </section>
        <section class="drawer-section">
          <h4>变更摘要 <i>CHANGE</i></h4>
          <p>${esc(round.change_summary || "（未记录）")}</p>
        </section>
        <section class="drawer-section">
          <h4>本轮指标 <i>METRICS</i></h4>
          ${metricTable(metrics)}
        </section>
        ${panel ? `
        <section class="drawer-section">
          <h4>本轮 Panel 结论 <i>PANEL VERDICT</i></h4>
          <p class="drawer-panel-line">
            ${panelVerdict ? `<span class="badge ${panelVerdict.cls}">${panelVerdict.label}</span>` : ""}
            <span>${esc(panel.trigger || "")}</span>
          </p>
        </section>` : ""}
        <section class="drawer-section">
          <h4>检查点溯源 <i>CHECKPOINT LINEAGE</i></h4>
          <div class="lineage">
            <div class="lineage-item">
              <span class="lineage-label">父检查点</span>
              <code class="sha-block">${shaParent ? esc(shaParent) : "—"}</code>
            </div>
            <div class="lineage-item">
              <span class="lineage-label">候选检查点</span>
              <code class="sha-block">${shaCandidate ? esc(shaCandidate) : "待生成"}</code>
            </div>
          </div>
        </section>
        <section class="drawer-section">
          <h4>时间 <i>TIMESTAMP</i></h4>
          <p class="mono">${esc(fmtDateTime(round.created_at))} <span class="muted">（${esc(relTime(round.created_at)) || "—"}）</span></p>
        </section>
      </div>`;
    drawer.classList.remove("hidden");
    backdrop.classList.remove("hidden");
    document.body.classList.add("drawer-open");
  }

  function closeDrawer() {
    ui.drawer = null;
    $("drawer").classList.add("hidden");
    $("drawer-backdrop").classList.add("hidden");
    document.body.classList.remove("drawer-open");
    if (ui.drawerReturnFocus && typeof ui.drawerReturnFocus.focus === "function") {
      ui.drawerReturnFocus.focus();
    }
    ui.drawerReturnFocus = null;
  }

  function renderFooter() {
    const data = ui.payload;
    const meta = (data && data._meta) || {};
    const run = (data && data.run) || {};
    $("footer-state-path").textContent = meta.state_path || "—";
    $("footer-state-path").title = meta.state_path || "";
    const generated = fmtDateTime(meta.generated_at);
    $("footer-generated").textContent = generated;
    $("footer-generated").title = generated;
    $("footer-schema").textContent = meta.schema_version == null ? "—" : `v${meta.schema_version}`;
    $("footer-version").textContent = `v${meta.dashboard_version || "1.0.0"}`;
    const stale = currentStaleSeconds();
    const footerStale = $("footer-stale");
    footerStale.textContent = stale === null ? "数据延迟 —" : `数据延迟 ${stale < 60 ? stale.toFixed(0) + "s" : (stale / 60).toFixed(1) + "min"}`;
    footerStale.classList.toggle("is-warn", stale !== null && stale > 8);
    footerStale.classList.toggle("is-bad", stale !== null && stale > 30);
    if (run.status && !$("run-meta").querySelector(".run-status")) renderHeader();
  }

  function currentStaleSeconds() {
    const meta = (ui.payload && ui.payload._meta) || {};
    const base = num(meta.stale_seconds);
    if (base === null) return null;
    const drift = ui.fetchAt ? (Date.now() - ui.fetchAt) / 1000 : 0;
    return base + Math.max(0, drift);
  }

  /* ------------------------------------------------------------------ */
  /* 渲染总控                                                            */
  /* ------------------------------------------------------------------ */
  function renderAll() {
    renderHeader();
    renderBudget();
    renderTasks();
    renderEvents();
    renderSummary();
    renderFooter();
    renderDrawer();
    syncFilters();
  }

  function renderTick() {
    const elapsed = currentElapsedSeconds();
    const el = $("run-elapsed");
    if (el && elapsed !== null) el.textContent = fmtDuration(elapsed);
    const stale = currentStaleSeconds();
    const footerStale = $("footer-stale");
    if (footerStale && stale !== null) {
      footerStale.textContent = `数据延迟 ${stale < 60 ? stale.toFixed(0) + "s" : (stale / 60).toFixed(1) + "min"}`;
      footerStale.classList.toggle("is-warn", stale > 8);
      footerStale.classList.toggle("is-bad", stale > 30);
    }
  }

  /* ------------------------------------------------------------------ */
  /* 数据获取                                                            */
  /* ------------------------------------------------------------------ */
  function signatureOf(data) {
    if (!data) return "";
    const meta = data._meta || {};
    return JSON.stringify({
      schema: data.schema_version,
      run: data.run,
      tasks: data.tasks,
      events: data.events,
      meta: { records_events: meta.records_events, error: meta.error || "" },
    });
  }

  async function fetchJSON(url, timeoutMs = 5000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const text = await response.text();
      let body = null;
      try {
        body = JSON.parse(text);
      } catch (error) {
        body = { ok: false, error: "响应不是合法 JSON" };
      }
      return { status: response.status, body };
    } finally {
      clearTimeout(timer);
    }
  }

  function showBanner(message, kind) {
    const banner = $("connection-banner");
    banner.textContent = message;
    banner.className = `connection-banner is-${kind || "error"}`;
  }

  function hideBanner() {
    const banner = $("connection-banner");
    banner.classList.add("hidden");
    banner.textContent = "";
  }

  async function poll() {
    if (ui.timer) {
      clearTimeout(ui.timer);
      ui.timer = null;
    }
    if (document.hidden) {
      scheduleNext();
      return;
    }
    try {
      const [stateResult, eventsResult] = await Promise.allSettled([
        fetchJSON("/api/state"),
        fetchJSON(`/api/events?limit=${EVENTS_LIMIT}`),
      ]);
      if (stateResult.status === "rejected") {
        throw new Error(stateResult.reason && stateResult.reason.name === "AbortError" ? "请求超时" : String(stateResult.reason));
      }
      const state = stateResult.value;
      if (state.status !== 200 || !state.body || state.body.ok === false) {
        const detail = (state.body && (state.body.error || state.body.detail)) || `HTTP ${state.status}`;
        const hint = state.body && state.body.hint ? `。${state.body.hint}` : "";
        throw new Error(`${detail}${hint}`);
      }
      const data = state.body;
      ui.payload = data;
      ui.fetchAt = Date.now();
      ui.pollOk = true;
      ui.lastError = "";
      if (eventsResult.status === "fulfilled" && eventsResult.value.status === 200 && eventsResult.value.body && Array.isArray(eventsResult.value.body.events)) {
        ui.events = eventsResult.value.body.events;
        ui.eventsTotal = num(eventsResult.value.body.total) === null ? ui.events.length : eventsResult.value.body.total;
      } else {
        ui.events = Array.isArray(data.events) ? data.events.slice() : [];
        ui.eventsTotal = ui.events.length;
      }
      if (data._meta && data._meta.error) {
        showBanner(`数据源降级：${data._meta.error}`, "warn");
      } else {
        hideBanner();
      }
      const signature = signatureOf(data) + `|${ui.events.length}|${ui.events[0] ? ui.events[0].ts : ""}`;
      if (signature !== ui.signature) {
        ui.signature = signature;
        renderAll();
      } else {
        updatePollPill();
      }
    } catch (error) {
      ui.pollOk = false;
      ui.lastError = error && error.message ? error.message : String(error);
      if (!ui.payload) renderHeader();
      updatePollPill();
      const base = ui.payload ? "上次数据仍保留，页面继续每 2s 重试。" : "请确认看板服务仍在运行。";
      showBanner(`数据源连接失败：${ui.lastError}。${base}`, "error");
    } finally {
      scheduleNext();
    }
  }

  function scheduleNext() {
    if (ui.timer) clearTimeout(ui.timer);
    ui.timer = setTimeout(poll, POLL_MS);
  }

  /* ------------------------------------------------------------------ */
  /* 交互                                                                */
  /* ------------------------------------------------------------------ */
  function handleAction(action, element) {
    const taskKey = element.dataset.task;
    switch (action) {
      case "toggle-timeline": {
        if (ui.expanded.has(taskKey)) ui.expanded.delete(taskKey);
        else ui.expanded.add(taskKey);
        renderTasks();
        break;
      }
      case "toggle-panel": {
        if (ui.panels.has(taskKey)) ui.panels.delete(taskKey);
        else ui.panels.add(taskKey);
        renderTasks();
        break;
      }
      case "open-round": {
        ui.drawerReturnFocus = element;
        ui.drawer = { task: element.dataset.task, roundId: element.dataset.round };
        renderDrawer();
        const closeBtn = document.querySelector('[data-action="close-drawer"]');
        if (closeBtn) closeBtn.focus();
        break;
      }
      case "close-drawer":
        closeDrawer();
        break;
      default:
        break;
    }
  }

  document.addEventListener("click", (event) => {
    const actionEl = event.target.closest("[data-action]");
    if (actionEl) {
      handleAction(actionEl.dataset.action, actionEl);
      return;
    }
    const levelBtn = event.target.closest("[data-filter-level]");
    if (levelBtn) {
      ui.filters.level = levelBtn.dataset.filterLevel;
      syncFilters();
      renderEvents();
      return;
    }
    const taskBtn = event.target.closest("[data-filter-task]");
    if (taskBtn) {
      ui.filters.task = taskBtn.dataset.filterTask;
      syncFilters();
      renderEvents();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && ui.drawer) closeDrawer();
  });

  $("drawer-backdrop").addEventListener("click", closeDrawer);

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      if (ui.timer) clearTimeout(ui.timer);
      poll();
    }
  });

  setInterval(renderTick, 1000);
  poll();
})();
