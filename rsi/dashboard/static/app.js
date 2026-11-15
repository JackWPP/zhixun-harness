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
  const CONTROLLER = {
    kernel: { label: "自主内核", en: "KERNEL", cls: "is-kernel" },
    pipeline: { label: "流水线", en: "PIPELINE", cls: "is-pipeline" },
  };
  const RECORD_TYPE = { event: "事件", round: "轮次", decision: "决策", panel: "Panel", llm_call: "LLM" };
  const GOAL_STATUS = { running: "进行中", done: "已完成", failed: "失败", blocked: "受阻", pending: "待开始" };
  const SOFT_VERDICT = {
    met: { label: "已达标", cls: "is-met" },
    missed: { label: "未达标", cls: "is-missed" },
    unknown: { label: "未知", cls: "is-unknown" },
  };
  const DRIFT_TYPE_LABEL = {
    covariate_shift: "协变量漂移",
    load_regime_shift: "工况漂移",
    label_shift: "标签漂移",
    concept_shift: "概念漂移",
    none: "无显著漂移",
  };
  const PROPOSAL_SOURCES = {
    seed: { label: "种子", en: "SEED", cls: "is-seed" },
    agent: { label: "Agent 提案", en: "AGENT", cls: "is-agent" },
    knowledge: { label: "知识命中", en: "KNOWLEDGE", cls: "is-knowledge" },
    panel: { label: "Panel 讨论", en: "PANEL", cls: "is-panel" },
  };

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
    memory: null,
    agent: null,
    goal: null,
    knowledge: null,
    search: { query: "", items: null, total: 0, busy: false, error: "", done: false },
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
          <div class="card-head-right">
            ${controllerBadge(task.controller || "pipeline")}
            ${statusChip(task.status)}
          </div>
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
  /* 渲染：软目标 / 批次 / 漂移雷达 / 知识 / 提案来源（schema v3）        */
  /* ------------------------------------------------------------------ */
  function fmtMetricValue(metric, value) {
    const n = num(value);
    return n === null ? "—" : fmtMetric(metric, n);
  }

  function softGoalAvailable(soft) {
    if (!soft || typeof soft !== "object") return false;
    if (soft.available === false) return false;
    return Boolean(soft.targets && Object.keys(soft.targets).length) || num(soft.gap) !== null;
  }

  function goalSoftHtml(task) {
    const soft = task.goal_soft && typeof task.goal_soft === "object" ? task.goal_soft : null;
    const verdict = (soft && SOFT_VERDICT[soft.verdict]) || SOFT_VERDICT.unknown;
    const metric = (soft && soft.metric) || task.primary_metric || "";
    const direction = (soft && soft.direction) || task.direction || "higher";
    const degraded = Boolean(soft && soft.degraded);
    const available = softGoalAvailable(soft);
    const targets = available ? soft.targets || {} : {};
    const currents = available ? soft.current || {} : {};
    const metricKeys = Object.keys(targets).length ? Object.keys(targets) : Object.keys(currents);
    const targetValue = metricKeys.length ? num(targets[metricKeys[0]]) : null;
    const currentValue = metricKeys.length ? num(currents[metricKeys[0]]) : null;
    const gap = available ? num(soft.gap) : null;
    let coverage = 0;
    if (targetValue && currentValue) {
      coverage = direction === "lower" ? targetValue / currentValue : currentValue / targetValue;
      coverage = Math.max(0, Math.min(1, coverage));
    }
    const goalLabel = task.goal && task.goal.goal ? task.goal.goal : "";
    if (!available) {
      return `<article class="goal-card is-empty">
        <div class="goal-card-head">
          <span class="task-badge">${esc(task.key)}</span>
          <h3>${esc(task.name || "任务 " + task.key)}</h3>
          <span class="goal-verdict is-unknown">未提供</span>
        </div>
        <p class="goal-empty">该任务未提供 goal_soft（软目标降级：仅展示当前指标与主目标）。</p>
        ${goalLabel ? `<p class="goal-evidence">主目标：${esc(goalLabel)}</p>` : ""}
      </article>`;
    }
    return `<article class="goal-card ${degraded ? "is-degraded" : ""}">
        <div class="goal-card-head">
          <span class="task-badge">${esc(task.key)}</span>
          <h3>${esc(task.name || "任务 " + task.key)}</h3>
          <span class="goal-verdict ${verdict.cls}">${verdict.label}</span>
        </div>
        <div class="goal-metrics">
          <div class="goal-metric"><span>软目标</span><b class="mono">${fmtMetricValue(metric, targetValue)}</b></div>
          <div class="goal-metric"><span>当前</span><b class="mono">${fmtMetricValue(metric, currentValue)}</b></div>
          <div class="goal-metric"><span>缺口</span><b class="mono ${gap !== null && gap > 0 ? "is-bad" : "is-good"}">${fmtMetricValue(metric, gap)}</b></div>
        </div>
        <div class="goal-bar" role="img" aria-label="软目标覆盖 ${Math.round(coverage * 100)}%">
          <i style="width:${(coverage * 100).toFixed(1)}%"></i>
        </div>
        <div class="goal-chips">
          <span class="metric-tag mono">${esc(metric || "metric")}</span>
          <span class="goal-flag ${degraded ? "is-warn" : "is-ok"}">${degraded ? "degraded · 已退化" : "正常"}</span>
          <span class="goal-note">${esc((soft && soft.note) || "软目标，不阻塞完成")}</span>
        </div>
        ${soft.evidence ? `<p class="goal-evidence">${esc(soft.evidence)}</p>` : ""}
      </article>`;
  }

  function batchTimelineHtml(batches) {
    if (!Array.isArray(batches) || !batches.length) {
      return '<p class="empty mini">尚无批次记录（state.batches 缺失或为空，自动降级）。</p>';
    }
    return `<ol class="batch-timeline">${batches.map((raw) => {
      const batch = raw && typeof raw === "object" ? raw : {};
      const before = batch.before_metrics && typeof batch.before_metrics === "object" ? batch.before_metrics : {};
      const after = batch.after_metrics && typeof batch.after_metrics === "object" ? batch.after_metrics : {};
      const metric = Object.keys(after)[0] || Object.keys(before)[0] || "";
      const beforeValue = metric ? num(before[metric]) : null;
      const afterValue = metric ? num(after[metric]) : null;
      let deltaHtml = '<span class="batch-delta">—</span>';
      if (beforeValue !== null && afterValue !== null) {
        const delta = afterValue - beforeValue;
        const arrow = delta > 0 ? "▲" : delta < 0 ? "▼" : "◆";
        deltaHtml = `<span class="batch-delta mono">${arrow} ${esc(fmtMetric(metric, Math.abs(delta)))}</span>`;
      }
      return `<li class="batch-item">
          <div class="batch-index mono">B${esc(batch.index === undefined ? "?" : batch.index)}</div>
          <div class="batch-main">
            <span class="batch-task">任务 ${esc(batch.task || "—")}</span>
            <span class="batch-metric mono">${esc(metric || "metric")} <b>${esc(beforeValue === null ? "—" : fmtMetric(metric, beforeValue))}</b> → <b>${esc(afterValue === null ? "—" : fmtMetric(metric, afterValue))}</b></span>
            ${deltaHtml}
          </div>
          ${decisionBadge(batch.decision)}
          ${batch.note ? `<p class="batch-note">${esc(batch.note)}</p>` : ""}
        </li>`;
    }).join("")}</ol>`;
  }

  function clamp01(value) {
    const n = num(value);
    return n === null ? 0 : Math.max(0, Math.min(1, n));
  }

  function metricPercentText(value) {
    return `${Math.round(clamp01(value) * 100)}%`;
  }

  function radarHtml(drift) {
    const d = drift && typeof drift === "object" ? drift : {};
    const severity = DRIFT[String(d.severity || "").toLowerCase()];
    const axes = [
      { label: "严重度", value: clamp01(d.score) },
      { label: "特征位移", value: clamp01(d.feature_shift) },
      { label: "KS max", value: clamp01(d.ks_max) },
      { label: "等级", value: severity ? severity.level / 3 : 0 },
    ];
    const cx = 100;
    const cy = 84;
    const radius = 52;
    const pointAt = (index, scale) => {
      const angle = -Math.PI / 2 + (index * 2 * Math.PI) / axes.length;
      return [cx + Math.cos(angle) * radius * scale, cy + Math.sin(angle) * radius * scale];
    };
    const grid = [0.25, 0.5, 0.75, 1].map((scale) => {
      const points = axes.map((_, index) => pointAt(index, scale).map((v) => v.toFixed(1)).join(",")).join(" ");
      return `<polygon points="${points}" class="radar-grid${scale === 1 ? " is-outer" : ""}" />`;
    }).join("");
    const spokes = axes.map((_, index) => {
      const [x, y] = pointAt(index, 1);
      return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="radar-spoke" />`;
    }).join("");
    const areaPoints = axes.map((axis, index) => pointAt(index, axis.value).map((v) => v.toFixed(1)).join(",")).join(" ");
    const labels = axes.map((axis, index) => {
      const [x, y] = pointAt(index, 1.38);
      const anchor = Math.abs(x - cx) < 8 ? "middle" : x > cx ? "start" : "end";
      return `<text x="${x.toFixed(1)}" y="${(y + 3).toFixed(1)}" text-anchor="${anchor}" class="radar-label">${esc(axis.label)}</text>`;
    }).join("");
    const aria = `漂移雷达：严重度 ${metricPercentText(d.score)}，特征位移 ${metricPercentText(d.feature_shift)}，KS max ${metricPercentText(d.ks_max)}`;
    return `<svg viewBox="0 0 200 172" class="radar-svg" role="img" aria-label="${esc(aria)}">${grid}${spokes}<polygon points="${areaPoints}" class="radar-area" />${axes.map((axis, index) => {
      const [x, y] = pointAt(index, axis.value);
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.4" class="radar-point" />`;
    }).join("")}${labels}</svg>`;
  }

  function driftRadarHtml(goalData) {
    const tasks = goalData && Array.isArray(goalData.tasks) ? goalData.tasks : [];
    if (!tasks.length) {
      return '<p class="empty mini">暂无漂移数据（/api/goal 不可用时自动降级）。</p>';
    }
    return tasks.map((task) => {
      const drift = task.drift && typeof task.drift === "object" ? task.drift : {};
      const sevKey = DRIFT[String(drift.severity || "").toLowerCase()] ? String(drift.severity).toLowerCase() : "stable";
      const severity = DRIFT[sevKey];
      const typeLabel = DRIFT_TYPE_LABEL[drift.type] || drift.type || "未分类";
      const attribution = Array.isArray(drift.attribution) ? drift.attribution.slice(0, 4) : [];
      const causes = attribution.length ? attribution : Array.isArray(drift.causes) ? drift.causes.slice(0, 4) : [];
      return `<article class="radar-card sev-${esc(sevKey)}">
          <div class="radar-card-head">
            <span class="task-badge">${esc(task.key)}</span>
            <h4>${esc(task.name || "任务 " + task.key)}</h4>
            <span class="drift-sev sev-${esc(sevKey)}">${severity.label}</span>
          </div>
          <div class="radar-body">
            ${radarHtml(drift)}
            <div class="radar-meta">
              <p><span>类型</span><b>${esc(typeLabel)}</b></p>
              <p><span>归因</span>${causes.length ? causes.map((item) => `<i class="attrib-chip">${esc(item)}</i>`).join("") : '<b class="muted">—</b>'}</p>
              ${drift.evidence ? `<p class="radar-evidence">${esc(drift.evidence)}</p>` : ""}
            </div>
          </div>
        </article>`;
    }).join("");
  }

  function knowledgeCardHtml() {
    const data = ui.knowledge;
    const knowledge = data && data.knowledge && typeof data.knowledge === "object" ? data.knowledge : null;
    const available = Boolean(knowledge) && knowledge.available !== false;
    if (!available) {
      return '<p class="empty mini">知识接口不可用（/api/knowledge 未返回或为空），已降级显示。</p>';
    }
    const docs = Array.isArray(knowledge.docs) ? knowledge.docs : [];
    const skills = Array.isArray(knowledge.skills_used) && knowledge.skills_used.length
      ? knowledge.skills_used
      : (Array.isArray(data.skills_used) ? data.skills_used : []);
    const lessons = Array.isArray(data.lessons) ? data.lessons : [];
    const maxUses = skills.reduce((max, skill) => Math.max(max, num(skill.uses) || 0), 0);
    return `<div class="knowledge-stats">
        <div class="knowledge-stat"><span>知识命中</span><b class="mono">${fmtInt(knowledge.hits)}</b></div>
        <div class="knowledge-stat"><span>文档</span><b class="mono">${fmtInt(docs.length)}</b></div>
        <div class="knowledge-stat"><span>lessons</span><b class="mono">${fmtInt(lessons.length)}</b></div>
        <div class="knowledge-stat"><span>技能</span><b class="mono">${fmtInt(skills.length)}</b></div>
      </div>
      ${docs.length ? `<ul class="doc-list">${docs.slice(0, 5).map((raw) => {
        const doc = raw && typeof raw === "object" ? raw : {};
        return `<li><span class="doc-title">${esc(doc.title || doc.name || doc.path || "文档")}</span>
            <span class="doc-path mono">${esc(doc.path || "")}</span><b class="mono">${fmtInt(doc.hits)}</b></li>`;
      }).join("")}</ul>` : '<p class="empty mini">暂无知识文档命中。</p>'}
      ${skills.length ? `<div class="skill-bars">${skills.slice(0, 6).map((raw) => {
        const skill = raw && typeof raw === "object" ? raw : {};
        const uses = num(skill.uses) || 0;
        const width = maxUses > 0 ? (uses / maxUses) * 100 : 0;
        return `<div class="skill-row"><span class="skill-name">${esc(skill.skill || "技能")}</span>
            <span class="skill-bar"><i style="width:${width.toFixed(1)}%"></i></span>
            <b class="mono">${fmtInt(uses)}</b></div>`;
      }).join("")}</div>` : ""}`;
  }

  function aggregateProposalSources(agentData) {
    const totals = { seed: 0, agent: 0, knowledge: 0, panel: 0 };
    let rounds = 0;
    const tasks = agentData && Array.isArray(agentData.tasks) ? agentData.tasks : [];
    for (const task of tasks) {
      const taskRounds = Array.isArray(task.rounds) ? task.rounds : [];
      for (const round of taskRounds) {
        const source = round && round.agent && round.agent.proposal_sources;
        if (!source || typeof source !== "object" || !Object.keys(source).length) continue;
        rounds += 1;
        for (const key of Object.keys(totals)) totals[key] += num(source[key]) || 0;
      }
    }
    return { totals, rounds };
  }

  function proposalBarHtml(sources) {
    const safe = sources && typeof sources === "object" ? sources : {};
    const order = ["seed", "agent", "knowledge", "panel"];
    const total = order.reduce((sum, key) => sum + (num(safe[key]) || 0), 0);
    if (!total) {
      return '<p class="empty mini">暂无提案来源数据（round.agent.proposal_sources 缺失时自动降级）。</p>';
    }
    return `<div class="proposal-bar" role="img" aria-label="提案来源占比">
        ${order.map((key) => {
          const source = PROPOSAL_SOURCES[key];
          const value = num(safe[key]) || 0;
          const pct = (value / total) * 100;
          return `<span class="proposal-seg ${source.cls}" style="width:${pct.toFixed(2)}%" title="${source.label} ${value}（${pct.toFixed(0)}%）"></span>`;
        }).join("")}
      </div>
      <div class="proposal-legend">
        ${order.map((key) => {
          const source = PROPOSAL_SOURCES[key];
          const value = num(safe[key]) || 0;
          const pct = (value / total) * 100;
          return `<span class="proposal-item"><i class="${source.cls}" aria-hidden="true"></i>${source.label} <b class="mono">${value}</b><em>${pct.toFixed(0)}%</em></span>`;
        }).join("")}
      </div>`;
  }

  function renderGoal() {
    const panel = $("goal-panel");
    if (!panel) return;
    const data = ui.goal;
    if (!data || !Array.isArray(data.tasks) || !data.tasks.length) {
      panel.innerHTML = '<p class="empty">软目标数据不可用（/api/goal 未返回或为空）；主看板其余功能不受影响。</p>';
      return;
    }
    const summary = data.summary && typeof data.summary === "object" ? data.summary : {};
    const softCount = num(summary.goal_soft) !== null
      ? num(summary.goal_soft)
      : data.tasks.filter((task) => softGoalAvailable(task.goal_soft)).length;
    const degradedCount = num(summary.degraded) !== null
      ? num(summary.degraded)
      : data.tasks.filter((task) => task.goal_soft && task.goal_soft.degraded).length;
    const batches = Array.isArray(data.batches) ? data.batches : [];
    const batchIndex = num(data.batch_index) !== null ? num(data.batch_index) : batches.length;
    panel.innerHTML = `<div class="goal-summary">
        <span class="goal-summary-item"><b class="mono">${softCount}</b> 项软目标</span>
        <span class="goal-summary-item is-warn"><b class="mono">${degradedCount}</b> 项 degraded</span>
        <span class="goal-summary-item"><b class="mono">${batchIndex}</b> 个批次</span>
        <span class="goal-note-pill">软目标，不阻塞完成</span>
      </div>
      ${data.note ? `<p class="goal-banner">${esc(data.note)}</p>` : ""}
      <div class="goal-grid">${data.tasks.map(goalSoftHtml).join("")}</div>
      <div class="batch-block">
        <div class="batch-head"><h3>批次时间线 <i>BATCH TIMELINE</i></h3><span class="panel-sub mono">BATCH INDEX ${esc(batchIndex)}</span></div>
        ${batchTimelineHtml(batches)}
      </div>`;
  }

  function renderInsight() {
    const panel = $("insight-panel");
    if (!panel) return;
    const proposals = aggregateProposalSources(ui.agent);
    const knowledgeReady = Boolean(ui.knowledge && ui.knowledge.knowledge);
    panel.innerHTML = `<div class="insight-grid">
        <section class="insight-card radar-panel">
          <div class="card-section-head"><h3>漂移雷达 <i>DRIFT RADAR</i></h3><span class="panel-sub mono">score · shift · KS</span></div>
          <div class="radar-list">${driftRadarHtml(ui.goal)}</div>
        </section>
        <section class="insight-card knowledge-panel">
          <div class="card-section-head"><h3>知识 / 技能 <i>KNOWLEDGE</i></h3><span class="panel-sub mono">${knowledgeReady ? "db + state" : "降级"}</span></div>
          ${knowledgeCardHtml()}
        </section>
        <section class="insight-card proposal-panel">
          <div class="card-section-head"><h3>提案来源 <i>PROPOSALS</i></h3><span class="panel-sub mono">${proposals.rounds} 轮有数据</span></div>
          ${proposalBarHtml(proposals.totals)}
          <p class="proposal-hint">来源统计自 round.agent.proposal_sources（seed / agent / knowledge / panel）。</p>
        </section>
      </div>`;
  }

  /* ------------------------------------------------------------------ */
  /* 渲染：记忆与自主内核                                                */
  /* ------------------------------------------------------------------ */
  function controllerBadge(controller) {
    const ctrl = CONTROLLER[String(controller || "pipeline").toLowerCase()] || CONTROLLER.pipeline;
    return `<span class="controller-chip ${ctrl.cls}" title="controller = ${esc(controller || "pipeline")}"><i aria-hidden="true"></i>${ctrl.en}</span>`;
  }

  function parsePercent(text) {
    const match = String(text || "").match(/(\d+(?:\.\d+)?)\s*%/);
    return match ? Math.max(0, Math.min(100, Number(match[1]))) : null;
  }

  function aggregateStrategies(rows) {
    const map = new Map();
    (Array.isArray(rows) ? rows : []).forEach((row) => {
      const key = `${row.strategy || "unknown"}|${row.postprocess_type || "none"}`;
      const item = map.get(key) || {
        strategy: row.strategy || "unknown",
        postprocess_type: row.postprocess_type || "none",
        trials: 0,
        promotions: 0,
        deltas: [],
      };
      const trials = num(row.trials);
      const promotions = num(row.promotions);
      item.trials += trials === null ? 0 : trials;
      item.promotions += promotions === null ? 0 : promotions;
      const mean = num(row.mean_delta);
      if (mean !== null) item.deltas.push(mean);
      map.set(key, item);
    });
    return [...map.values()]
      .map((item) => ({
        strategy: item.strategy,
        postprocess_type: item.postprocess_type,
        trials: item.trials,
        promotions: item.promotions,
        win_rate: item.trials ? item.promotions / item.trials : 0,
        mean_delta: item.deltas.length ? item.deltas.reduce((a, b) => a + b, 0) / item.deltas.length : null,
      }))
      .sort((a, b) => b.win_rate - a.win_rate || b.trials - a.trials)
      .slice(0, 8);
  }

  function findRoundAgent(taskKey, roundIndex) {
    const payload = ui.agent;
    if (!payload || !Array.isArray(payload.tasks)) return null;
    const task = payload.tasks.find((item) => String(item.key) === String(taskKey));
    if (!task) return null;
    return (Array.isArray(task.rounds) ? task.rounds : []).find(
      (row) => String(row.index) === String(roundIndex)
    ) || null;
  }

  function searchResultsHtml() {
    const search = ui.search;
    if (search.busy) return '<div class="search-state">正在检索 records / rounds…</div>';
    if (search.error) return `<div class="search-state is-bad">检索失败：${esc(search.error)}</div>`;
    if (!search.done) return "";
    const items = Array.isArray(search.items) ? search.items : [];
    const head = `
      <div class="search-results-head">
        <span>检索 <b class="mono">${esc(search.query || "（最近记录）")}</b> · 命中 <b class="mono">${fmtInt(search.total)}</b> 条，显示 ${items.length} 条</span>
        <button type="button" class="search-clear" data-action="search-clear">清除结果</button>
      </div>`;
    if (!items.length) return `<div class="search-results">${head}<div class="search-state">没有匹配的记录</div></div>`;
    const rows = items.map((item) => {
      const type = RECORD_TYPE[item.type] || item.type || "记录";
      const task = item.task ? `<span class="task-badge mini">${esc(item.task)}</span>` : "";
      const round = num(item.round) === null ? "" : `<span class="round-chip mono">R${esc(item.round)}</span>`;
      return `
        <div class="search-hit">
          <span class="search-hit-time mono">${esc(fmtDateTime(item.ts))}</span>
          <span class="record-type">${esc(type)}</span>
          ${task}${round}
          <span class="search-hit-msg" title="${esc(item.message)}">${esc(item.message)}</span>
          <span class="search-hit-src mono">${esc(item.source || "")}</span>
        </div>`;
    }).join("");
    return `<div class="search-results">${head}<div class="search-hit-list">${rows}</div></div>`;
  }

  function kernelPanelHtml(data) {
    const runAgent = data.run_agent || ((ui.payload && ui.payload.run && ui.payload.run.agent) || {});
    const controller = String(runAgent.controller || data.controller || "pipeline").toLowerCase();
    const memory = data.memory || {};
    const memTasks = Array.isArray(data.tasks) ? data.tasks : [];
    const lessons = Array.isArray(data.lessons) ? data.lessons : [];
    const strategies = aggregateStrategies(data.strategy_stats);
    const skills = Array.isArray(data.skill_usage) ? data.skill_usage : [];
    const files = Array.isArray(data.memory_files) ? data.memory_files : [];
    const usagePct = parsePercent(memory.usage);
    const failures = num(runAgent.kernel_failures);
    const counters = [
      ["内核轮次", runAgent.kernel_rounds],
      ["派发", runAgent.delegates],
      ["Panel", runAgent.panels],
      ["记忆写入", runAgent.memory_writes],
      ["技能写入", runAgent.skill_writes],
      ["Kernel tokens", fmtCompact(runAgent.kernel_tokens)],
    ];
    const db = data.db || {};
    const dbChip = db.available
      ? '<span class="db-chip is-ok">sqlite 已连接</span>'
      : `<span class="db-chip is-warn" title="${esc(db.error || "")}">sqlite 未连接 · 回退 state 快照</span>`;

    const fileRows = files.length
      ? files.map((file) => `
          <li class="memory-file">
            <span class="memory-file-name mono">${esc(file.name)}</span>
            <span class="memory-file-meta">${fmtInt(file.entries)} 条 · ${esc(file.usage || "")}</span>
          </li>`).join("")
      : '<li class="empty mini">memory/ 目录暂无 Markdown 快照</li>';

    const taskRows = memTasks.map((task) => {
      const goal = task.goal || {};
      const tmem = task.memory || {};
      const rounds = num(goal.rounds_done);
      const target = num(goal.rounds_target);
      return `
        <li class="memory-task">
          <span class="task-badge mini">${esc(task.key)}</span>
          <span class="memory-task-metrics">${fmtInt(tmem.entries)} 条 · ${fmtInt(tmem.snapshot_chars)} chars</span>
          <span class="memory-task-goal mono">${rounds === null ? "—" : `${esc(rounds)}/${esc(target === null ? "?" : target)}`}</span>
        </li>`;
    }).join("");

    const lessonRows = lessons.length
      ? lessons.map((lesson) => `
          <li class="lesson">
            <span class="task-badge mini">${esc(lesson.task || "?")}</span>
            <p class="lesson-text">${esc(lesson.text || "")}</p>
            <span class="lesson-wins mono" title="该经验被复用并带来提升的次数">+${fmtInt(lesson.wins)}</span>
          </li>`).join("")
      : '<li class="empty mini">暂无经验条目（等待首轮决策或数据库生成）</li>';

    const strategyRows = strategies.length
      ? strategies.map((row) => {
          const pct = Math.max(0, Math.min(1, row.win_rate)) * 100;
          const delta = row.mean_delta === null ? "" : `${row.mean_delta > 0 ? "+" : ""}${row.mean_delta.toFixed(3)}`;
          return `
            <li class="strategy-row">
              <span class="strategy-name mono" title="${esc(row.strategy)} / ${esc(row.postprocess_type)}">${esc(row.strategy)} <em>${esc(row.postprocess_type)}</em></span>
              <span class="strategy-bar"><i style="width:${pct.toFixed(1)}%"></i></span>
              <span class="strategy-rate mono">${pct.toFixed(0)}%</span>
              <span class="strategy-trials mono" title="promotions / trials">${fmtInt(row.promotions)}/${fmtInt(row.trials)}</span>
              <span class="strategy-delta mono ${Number(delta) > 0 ? "is-good" : ""}">${esc(delta)}</span>
            </li>`;
        }).join("")
      : '<li class="empty mini">暂无策略统计</li>';

    const maxSkillUses = skills.reduce((max, item) => Math.max(max, num(item.uses) || 0), 0) || 1;
    const skillRows = skills.length
      ? skills.map((item) => {
          const pct = (Math.max(0, num(item.uses) || 0) / maxSkillUses) * 100;
          return `
            <li class="skill-row">
              <span class="skill-name mono">${esc(item.skill || "?")}</span>
              <span class="skill-bar"><i style="width:${pct.toFixed(1)}%"></i></span>
              <span class="skill-uses mono">${fmtInt(item.uses)}</span>
            </li>`;
        }).join("")
      : '<li class="empty mini">暂无技能使用记录</li>';

    return `
      <div class="kernel-head">
        <div class="kernel-ident">
          <span class="kernel-kicker">AUTONOMOUS KERNEL</span>
          <div class="kernel-controller">
            ${controllerBadge(controller)}
            <span class="kernel-controller-label">${esc((CONTROLLER[controller] || CONTROLLER.pipeline).label)}</span>
            ${failures ? `<span class="warn-chip" title="内核失败 / fallback 次数">失败 ${fmtInt(failures)}</span>` : ""}
          </div>
        </div>
        <div class="kernel-counters">
          ${counters.map(([label, value]) => `
            <span class="kernel-counter"><i>${esc(label)}</i><b class="mono">${value == null || value === "" ? "—" : (typeof value === "string" ? esc(value) : fmtInt(value))}</b></span>`).join("")}
        </div>
        <form id="kernel-search-form" class="kernel-search" role="search">
          <input id="kernel-search-input" type="search" name="q" value="${esc(ui.search.query)}"
            placeholder="搜索 records / rounds（Enter）" autocomplete="off" aria-label="搜索运行记录">
          <button type="submit" class="search-btn">检索</button>
          ${ui.search.done ? '<button type="button" class="search-btn is-ghost" data-action="search-clear">清除</button>' : ""}
        </form>
      </div>

      <div class="kernel-grid">
        <section class="kernel-card memory-card">
          <div class="kernel-card-head">
            <h3>MEMORY <span>记忆库</span></h3>
            <span class="kernel-entries mono">${fmtInt(memory.entries)} 条</span>
          </div>
          <div class="memory-usage">
            <div class="memory-bar" role="img" aria-label="记忆占用 ${usagePct === null ? "未知" : usagePct.toFixed(0) + "%"}">
              <i style="width:${usagePct === null ? 0 : usagePct.toFixed(1)}%"></i>
            </div>
            <span class="memory-usage-text mono">${esc(memory.usage || (usagePct === null ? "—" : usagePct.toFixed(0) + "%"))}</span>
          </div>
          <div class="memory-meta">
            <span>目录 <code class="mono" title="${esc(memory.dir || "")}">${esc(memory.dir || "—")}</code></span>
            ${dbChip}
          </div>
          <ul class="memory-files">${fileRows}</ul>
          <ul class="memory-tasks">${taskRows || '<li class="empty mini">暂无任务级记忆</li>'}</ul>
        </section>

        <section class="kernel-card lessons-card">
          <div class="kernel-card-head">
            <h3>LESSONS <span>经验回溯</span></h3>
            <span class="kernel-entries mono">${lessons.length} 条</span>
          </div>
          <ol class="lesson-list">${lessonRows}</ol>
        </section>

        <section class="kernel-card strategy-card">
          <div class="kernel-card-head">
            <h3>STRATEGY <span>策略胜率</span></h3>
            <span class="kernel-entries mono">${strategies.length} 组</span>
          </div>
          <ul class="strategy-list">${strategyRows}</ul>
          <div class="skills-head"><span>技能使用 <i>SKILLS</i></span><span class="mono">${skills.length}</span></div>
          <ul class="skill-list">${skillRows}</ul>
        </section>
      </div>

      ${searchResultsHtml()}`;
  }

  function renderKernel() {
    const panel = $("kernel-panel");
    if (!panel) return;
    if (!ui.memory) {
      panel.innerHTML = '<p class="placeholder">暂无记忆 / 内核数据（/api/memory 不可用时会自动降级）</p>';
      return;
    }
    const focusInput = document.activeElement && document.activeElement.id === "kernel-search-input";
    const caret = focusInput ? document.activeElement.selectionStart : null;
    panel.innerHTML = kernelPanelHtml(ui.memory);
    if (focusInput) {
      const input = $("kernel-search-input");
      if (input) {
        input.focus();
        if (caret !== null) {
          try { input.setSelectionRange(caret, caret); } catch (error) { /* ignore */ }
        }
      }
    }
  }

  async function performSearch() {
    const input = $("kernel-search-input");
    const query = input ? input.value.trim() : ui.search.query;
    ui.search.query = query;
    ui.search.busy = true;
    ui.search.error = "";
    ui.search.done = false;
    renderKernel();
    try {
      const result = await fetchJSON(`/api/records/search?q=${encodeURIComponent(query)}&limit=20`, 6000);
      if (result.status !== 200 || !result.body || result.body.ok === false) {
        const detail = (result.body && (result.body.error || result.body.detail)) || `HTTP ${result.status}`;
        throw new Error(detail);
      }
      ui.search.items = Array.isArray(result.body.items) ? result.body.items : [];
      ui.search.total = num(result.body.total) || 0;
      ui.search.done = true;
    } catch (error) {
      ui.search.items = [];
      ui.search.total = 0;
      ui.search.error = error && error.message ? error.message : String(error);
      ui.search.done = true;
    } finally {
      ui.search.busy = false;
      renderKernel();
    }
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
    const goal = task.goal && typeof task.goal === "object" ? task.goal : {};
    const subgoals = Array.isArray(goal.subgoals) ? goal.subgoals : [];
    const agentInfo = findRoundAgent(task.key, round.index);
    const agent = agentInfo && agentInfo.agent ? agentInfo.agent : null;
    const trace = agent && Array.isArray(agent.trace) ? agent.trace : [];
    const agentUsage = (agent && agent.usage) || {};
    const shaParent = String(round.parent_checkpoint_sha256 || "");
    const shaCandidate = String(round.candidate_checkpoint_sha256 || "");
    const statusLabel = round.status === "running" ? "进行中" : round.status === "failed" ? "失败" : "已完成";

    const goalTask = ui.goal && Array.isArray(ui.goal.tasks)
      ? ui.goal.tasks.find((item) => String(item.key) === String(task.key))
      : null;
    const soft = goalTask && goalTask.goal_soft && typeof goalTask.goal_soft === "object" ? goalTask.goal_soft : null;
    const softAvailable = softGoalAvailable(soft);
    let softGoalSection = "";
    if (softAvailable) {
      const softMetric = soft.metric || metric;
      const softKeys = Object.keys(soft.targets || {}).length ? Object.keys(soft.targets) : Object.keys(soft.current || {});
      const softTarget = softKeys.length ? num((soft.targets || {})[softKeys[0]]) : null;
      const softCurrent = softKeys.length ? num((soft.current || {})[softKeys[0]]) : null;
      const softGap = num(soft.gap);
      const softVerdict = SOFT_VERDICT[soft.verdict] || SOFT_VERDICT.unknown;
      softGoalSection = `
        <section class="drawer-section">
          <h4>软目标 <i>SOFT GOAL</i></h4>
          <div class="soft-goal-row">
            <span class="goal-metric"><span>目标</span><b class="mono">${fmtMetricValue(softMetric, softTarget)}</b></span>
            <span class="goal-metric"><span>当前</span><b class="mono">${fmtMetricValue(softMetric, softCurrent)}</b></span>
            <span class="goal-metric"><span>缺口</span><b class="mono ${softGap !== null && softGap > 0 ? "is-bad" : "is-good"}">${fmtMetricValue(softMetric, softGap)}</b></span>
            <span class="goal-verdict ${softVerdict.cls}">${softVerdict.label}</span>
            ${soft.degraded ? '<span class="goal-flag is-warn">degraded</span>' : ""}
          </div>
          <p class="goal-note">软目标，不阻塞完成</p>
          ${soft.evidence ? `<p class="goal-evidence">${esc(soft.evidence)}</p>` : ""}
        </section>`;
    }
    const proposalSources = agent && agent.proposal_sources && typeof agent.proposal_sources === "object" && Object.keys(agent.proposal_sources).length
      ? agent.proposal_sources
      : null;
    const proposalSection = proposalSources
      ? `<section class="drawer-section"><h4>提案来源 <i>PROPOSALS</i></h4>${proposalBarHtml(proposalSources)}</section>`
      : "";

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
        ${softGoalSection}
        ${proposalSection}
        ${goal && (goal.goal || goal.rounds_target) ? `
        <section class="drawer-section">
          <h4>任务目标 <i>GOAL</i></h4>
          <p>${esc(goal.goal || "（未记录目标）")}</p>
          <div class="goal-meta mono">
            <span>轮次 <b>${esc(goal.rounds_done == null ? "—" : goal.rounds_done)} / ${esc(goal.rounds_target == null ? "—" : goal.rounds_target)}</b></span>
            <span>状态 <b>${esc(GOAL_STATUS[goal.status] || goal.status || "—")}</b></span>
            <span>verdict <b>${esc(goal.last_verdict || "—")}</b></span>
            <span>turns <b>${esc(goal.turns_used == null ? "—" : goal.turns_used)} / ${esc(goal.max_turns == null ? "—" : goal.max_turns)}</b></span>
          </div>
          ${subgoals.length ? `<ul class="goal-subgoals">${subgoals.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}
        </section>` : ""}
        <section class="drawer-section">
          <h4>内核轨迹 <i>KERNEL TRACE</i></h4>
          ${agent ? `
          <div class="agent-stats">
            <span>usage <b class="mono">${fmtInt(agentUsage.input)} in / ${fmtInt(agentUsage.output)} out / ${fmtInt(agentUsage.reasoning)} reason</b></span>
            <span>latency <b class="mono">${agent.latency == null ? "—" : Number(agent.latency).toFixed(1) + "s"}</b></span>
            <span>tools <b class="mono">${fmtInt(agent.tool_calls)}</b></span>
            ${agent.fallback ? '<span class="warn-chip">fallback</span>' : ""}
            ${agent.finished === false ? '<span class="chip">未完成</span>' : ""}
          </div>
          ${agent.error ? `<p class="agent-error">${esc(agent.error)}</p>` : ""}
          ${trace.length ? `<ol class="trace-list">${trace.map((step) => `
            <li class="trace-item ${step.ok === false ? "is-bad" : ""}">
              <span class="trace-dot" aria-hidden="true"></span>
              <span class="trace-tool mono">${esc(step.tool || "?")}</span>
              <span class="trace-detail">${esc(step.detail || "")}</span>
              <span class="trace-at mono">${esc(fmtClock(step.at))}</span>
            </li>`).join("")}</ol>` : '<p class="empty mini">本轮未记录工具轨迹（pipeline 模式或尚未开始）</p>'}
          ` : '<p class="empty mini">本轮无内核数据（round.agent 为空时会等待 /api/agent）</p>'}
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
    renderKernel();
    renderGoal();
    renderInsight();
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
  function hashString(text) {
    let hash = 2166136261;
    for (let i = 0; i < text.length; i += 1) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(36);
  }

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
      const [stateResult, eventsResult, memoryResult, agentResult, goalResult, knowledgeResult] = await Promise.allSettled([
        fetchJSON("/api/state"),
        fetchJSON(`/api/events?limit=${EVENTS_LIMIT}`),
        fetchJSON("/api/memory", 6000),
        fetchJSON("/api/agent", 6000),
        fetchJSON("/api/goal", 6000),
        fetchJSON("/api/knowledge", 6000),
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
      if (memoryResult.status === "fulfilled" && memoryResult.value.status === 200 && memoryResult.value.body && memoryResult.value.body.ok !== false) {
        ui.memory = memoryResult.value.body;
      }
      if (agentResult.status === "fulfilled" && agentResult.value.status === 200 && agentResult.value.body && agentResult.value.body.ok !== false) {
        ui.agent = agentResult.value.body;
      }
      if (goalResult.status === "fulfilled" && goalResult.value.status === 200 && goalResult.value.body && goalResult.value.body.ok !== false) {
        ui.goal = goalResult.value.body;
      }
      if (knowledgeResult.status === "fulfilled" && knowledgeResult.value.status === 200 && knowledgeResult.value.body && knowledgeResult.value.body.ok !== false) {
        ui.knowledge = knowledgeResult.value.body;
      }
      if (data._meta && data._meta.error) {
        showBanner(`数据源降级：${data._meta.error}`, "warn");
      } else {
        hideBanner();
      }
      const kernelSignature = `${ui.memory ? JSON.stringify(ui.memory) : ""}|${ui.agent ? JSON.stringify(ui.agent) : ""}|${ui.goal ? JSON.stringify(ui.goal) : ""}|${ui.knowledge ? JSON.stringify(ui.knowledge) : ""}`;
      const signature = signatureOf(data) + `|${ui.events.length}|${ui.events[0] ? ui.events[0].ts : ""}|${hashString(kernelSignature)}`;
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
      case "search-clear": {
        ui.search = { query: "", items: null, total: 0, busy: false, error: "", done: false };
        renderKernel();
        const input = $("kernel-search-input");
        if (input) input.focus();
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

  document.addEventListener("submit", (event) => {
    if (event.target && event.target.id === "kernel-search-form") {
      event.preventDefault();
      performSearch();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && ui.drawer) {
      closeDrawer();
      return;
    }
    if (event.key === "Escape" && ui.search.done) {
      ui.search = { query: "", items: null, total: 0, busy: false, error: "", done: false };
      renderKernel();
    }
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
