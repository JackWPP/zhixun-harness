#!/usr/bin/env python3
"""契约检查（task-4 交付物，只读）。

用途
----
在「正式评测未开放、官方 SDK/文档缺失」的前提下，把可以客观验证的契约固化成一个
可重复执行、退出码化的检查：

1. 官方 baseline 三方校验：仓库 zip == 平台下载 zip == archives/SHA256SUMS，
   且包内 baseline-info.json 声明的 checkpoint_sha256 == 实测值。
2. 平台契约快照检查（离线用已抓取快照；--online 会重新抓取 11 个只读端点并做差分）。
3. 本仓库交付物完整性：docs/19_SUBMISSION_CONTRACT.md、scripts/emulate_stream_platform.py。
4. 凭据卫生：.scratch/platform/token 不进 git、不进任何交付物/日志。

退出码
------
0 = 通过（允许 INFO 级「已知不一致」，见 KNOWN 表）
1 = 硬失败（必须修）
2 = 仅告警（未预期漂移 / 快照过期 / 最小前提缺失）
3 = 网络或鉴权失败（--online 时连不上平台 / token 无效）

只读保证
--------
* 平台侧只发 GET；token 只从 .scratch/platform/token 读取，绝不在输出/日志中出现。
* 本地只写 .scratch/contract/api/ 与 .scratch/contract/logs/。

用法
----
    ./.venv/bin/python scripts/contract_check.py                 # 离线检查（默认）
    ./.venv/bin/python scripts/contract_check.py --online        # 刷新 API 快照 + 检查
    ./.venv/bin/python scripts/contract_check.py --online --strict-baselines -v
    ./.venv/bin/python scripts/contract_check.py --json          # 机器可读输出
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://rsi-index.115.zone/api/v1"
TOKEN_FILE = ROOT / ".scratch" / "platform" / "token"
CONTRACT_DIR = ROOT / ".scratch" / "contract"
API_DIR = CONTRACT_DIR / "api"
LOG_DIR = CONTRACT_DIR / "logs"
BASELINE_DIR = CONTRACT_DIR / "baselines"
DOC_FILE = ROOT / "docs" / "19_SUBMISSION_CONTRACT.md"
EMULATOR_FILE = ROOT / "scripts" / "emulate_stream_platform.py"
SHA256SUMS = ROOT / "archives" / "SHA256SUMS"

ENDPOINTS = [
    "rules",
    "announcements",
    "baselines",
    "dashboard",
    "submissions",
    "image-archives",
    "runs",
    "teams",
    "leaderboard",
    "module-visibility",
    "scoring-rulesets",
]

# 已确认的、暂不可消除的差异：只降级为 INFO，不让退出码变红。
# code_sha256：构建方对未随包公开的源码树做的声明式摘要，算法未公开（实测约 30 种组合均不匹配）。
# label_status：/baselines 与 workspace_suite 把 A/C 标为 approved，但包内 baseline-info.json 写 draft。
KNOWN_CODE_SHA256_UNVERIFIABLE = {
    "cpu-a-guide-20261110-v1",
    "cpu-b-rul-20261110-v1",
    "cpu-c-guide-20261110-v1",
}
KNOWN_LABEL_STATUS_MISMATCH = {
    "cpu-a-guide-20261110-v1": ("draft", "approved"),
    "cpu-c-guide-20261110-v1": ("draft", "approved"),
}

# 2026-11-15T23:18 CST 快照的权威值；值漂移只告警（平台可能在正式开放时改），
# 但字段缺失视为硬失败（契约被改坏）。
EXPECTED_RULES = {
    "execution_protocol": "contest.harness.workspace.v1",
    "evaluation_config_id": "76e44dafec4225a250977eed7a48b811",
    "formula": "contest-objective-20261110-rul-v3",
    "dataset_family": "uci-gas-drift",
    "task": "classification",
    "metric": "macro_f1",
    "metric_direction": "higher",
    "target_unit": "class",
    "evaluation_mode": "batch",
    "wall_seconds": 21600,
    "token_budget": 8000000,
    "training_budget": 0,
    "label_budget": 9628,
    "allowed_models": ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B"],
}
REQUIRED_RULES_FIELDS = list(EXPECTED_RULES) + [
    "tie_policy",
    "records_policy",
    "workspace_suite",
]
REQUIRED_TOP_FIELDS = [
    "formal_ready",
    "blockers",
    "trial_enabled",
    "trial_blockers",
    "trial_usage",
    "formal_usage",
]

DOC_MARKERS = [
    "19.0 TL;DR",
    "已证实",
    "推断",
    "未知",
    "上线前验证步骤",
    "NO-GO",
]

SECRET_RE = re.compile(rb"(?:sk-[A-Za-z0-9_\-]{16,}|Bearer\s+[0-9a-fA-F]{32,})")
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".txt", ".sh", ".yml", ".yaml", ".toml", ".cfg"}

EXIT_OK, EXIT_FAIL, EXIT_WARN, EXIT_NET = 0, 1, 2, 3


class NetError(RuntimeError):
    """--online 时的网络/鉴权失败。"""


class Report:
    def __init__(self, verbose: bool = False) -> None:
        self.rows: list[dict] = []
        self.verbose = verbose

    def add(self, level: str, name: str, detail: str) -> None:
        self.rows.append({"level": level, "name": name, "detail": detail})

    def pass_(self, name: str, detail: str = "") -> None:
        self.add("PASS", name, detail)

    def info(self, name: str, detail: str) -> None:
        self.add("INFO", name, detail)

    def warn(self, name: str, detail: str) -> None:
        self.add("WARN", name, detail)

    def fail(self, name: str, detail: str) -> None:
        self.add("FAIL", name, detail)

    def counts(self) -> dict:
        return {
            "pass": sum(1 for r in self.rows if r["level"] == "PASS"),
            "info": sum(1 for r in self.rows if r["level"] == "INFO"),
            "warn": sum(1 for r in self.rows if r["level"] == "WARN"),
            "fail": sum(1 for r in self.rows if r["level"] == "FAIL"),
        }

    def exit_code(self) -> int:
        c = self.counts()
        if c["fail"]:
            return EXIT_FAIL
        if c["warn"]:
            return EXIT_WARN
        return EXIT_OK


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_snapshots() -> dict:
    """读本地已抓取的 API 快照。"""
    bodies = {}
    for name in ENDPOINTS:
        bodies[name] = read_json(API_DIR / "body" / f"{name}.json")
    return bodies


def read_token() -> str | None:
    try:
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not token:
        return None
    return token


def fetch_endpoint(name: str, token: str, api_base: str, timeout: float = 25.0):
    url = f"{api_base.rstrip('/')}/{name}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")), raw
    except urllib.error.HTTPError as exc:  # 401/403 视为鉴权失败
        raise NetError(f"GET /{name} -> HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NetError(f"GET /{name} 失败: {exc}") from exc
    except ValueError as exc:
        raise NetError(f"GET /{name} 返回非 JSON: {exc}") from exc


def diff_json(old, new, path: str = "", depth: int = 0) -> list[str]:
    if depth > 3:
        return []
    out: list[str] = []
    if type(old) is not type(new):
        out.append(f"{path}: {type(old).__name__} -> {type(new).__name__}")
    elif isinstance(old, dict):
        for key in sorted(set(old) | set(new)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in old:
                out.append(f"{sub}: added = {short(new[key])}")
            elif key not in new:
                out.append(f"{sub}: removed")
            else:
                out.extend(diff_json(old[key], new[key], sub, depth + 1))
    elif isinstance(old, list):
        if len(old) != len(new):
            out.append(f"{path}: len {len(old)} -> {len(new)}")
        for idx, (a, b) in enumerate(zip(old, new)):
            out.extend(diff_json(a, b, f"{path}[{idx}]", depth + 1))
    elif old != new:
        out.append(f"{path}: {short(old)} -> {short(new)}")
    return out


def short(value) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = repr(value)
    return text if len(text) <= 90 else text[:87] + "..."


# ---------------------------------------------------------------- online


def refresh_snapshots(report: Report, api_base: str) -> dict:
    token = read_token()
    if not token:
        raise NetError(f"缺少 token 文件 {TOKEN_FILE.relative_to(ROOT)}（期望 0600）")

    API_DIR.mkdir(parents=True, exist_ok=True)
    (API_DIR / "body").mkdir(parents=True, exist_ok=True)
    old_bodies = load_snapshots()
    old_meta = read_json(API_DIR / "fetch_meta.json") or {}
    new_bodies = {}
    meta = {
        "fetched_at": iso(now_utc()),
        "api_base": api_base,
        "endpoints": {},
    }
    changes: dict[str, list[str]] = {}
    for name in ENDPOINTS:
        body, raw = fetch_endpoint(name, token, api_base)
        new_bodies[name] = body
        meta["endpoints"][name] = {
            "http_status": 200,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "top_keys": sorted(body.keys()) if isinstance(body, dict) else [],
        }
        diff = diff_json(old_bodies.get(name), body)
        changes[name] = diff
        target = API_DIR / "body" / f"{name}.json"
        target.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (API_DIR / "fetch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    changes_path = API_DIR / f"changes_{stamp}.json"
    changes_doc = {
        "generated_at": iso(now_utc()),
        "previous_snapshot_at": old_meta.get("fetched_at", "unknown"),
        "changes": {k: v for k, v in changes.items() if v},
    }
    changes_path.write_text(
        json.dumps(changes_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    total = sum(len(v) for v in changes.values())
    if total:
        report.warn("api.drift", f"检出 {total} 处字段变化，见 {changes_path.relative_to(ROOT)}")
        for name, fields in changes.items():
            for field in fields[:20]:
                report.info(f"api.drift.{name}", field)
    else:
        report.pass_("api.drift", "与上一份快照逐字段一致")
    report.pass_("api.fetch", f"11 个只读端点抓取成功（{iso(now_utc())}）")
    return new_bodies


# ---------------------------------------------------------------- checks


def baseline_checks(report: Report, bodies: dict, strict: bool) -> None:
    verification = read_json(BASELINE_DIR / "verification.json")
    if not verification or not verification.get("packages"):
        report.warn("baseline.verification", "缺少 .scratch/contract/baselines/verification.json")
        verification = {"packages": []}

    declared_zip: dict[str, str] = {}
    try:
        for line in SHA256SUMS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                declared_zip[parts[1].lstrip("*")] = parts[0]
    except OSError:
        report.fail("baseline.sha256sums", "archives/SHA256SUMS 不存在")
        declared_zip = {}

    api_items = {}
    if isinstance(bodies.get("baselines"), dict):
        for item in bodies["baselines"].get("items", []):
            api_items[item.get("id")] = item

    if not verification["packages"]:
        # 即便 verification.json 缺失，也尽量直接从 zip 重新算
        for zip_path in sorted((ROOT / "archives").glob("cpu-*.zip")):
            verification["packages"].append({"id": zip_path.stem, "zip": {}, "checkpoint": {}, "code": {}})

    for pkg in verification["packages"]:
        pid = pkg.get("id") or "unknown"
        zip_info = pkg.get("zip") or {}
        repo_zip = ROOT / (zip_info.get("repo") or f"archives/{pid}.zip")
        downloaded = ROOT / (zip_info.get("downloaded") or "") if zip_info.get("downloaded") else None

        # 1) zip 三方
        if not repo_zip.is_file():
            report.fail(f"baseline.zip.{pid}", f"仓库包不存在: {rel(repo_zip)}")
            continue
        repo_sha = sha256_file(repo_zip)
        sums_sha = declared_zip.get(repo_zip.name)
        if sums_sha is None:
            report.fail(f"baseline.sums.{pid}", f"SHA256SUMS 未列出 {repo_zip.name}")
        elif sums_sha != repo_sha:
            report.fail(f"baseline.sums.{pid}", f"{repo_zip.name} 与 SHA256SUMS 不符")
        else:
            report.pass_(f"baseline.sums.{pid}", f"zip sha256 = {repo_sha[:16]}… 与 SHA256SUMS 一致")

        if downloaded is not None:
            if not downloaded.is_file():
                report.warn(f"baseline.downloaded.{pid}", f"下载包不存在: {rel(downloaded)}")
            else:
                dl_sha = sha256_file(downloaded)
                if dl_sha != repo_sha:
                    report.fail(
                        f"baseline.downloaded.{pid}", f"下载包与仓库包不一致 {dl_sha[:16]}… != {repo_sha[:16]}…"
                    )
                else:
                    report.pass_(f"baseline.downloaded.{pid}", "下载包与仓库包逐字节一致")

        # 2) checkpoint 声明值 vs 包内实测
        try:
            with zipfile.ZipFile(repo_zip) as zf:
                names = set(zf.namelist())
                info_name = next((n for n in names if n.endswith("baseline-info.json")), None)
                ckpt_name = next((n for n in names if n.endswith("checkpoint.npz")), None)
                if not info_name or not ckpt_name:
                    report.fail(f"baseline.layout.{pid}", "包内缺少 baseline-info.json 或 checkpoint.npz")
                    continue
                info = json.loads(zf.read(info_name).decode("utf-8"))
                actual = "sha256:" + hashlib.sha256(zf.read(ckpt_name)).hexdigest()
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            report.fail(f"baseline.zip.{pid}", f"zip 读取失败: {exc}")
            continue

        declared_ckpt = info.get("checkpoint_sha256")
        if actual == declared_ckpt:
            report.pass_(f"baseline.checkpoint.{pid}", f"checkpoint sha256 与包内声明一致（{actual[:20]}…）")
        else:
            report.fail(
                f"baseline.checkpoint.{pid}",
                f"checkpoint sha256 实测 {actual[:20]}… != 声明 {str(declared_ckpt)[:20]}…",
            )

        # 3) code_sha256：已知不可复现 → INFO；若出现新异常（声明值变了）→ WARN
        declared_code = info.get("code_sha256")
        code_rec = pkg.get("code") or {}
        if pid in KNOWN_CODE_SHA256_UNVERIFIABLE:
            report.info(
                f"baseline.code.{pid}",
                f"code_sha256 不可复现（构建方声明 {str(declared_code)[:20]}…，算法未公开）——已知项",
            )
        elif code_rec.get("match") is False:
            level = report.fail if strict else report.warn
            level(f"baseline.code.{pid}", "code_sha256 声明值无法从包内容复现（未列入已知表）")

        # 4) label_status：包内 vs API
        pkg_label = info.get("label_status")
        api_label = (api_items.get(pid) or {}).get("label_status")
        if api_label is None:
            report.warn(f"baseline.label.{pid}", "API 快照缺少该 baseline 的 label_status")
        elif pkg_label == api_label:
            report.pass_(f"baseline.label.{pid}", f"label_status={pkg_label}（包内=API）")
        else:
            known = KNOWN_LABEL_STATUS_MISMATCH.get(pid)
            if known == (pkg_label, api_label):
                report.info(
                    f"baseline.label.{pid}",
                    f"已知不一致：包内 {pkg_label} vs API {api_label}（A/C 待组委会澄清）",
                )
            elif strict:
                report.fail(f"baseline.label.{pid}", f"包内 {pkg_label} vs API {api_label} 不一致且未登记")
            else:
                report.warn(f"baseline.label.{pid}", f"包内 {pkg_label} vs API {api_label} 不一致且未登记")

        # 5) 数据集/task 一致性（包内声明 vs API）
        api_item = api_items.get(pid) or {}
        for key in ("task", "target_unit", "parameter_count"):
            if key in info and key in api_item and info[key] != api_item[key]:
                report.warn(
                    f"baseline.{key}.{pid}", f"包内 {short(info[key])} vs API {short(api_item[key])}"
                )


def rules_checks(report: Report, bodies: dict) -> None:
    rules_doc = bodies.get("rules")
    if not isinstance(rules_doc, dict) or not rules_doc:
        report.warn("rules.snapshot", "无 /rules 快照（先跑 --online）")
        return
    for key in REQUIRED_TOP_FIELDS:
        if key not in rules_doc:
            report.fail("rules.top", f"/rules 缺少顶层字段 {key}")
    rules = rules_doc.get("rules")
    if not isinstance(rules, dict):
        report.fail("rules.body", "/rules.rules 缺失或不是对象")
        return
    for key in REQUIRED_RULES_FIELDS:
        if key not in rules:
            report.fail("rules.fields", f"/rules.rules 缺少字段 {key}")
    for key, expected in EXPECTED_RULES.items():
        if key not in rules:
            continue
        if rules[key] != expected:
            report.warn(f"rules.{key}", f"值漂移：{short(rules[key])}（期望 {short(expected)}）")

    suite = rules.get("workspace_suite")
    if not isinstance(suite, dict):
        report.fail("rules.workspace_suite", "workspace_suite 缺失")
    else:
        if suite.get("rounds") != 10:
            report.warn("rules.workspace_suite.rounds", f"rounds={short(suite.get('rounds'))}（期望 10）")
        datasets = suite.get("datasets") or []
        if len(datasets) != 3:
            report.warn("rules.workspace_suite.datasets", f"datasets 数量 {len(datasets)}（期望 3）")

    formal_ready = rules_doc.get("formal_ready")
    if formal_ready is True:
        report.info("rules.formal_ready", "formal_ready=true：正式评测已开放，请立即复核 19.4 未知项")
    else:
        report.info("rules.formal_ready", f"formal_ready={short(formal_ready)}（正式评测未开放，符合预期）")

    if rules_doc.get("trial_enabled") and not rules_doc.get("trial_blockers"):
        report.pass_("rules.trial", "试运行可用（trial_enabled=true, trial_blockers 为空）")
    else:
        report.warn("rules.trial", "试运行当前不可用（trial_enabled 或 trial_blockers 变化）")

    usage = rules_doc.get("trial_usage") or {}
    remaining = usage.get("remaining") if isinstance(usage, dict) else None
    if isinstance(remaining, int) and remaining <= 0:
        report.warn("rules.trial_usage", "试运行剩余次数为 0")


def image_checks(report: Report, bodies: dict) -> None:
    doc = bodies.get("image-archives")
    if not isinstance(doc, dict):
        report.warn("image.snapshot", "无 /image-archives 快照")
        return
    items = doc.get("items") or []
    if not items:
        report.warn("image.archive", "尚无镜像归档（需要先 docker save + POST /image-archives）")
        return
    for item in items:
        status = item.get("status")
        name = f"image.{str(item.get('id'))[:8]}"
        if status == "ready":
            report.pass_(name, f"镜像可用 size={item.get('size_bytes')} bytes")
        elif status in {"failed", "expired"}:
            report.fail(name, f"镜像状态 {status}，需重新上传（{item.get('error_code') or '无错误码'}）")
        else:
            report.warn(name, f"镜像状态 {status}（等待导入/核验中）")


def docs_checks(report: Report) -> None:
    if not DOC_FILE.is_file():
        report.warn("docs.19", f"缺少 {rel(DOC_FILE)}")
    else:
        text = DOC_FILE.read_text(encoding="utf-8", errors="replace")
        missing = [m for m in DOC_MARKERS if m not in text]
        if missing:
            report.warn("docs.19", f"缺少章节标记: {', '.join(missing)}")
        else:
            report.pass_("docs.19", f"契约文档存在（{len(text)} chars，{len(DOC_MARKERS)} 个必需章节齐全）")
        if "U1" not in text or "NO-GO" not in text:
            report.warn("docs.19.risk", "文档缺少未知项/NO-GO 清单")

    if not EMULATOR_FILE.is_file():
        report.warn("scripts.emulator", f"缺少 {rel(EMULATOR_FILE)}")
    else:
        try:
            compile(EMULATOR_FILE.read_text(encoding="utf-8"), str(EMULATOR_FILE), "exec")
            report.pass_("scripts.emulator", "scripts/emulate_stream_platform.py 存在且可编译")
        except SyntaxError as exc:
            report.fail("scripts.emulator", f"语法错误: {exc}")
    if not Path(__file__).is_file():
        report.fail("scripts.contract_check", "自检失败：本脚本路径不存在")


def hygiene_checks(report: Report) -> None:
    token = read_token()
    if not token:
        report.warn("hygiene.token", "缺少 .scratch/platform/token（--online 需要）")
        return
    try:
        st = TOKEN_FILE.stat()
        if st.st_mode & 0o077:
            report.warn("hygiene.token.perm", f"{rel(TOKEN_FILE)} 权限 {oct(st.st_mode & 0o777)}（期望 0600）")
        else:
            report.pass_("hygiene.token.perm", "token 文件权限 0600")
    except OSError:
        pass

    token_bytes = token.encode("utf-8")
    hits: list[str] = []

    # 1) token 不能被 git 跟踪
    try:
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(TOKEN_FILE.relative_to(ROOT))],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if tracked.returncode == 0:
            report.fail("hygiene.token.tracked", f"{rel(TOKEN_FILE)} 被 git 跟踪！")
        else:
            report.pass_("hygiene.token.tracked", "token 文件未被 git 跟踪")
    except (OSError, subprocess.SubprocessError):
        report.info("hygiene.token.tracked", "无法执行 git（跳过跟踪检查）")

    # 2) token 不能出现在交付物 / 文档 / 脚本
    scan_roots = [ROOT / "docs", ROOT / "scripts", CONTRACT_DIR]
    for base in scan_roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if token_bytes in data:
                hits.append(str(path.relative_to(ROOT)))
            if SECRET_RE.search(data):
                hits.append(f"{path.relative_to(ROOT)} (疑似密钥模式)")
    if hits:
        report.fail("hygiene.secret", "发现疑似凭据: " + ", ".join(sorted(set(hits))[:10]))
    else:
        report.pass_("hygiene.secret", "docs/、scripts/、.scratch/contract/ 未发现 token 或疑似密钥")

    # 3) 交付物日志里也不能带 token（logs 目录由本脚本写）
    if LOG_DIR.exists():
        for path in sorted(LOG_DIR.glob("contract_check_*.json"))[-5:]:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if token_bytes in data:
                report.fail("hygiene.log", f"{rel(path)} 含 token")


def snapshot_freshness(report: Report) -> None:
    meta = read_json(API_DIR / "fetch_meta.json")
    if not meta:
        report.warn("api.snapshot", "无 fetch_meta.json（先跑 --online 生成快照）")
        return
    fetched = meta.get("fetched_at")
    try:
        dt = datetime.fromisoformat(str(fetched))
    except ValueError:
        report.warn("api.snapshot", f"无法解析快照时间 {fetched!r}")
        return
    age = now_utc() - dt.astimezone(timezone.utc)
    if age > timedelta(hours=24):
        report.warn("api.snapshot.age", f"最新快照已过期 {age.total_seconds() / 3600:.1f} 小时（重跑 --online）")
    else:
        report.pass_("api.snapshot.age", f"快照时间 {fetched}，{age.total_seconds() / 60:.0f} 分钟前")


# ---------------------------------------------------------------- main


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_report(report: Report, as_json: bool) -> None:
    counts = report.counts()
    if as_json:
        print(json.dumps({"counts": counts, "checks": report.rows}, ensure_ascii=False, indent=2))
        return
    width = max((len(r["name"]) for r in report.rows), default=10)
    for row in report.rows:
        if not report.verbose and row["level"] in {"PASS"} and not row["detail"]:
            continue
        print(f"[{row['level']:4}] {row['name']:<{width}}  {row['detail']}")
    print(
        f"\n合计: PASS={counts['pass']} INFO={counts['info']} WARN={counts['warn']} FAIL={counts['fail']}"
    )


def write_log(report: Report, mode: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    path = LOG_DIR / f"contract_check_{stamp}.json"
    doc = {
        "generated_at": iso(now_utc()),
        "mode": mode,
        "counts": report.counts(),
        "checks": report.rows,
        "exit_code": report.exit_code(),
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="智训引擎提交契约检查（只读、退出码化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="退出码: 0=通过 1=硬失败 2=仅告警 3=网络/鉴权失败",
    )
    parser.add_argument("--online", action="store_true", help="重新抓取 11 个只读 API 端点并做差分")
    parser.add_argument("--offline", action="store_true", help="仅用本地快照（默认行为）")
    parser.add_argument("--strict-baselines", action="store_true", help="未登记的不一致按硬失败处理")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument("--no-log", action="store_true", help="不写 .scratch/contract/logs/")
    parser.add_argument("--api-base", default=API_BASE, help=f"API base（默认 {API_BASE}）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    report = Report(verbose=args.verbose)
    mode = "online" if args.online else "offline"

    if args.online:
        try:
            bodies = refresh_snapshots(report, args.api_base)
        except NetError as exc:
            report.fail("api.online", str(exc))
            print_report(report, args.json)
            write_log(report, mode)
            return EXIT_NET
    else:
        bodies = load_snapshots()

    baseline_checks(report, bodies, strict=args.strict_baselines)
    rules_checks(report, bodies)
    image_checks(report, bodies)
    snapshot_freshness(report)
    docs_checks(report)
    hygiene_checks(report)

    print_report(report, args.json)
    if not args.no_log:
        path = write_log(report, mode)
        if not args.json:
            print(f"日志: {rel(path)}")
    code = report.exit_code()
    if not args.json:
        print(f"退出码: {code} ({'通过' if code == 0 else '硬失败' if code == 1 else '仅告警' if code == 2 else '网络/鉴权失败'})")
    return code


if __name__ == "__main__":
    sys.exit(run())
