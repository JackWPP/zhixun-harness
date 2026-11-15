"""``python -m rsi.dashboard`` 命令行入口。

示例
----
    ./.venv/bin/python -m rsi.dashboard --state .scratch/demo_state.json --port 7861
    ./.venv/bin/python -m rsi.dashboard                      # 自动生成并展示 v2 演示包 .scratch/demo_state.json
    ./.venv/bin/python -m rsi.dashboard --state runs/state.json --records runs/records.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .demo import DEFAULT_DEMO_STATE, write_demo_bundle
from .server import VERSION, create_server, serve

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_demo_state(path: Path, refresh: bool) -> None:
    if path.is_file() and not refresh:
        return
    print(f"[dashboard] 未指定 --state，生成 v2 演示数据包 → {path}")
    write_demo_bundle(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rsi.dashboard",
        description="智训引擎 RSI Harness 本地实时看板（零依赖，读 state.json）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--state", default=None, help="harness 写出的 state.json 路径；省略则使用 v2 演示数据包")
    parser.add_argument("--follow-dir", default=None,
                        help="运行根目录（如 .scratch/rsi-v05）；自动跟随其中最新的 <run>/state.json，"
                             "新 run 启动后无需重启看板")
    parser.add_argument("--records", default=None, help="records.jsonl 路径；省略时读取 state.json 同目录下的 records.jsonl")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7861, help="监听端口（0 表示随机空闲端口）")
    parser.add_argument("--demo-out", default=DEFAULT_DEMO_STATE, help="未指定 --state 时演示数据的生成路径")
    parser.add_argument("--refresh-demo", action="store_true", help="即使演示数据已存在也重新生成")
    parser.add_argument("--quiet", action="store_true", help="关闭每个请求的访问日志")
    parser.add_argument("--version", action="version", version=f"zhixun-dashboard {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    follow_dir = Path(args.follow_dir).expanduser() if args.follow_dir else None
    if follow_dir is not None:
        if not follow_dir.is_dir():
            print(f"[dashboard] 错误：--follow-dir 不是目录：{follow_dir}", file=sys.stderr)
            return 2
        latest = sorted(
            (p / "state.json" for p in follow_dir.iterdir()
             if p.is_dir() and (p / "state.json").is_file()),
            key=lambda p: p.stat().st_mtime_ns,
        )
        if not latest:
            print(f"[dashboard] 错误：{follow_dir} 下没有 */state.json", file=sys.stderr)
            return 2
        state_path = latest[-1]
    elif args.state:
        state_path = Path(args.state).expanduser()
        if not state_path.is_file():
            print(
                f"[dashboard] 错误：state 文件不存在：{state_path}\n"
                f"            先运行 harness 生成 state.json；想看演示数据可直接省略 "
                f"--state（自动生成 v2 演示包），或加 --refresh-demo 重建。",
                file=sys.stderr,
            )
            return 2
    else:
        state_path = Path(args.demo_out).expanduser()
        try:
            _ensure_demo_state(state_path, args.refresh_demo)
        except Exception as exc:  # noqa: BLE001 - CLI 边界，给出可读错误即可
            print(f"[dashboard] 错误：生成演示数据失败：{exc}", file=sys.stderr)
            return 1

    static_index = Path(__file__).resolve().parent / "static" / "index.html"
    if not static_index.is_file():
        print(f"[dashboard] 错误：静态资源缺失：{static_index}", file=sys.stderr)
        return 1

    records = Path(args.records).expanduser() if args.records else None
    try:
        server = create_server(
            state_path,
            host=args.host,
            port=args.port,
            records_path=records,
            quiet=args.quiet,
            follow_dir=follow_dir,
        )
    except OSError as exc:
        print(f"[dashboard] 错误：无法监听 {args.host}:{args.port}：{exc}", file=sys.stderr)
        return 1

    try:
        serve(server)
    except OSError as exc:
        print(f"[dashboard] 网络错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
