"""``python -m rsi.dashboard`` 命令行入口。

示例
----
    ./.venv/bin/python -m rsi.dashboard --state .scratch/demo_state.json --port 7861
    ./.venv/bin/python -m rsi.dashboard                      # 自动生成并展示 .scratch/demo_state.json
    ./.venv/bin/python -m rsi.dashboard --state runs/state.json --records runs/records.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .server import VERSION, create_server, serve

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_STATE = ".scratch/demo_state.json"
MOCK_SCRIPT = REPO_ROOT / "scripts" / "mock_state.py"


def _load_mock_module():
    """按文件路径加载 scripts/mock_state.py（scripts/ 不是 Python 包）。"""
    if not MOCK_SCRIPT.is_file():
        raise FileNotFoundError(f"找不到 mock 数据生成器：{MOCK_SCRIPT}")
    spec = importlib.util.spec_from_file_location("zhixun_mock_state", MOCK_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 mock 数据生成器：{MOCK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_demo_state(path: Path, refresh: bool) -> None:
    if path.is_file() and not refresh:
        return
    print(f"[dashboard] 未指定 --state，生成演示数据 → {path}")
    mock = _load_mock_module()
    mock.write_state(mock.build_state(), path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rsi.dashboard",
        description="智训引擎 RSI Harness 本地实时看板（零依赖，读 state.json）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--state", default=None, help="harness 写出的 state.json 路径；省略则使用演示数据")
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

    if args.state:
        state_path = Path(args.state).expanduser()
        if not state_path.is_file():
            print(
                f"[dashboard] 错误：state 文件不存在：{state_path}\n"
                f"            先运行 harness 生成 state.json，或用 "
                f"`./.venv/bin/python scripts/mock_state.py --out {DEFAULT_DEMO_STATE}` 生成演示数据。",
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
