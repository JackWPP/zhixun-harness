"""智训引擎 RSI Harness 本地看板（零依赖 WebUI）。

模块结构::

    rsi/dashboard/
      __init__.py     # 包入口，导出 VERSION / create_server / StateStore
      __main__.py     # `python -m rsi.dashboard` 命令行入口
      server.py       # HTTP 服务端：/api/state、/api/events、静态资源
      static/
        index.html    # 页面骨架（中文 UI）
        app.js        # 手写 vanilla JS：轮询、渲染、过滤、详情抽屉
        style.css     # 深色工业控制塔风格样式

快速开始::

    ./.venv/bin/python scripts/mock_state.py --out .scratch/demo_state.json
    ./.venv/bin/python -m rsi.dashboard --state .scratch/demo_state.json --port 7861
    # 浏览器打开 http://127.0.0.1:7861/
"""
from __future__ import annotations

from .server import (
    DashboardServer,
    StateStore,
    StateUnavailable,
    VERSION,
    create_server,
    serve,
)

__version__ = VERSION

__all__ = [
    "DashboardServer",
    "StateStore",
    "StateUnavailable",
    "VERSION",
    "create_server",
    "serve",
]
