#!/usr/bin/env python
"""Build / inspect the offline curated knowledge index (``knowledge/index.sqlite``).

The evaluation environment has no network access, so the agent's external
knowledge is a set of Markdown documents curated in this repository and indexed
with SQLite FTS5.  This CLI is the maintenance entry point: rebuild the index
after editing documents, inspect what the agent will actually retrieve, and
check in CI that the committed index is up to date.

Examples
--------
    ./.venv/bin/python scripts/build_knowledge.py                 # build (idempotent)
    ./.venv/bin/python scripts/build_knowledge.py --force         # rebuild anyway
    ./.venv/bin/python scripts/build_knowledge.py --check         # exit 1 if stale
    ./.venv/bin/python scripts/build_knowledge.py -q "KS 阈值" -k 3
    ./.venv/bin/python scripts/build_knowledge.py -q "concept drift" --render
    ./.venv/bin/python scripts/build_knowledge.py --stats --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow ``./.venv/bin/python scripts/...``
    sys.path.insert(0, str(REPO_ROOT))

from rsi.knowledge import (  # noqa: E402
    DEFAULT_ROOT,
    KnowledgeBase,
    fts5_available,
    render_for_brief,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/inspect the offline curated knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[-1],
    )
    parser.add_argument("--root", default=None,
                        help=f"knowledge directory (default: {DEFAULT_ROOT})")
    parser.add_argument("--db", default=None,
                        help="index path (default: <root>/index.sqlite)")
    parser.add_argument("--engine", choices=("auto", "fts5", "like"), default="auto",
                        help="force the retrieval engine; auto detects FTS5")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="exit 1 when the index is missing or stale")
    mode.add_argument("--stats", action="store_true",
                      help="print index statistics only")
    mode.add_argument("--list-slugs", action="store_true",
                      help="print indexed document slugs")
    parser.add_argument("--force", action="store_true", help="rebuild even if fresh")
    parser.add_argument("-q", "--query", default=None, help="run a retrieval query")
    parser.add_argument("--tags", default=None, help="comma separated tag filter")
    parser.add_argument("-k", type=int, default=3, help="number of hits (default 3)")
    parser.add_argument("--snippet", type=int, default=500, help="max snippet chars")
    parser.add_argument("--render", action="store_true",
                        help="render the hits exactly as they would be injected into a brief")
    parser.add_argument("--json", action="store_true", help="machine readable output")
    return parser


def _knowledge(args) -> KnowledgeBase:
    use_fts = None
    if args.engine == "fts5":
        use_fts = True
    elif args.engine == "like":
        use_fts = False
    return KnowledgeBase(args.root or DEFAULT_ROOT, db_path=args.db, use_fts=use_fts)


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    kb = _knowledge(args)

    if args.query:
        if not kb.db_path.is_file() or args.force:
            kb.build(force=args.force)
        else:
            kb.ensure_index()
        hits = kb.search(args.query, tags=args.tags, k=args.k, max_snippet=args.snippet)
        if args.render:
            print(render_for_brief(hits, max_chars=4000))
            return 0
        if args.json:
            print(json.dumps({"query": args.query, "hits": hits}, ensure_ascii=False, indent=2))
            return 0
        if not hits:
            print("（无命中）")
            return 1
        for index, hit in enumerate(hits, start=1):
            print(f"[K{index}] {hit['title']}  ({hit['slug']})")
            print(f"     tags: {','.join(hit['tags'])}")
            print(f"     path: {hit['path']}  score: {hit['score']}")
            print(f"     {hit['snippet']}")
        return 0

    result = kb.ensure_index(force=args.force)
    stats = kb.stats()
    if args.list_slugs:
        slugs = kb.slugs()
        if args.json:
            print(json.dumps(slugs, ensure_ascii=False))
        else:
            print("\n".join(slugs))
        return 0
    if args.stats:
        payload = stats
    else:
        payload = {"built": result.get("built"), "reason": result.get("reason"),
                   "engine": stats["engine"], "docs": stats["docs"],
                   "db": stats["db"], "bytes": stats["bytes"],
                   "built_at": stats["built_at"], "fts5_available": fts5_available()}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    if args.check and result.get("built"):
        print("知识索引已过期并已重建（--check 失败，请提交新索引）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
