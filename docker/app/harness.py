"""Frozen workspace v2 entry point. The platform appends describe, train or predict."""
import sys

# The image keeps the application and its dependencies under /opt/zhixun; make
# the bootstrap independent from the caller's environment (the host starts this
# file directly as ``python3 /app/harness.py ARG``).
for _path in ("/opt/zhixun/site-packages", "/opt/zhixun/app"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from contest_harness.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
