"""Frozen workspace v2 entry point. The platform appends describe, train or predict."""
import os
import sys

# The host can call this Python file directly, bypassing entrypoint.sh.  Set
# conservative defaults before the SDK loads contest_submission (and NumPy).
# Explicit host/user settings remain authoritative.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BASELINE_CPU_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("RSI_BUILD_MANIFEST", "/opt/zhixun/app/build-manifest.json")

# The image keeps the application and its dependencies under /opt/zhixun; make
# the bootstrap independent from the caller's environment (the host starts this
# file directly as ``python3 /app/harness.py ARG``).
for _path in ("/opt/zhixun/site-packages", "/opt/zhixun/app"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from contest_harness.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
