#!/bin/sh
# Independent prediction entry required by the rulebook.
#
# Reads JSONL requests ({"task":..., "inputs":...}) from stdin and writes one
# JSONL response per line, exactly like the official baselines/predict.py.
#
#   rsi-predict --checkpoint /path/checkpoint.npz [--postprocess '<json>']
#               [--mode batch|stream]
set -eu
export PYTHONPATH="/opt/zhixun/site-packages:/opt/zhixun/app${PYTHONPATH:+:$PYTHONPATH}"
exec python -m rsi.predict "$@"
