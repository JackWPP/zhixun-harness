# UCI Gas Sensor Drift — 外部独立验证（扩展数据，非训练）

官方 task A 仅含 batch1/batch2。本表用 UCI 公开 archive 的 batch1–batch10 逐批评估；
扩展数据只读、不参与训练（Guard 白名单排除），对应赛题 extended_data_validation 证据。

- baseline checkpoint: `datasets/cpu-a-guide-20261110-v1/checkpoint.npz`
- final checkpoint: `.scratch/rsi-ext/rsi-20261115-192527/A/experiments/exp-0011/candidate.npz`
- postprocess: `{"type": "recenter", "mode": "batch_mean", "weight": 0.3}`

| batch | rows | baseline macro-F1 | final macro-F1 | Δ |
|---|---:|---:|---:|---:|
| batch1 | 445 | 0.955754 | 0.975836 | +0.020083 |
| batch2 | 1244 | 0.714030 | 0.910207 | +0.196177 |
| batch3 | 1586 | 0.593897 | 0.683307 | +0.089410 |
| batch4 | 161 | 0.489335 | 0.536322 | +0.046987 |
| batch5 | 197 | 0.627000 | 0.883667 | +0.256667 |
| batch6 | 2300 | 0.322759 | 0.900198 | +0.577438 |
| batch7 | 3613 | 0.326685 | 0.795662 | +0.468977 |
| batch8 | 294 | 0.266411 | 0.629562 | +0.363151 |
| batch9 | 470 | 0.185163 | 0.534968 | +0.349805 |
| batch10 | 3600 | 0.380649 | 0.579614 | +0.198965 |

**micro weighted total**: baseline 0.425315 → final 0.746155 (+0.320840)

