# Full-scale batch-stream rehearsal

## L1 — protocol-level full-data streams

| task | arm | seed | batches | mean primary | worst | hidden mean | decisions | adapt kept | wall(s) |
|---|---|---|---|---|---|---|---|---|---|
| A | recenter | 42 | 10 | 0.5589 | 0.3933 | - | retain:10 | - | 20.4 |
| A | frozen | 42 | 10 | 0.4862 | 0.1852 | - | retain:10 | - | 20.6 |
| A | frozen | 43 | 10 | 0.4862 | 0.1852 | - | retain:10 | - | 20.7 |
| A | pair | 43 | 10 | 0.5276 | 0.2556 | - | retain:10 | - | 17.6 |
| A | pair | 42 | 10 | 0.5276 | 0.2556 | - | retain:10 | - | 17.8 |
| A | recenter | 43 | 10 | 0.5589 | 0.3933 | - | retain:10 | - | 18.1 |
| A | adapt | 42 | 10 | 0.7385 | 0.5619 | - | retain:10 | 9/9 | 39.4 |
| A | adapt | 43 | 10 | 0.7625 | 0.6345 | - | retain:10 | 9/9 | 40.3 |
| B | frozen | 42 | 8 | 650.0194 | 1252.5771 | - | retain:8 | - | 4.7 |
| B | frozen | 43 | 8 | 650.0194 | 1252.5771 | - | retain:8 | - | 11.4 |
| B | recenter | 42 | 8 | 572.6532 | 1081.6753 | - | retain:8 | - | 3.2 |
| B | recenter | 43 | 8 | 572.6532 | 1081.6753 | - | retain:8 | - | 4.4 |
| B | adapt | 42 | 8 | 435.5636 | 981.9687 | - | retain:8 | 5/7 | 12.8 |
| B | adapt | 43 | 8 | 432.3986 | 962.7438 | - | retain:8 | 5/7 | 13.1 |
| A | full | 42 | 10 | 0.7653 | 0.6227 | - | promote:9,retain:1 | 9/9 | 342.2 |
| A | full | 43 | 10 | 0.7684 | 0.6181 | - | promote:9,rollback:1 | 9/9 | 326.8 |
| B | full | 42 | 8 | 497.0132 | 1086.1384 | - | promote:5,retain:3 | 5/7 | 289.6 |
| C | frozen | 42 | 8 | 50.3731 | 72.6052 | - | retain:8 | - | 35.0 |
| C | frozen | 43 | 8 | 50.3731 | 72.6052 | - | retain:8 | - | 57.0 |
| C | recenter | 42 | 8 | 50.9446 | 72.6044 | - | retain:8 | - | 48.2 |
| C | recenter | 43 | 8 | 50.9446 | 72.6044 | - | retain:8 | - | 28.5 |
| C | adapt | 42 | 8 | 51.0819 | 70.7596 | - | retain:8 | 5/7 | 159.5 |
| C | adapt | 43 | 8 | 51.0277 | 71.3957 | - | retain:8 | 2/7 | 158.7 |
| B | full | 43 | 8 | 502.0124 | 1065.0753 | - | promote:6,retain:2 | 5/7 | 395.8 |

## L2 — workspace-v2 batch path (fake host, real entrypoint)

### A — PASS

| round | batch | hidden | history rows | prediction rows | metric |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 445 | 445 | macro_f1=0.9558 |
| 2 | 2 | 0 | 890 | 1244 | macro_f1=0.7140 |
| 3 | 3 | 0 | 2134 | 1586 | macro_f1=0.5939 |
| 4 | 4 | 0 | 3720 | 161 | macro_f1=0.5602 |
| 5 | 5 | 0 | 3881 | 197 | macro_f1=0.6255 |
| 6 | 6 | 0 | 4078 | 2300 | macro_f1=0.2757 |
| 7 | 7 | 0 | 6378 | 3613 | macro_f1=0.4498 |
| 8 | 8 | 0 | 9991 | 294 | macro_f1=0.4105 |
| 9 | 9 | 0 | 10285 | 470 | macro_f1=0.5260 |
| 10 | 10 | 0 | 10755 | 3600 | macro_f1=0.3940 |

### B — PASS

| round | batch | hidden | history rows | prediction rows | metric |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 7032 | 167 | rmse=348.4432 |
| 2 | 1 | 0 | 7199 | 167 | rmse=1252.5771 |
| 3 | 2 | 0 | 7366 | 167 | rmse=1005.9889 |
| 4 | 3 | 0 | 7533 | 167 | rmse=883.5035 |
| 5 | 4 | 0 | 7700 | 167 | rmse=702.7532 |
| 6 | 5 | 0 | 7867 | 167 | rmse=490.7682 |
| 7 | 6 | 0 | 8034 | 167 | rmse=269.8946 |
| 8 | 7 | 0 | 8201 | 167 | rmse=246.2265 |
| 9 | 8 | 1 | 8368 | 167 | rmse=392.5515 |
| 10 | 9 | 1 | 8368 | 176 | rmse=389.0964 |

### C — PASS

| round | batch | hidden | history rows | prediction rows | metric |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 113089 | 2462 | rmse=37.0953 |
| 2 | 1 | 0 | 115551 | 2264 | rmse=42.1958 |
| 3 | 2 | 0 | 117815 | 2294 | rmse=41.0916 |
| 4 | 3 | 0 | 120109 | 2203 | rmse=42.6845 |
| 5 | 4 | 0 | 122312 | 2424 | rmse=52.0611 |
| 6 | 5 | 0 | 124736 | 2207 | rmse=43.9490 |
| 7 | 6 | 0 | 126943 | 2358 | rmse=70.8600 |
| 8 | 7 | 0 | 129301 | 2435 | rmse=59.8447 |
| 9 | 8 | 1 | 131736 | 2219 | rmse=57.9876 |
| 10 | 9 | 1 | 131736 | 2183 | rmse=56.4253 |

## Issues found and fixed

- **SIM-001** [数据/协议] A 全量流少 57 行（13,853/13,910），prepare 审计 reveal 439/445。  
  fix: 新增 _shuffle(rows,seed)；label_fraction>=1.0 时 reveal 直接取全部行。  
  evidence: scripts/sim/prepare_full_batches.py PASS：A 13,910 行、B 1,679（1,336+343）、C 23,049（18,647+4,402）。  
  status: fixed
- **SIM-002** [测试工具/v2 协议] L2 报 RuntimeError: predict returned 3 rows for 445，无法对整批打分。  
  fix: scripts/sim/run_l2_batches.py 新增 _predict_all()：全量喂 predict 并逐行校验官方输出格式。  
  evidence: L2 A/B/C 各 10 轮全部 PASS（A macro_f1 逐批 0.956→0.394；B/C 见报告）。  
  status: fixed
- **SIM-003** [v2 适配层] A 的 v2 提交候选只有 recenter/shrink/续训，延迟标签 pair_threshold 机制完全没被使用。  
  fix: 新增 pair_threshold 候选 + fit_pair_threshold(history train, pair=(0,3))，None 时回退 none。  
  evidence: batch5 历史 362 条 AA 时 fit delta=-3.8914，历史 dev primary 0.7842→0.8481；L2 A 10 轮使用该候选。  
  status: fixed
- **SIM-004** [预算语义/kernel] L3 的 kernel 批次流每个 A 批次仅 19 秒、无真实 LLM 调用，疑似静默降级。  
  fix: TokenBudget: cap<=0 表示 unlimited（remaining=1e18、exhausted=False）；修复后 L3 重跑。  
  evidence: 单测：cap=0 exhausted=False；cap=100 用 150 时 exhausted=True。  
  status: fixed
- **SIM-005** [稳定性/长流] A 20 批长流 full lane 在 243s 被 SIGKILL（rc=-9），未写 stream_report.json。  
  fix: 单独串行重跑该 lane（lane-timeout 7200）；后续长流任务单独作业、限制并行=1。  
  evidence: 待重跑结果。  
  status: fixed-partial
- **SIM-006** [观察/选择偏差] A 第一批 selection/report 指标 0.99–1.0，明显高于全量验证。  
  fix: 不作为 bug 修改（符合“只用已揭示历史”约束）；在报告中标注早期批次指标不可与 hidden 指标直接比较，并依赖后续批次稳定化。  
  evidence: L1 A-full 前 2 批 dev 1.0/.99，hidden（后续 UCI 批）仍保持 .62–.77。  
  status: documented
