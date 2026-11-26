# 公开开发证据

作者：Arthur；日期：2026-11-27。

- `c_conservative_review.json`：官方父模型、完整10epoch更新、两个固定参数插值点的完整公开比较。
- `b_continuation_review.json`：真实父链15epoch、逐轴承结果和两个未采用的固定插值点。
- `c_runtime_verification.json`：SDK本地宿主十轮，逐轮重新加载checkpoint在完整验证集复测，并检查原生状态累计与缓存动作。
- `ab_runtime_verification.json`：A两轮、B三轮实际入口记录。A的fake gateway不提供可执行工具决策，实际发生kernel到pipeline回退；不证明真实自主kernel链已验证。
- `resource_benchmark_current.json`：旧后置retain、当前提前retain、当前保守C三路径，串行、单次、同机同线程。

所有训练只使用公开训练投影；公开验证参与测量/选择，故不构成独立确认性检验。分组通过不等于隐藏漂移阶段通过。
数据摘要、模型摘要与源码摘要按当次运行记录；早期单项探索和最终整合入口可能对应不同源码摘要，不能据此把早期探索描述成最终镜像验收。
本目录不包含模型权重或本机运行路径。CPU为本机观测，fake gateway调用不等于真实服务费用，未产生新的平台分数。
