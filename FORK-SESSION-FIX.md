# 修复：fork 出来的会话无法正常对话

## 结论（一句话）

DSH 创建 fork 会话时**没有在 session header 里写入 `parentSession`**，导致 billion-context（ACP 压缩插件）的 "fork adoption" 被跳过 —— 被 fork 的会话拿不到父会话的压缩账本，于是把父会话的**全部原始历史**（21.9 MiB / 约 575 万 tokens）塞进第一次请求，远超模型窗口 750000；ACP 只能现场做 preflight 压缩，压不完（>70 s），重试时被上游以 `413 preflight_compress_failed` 直接拒绝。表现就是"fork 出来的会话发消息不回、卡住、然后报错"。

不是日志损坏，也不是 fork 机制本身崩了（fork 出来的 21904 行事件全部可解析）。

## 证据链

1. fork 会话 header（`~/.dsh/sessions/--home-xgx-wpp1-zhixun-harness--/session-0c7657e8-5d39-45f5-bcca-a97b1fcbee5c/session.v4.jsonl.zstd` 第 1 行）：

   ```json
   {"type":"session","version":4,"id":"session-0c7657e8-…","cwd":"/home/xgx/wpp1/zhixun-harness","isSeeded":true,"delegationDepth":0,"agentPreset":"standard"}
   ```

   `isSeeded:true`，但**没有 `parentSession`**。而 `SessionHeader` 类型里这个字段是存在的（`readonly parentSession?: SessionId`），`dsh-subagent` 创建子会话时也会写它，只有 `fork()` 这条路径漏了。

2. ACP 日志 `~/.local/state/billion-context/bili.log:18344`（2026-11-19 10:35:17）：

   ```
   [warn] bili-native-dsh: fork adoption for session-0c7657e8-… skipped —
   the session header says isSeeded but carries no usable parentSession;
   the child starts without inherited compression state (#4215)
   ```

   ACP 判定条件就是 `session.header.parentSession` 必须是字符串（见 `billion-context/dist/agent/dsh-native.js`）。跳过在线继承后它退回模糊匹配 `resume-inheritance`，还**错配到了另一个会话**：

   ```
   :18346 [resume-inheritance] miss: no byte-exact resume among 47 eligible chain(s);
          closest session-e4668229-… first divergence … cells 0..3638 uninspectable
   :18349 [resume-inheritance] … weakly resumed closest chain after byte-exact miss
          session-e4668229-…: inherited 5389 ref(s), 5 block(s)
   ```

3. 于是 fork 的第一个请求变成全量原文（`bili.log:18465`，10:36:26）：

   ```
   request: 6668 msgs, inbound=21.9MiB, local=68566ms
   context 3325471 tokens reached preflight target 750000 … preflight compressing before forward
   [preflight] payload ~3599034 tok vs window 750000 (~4.8x) — raising summarization budget 16 -> 32
   preflight still running after 30000ms grace — committed early SSE headers + keep-alive to hold the client
   [preflight] summarization aborted: client disconnected → preflight aborted, not forwarding
   ```

   这 68.5 秒里 model 一个字都没出（DSH 侧 `request/header` 之后 `assistant/attempt` 是空流），用户 10:36:26 点了停止 —— 与 `turn/end {"kind":"aborted","reason":{"kind":"user"}}` 完全对应。

4. 用户随后发的 "continue"（turn 103）先撞上 DSH 的 300 s 流空闲超时，重试时被上游拒绝（`session.v4.jsonl.zstd` seq 21900/21902）：

   ```
   pi-ai stream idle timeout after 300000ms  (code TIMEOUT)
   413: {"code":"preflight_compress_failed","message":"payload core text alone is ~11065086 bytes —
   … that is ≥ ~1475345 tokens vs the model window 750000 (model=deepseek-v4.1-flash)"}
   ```

5. 对照组：父会话（`session-5063f7d0-…`）在 10:35–10:36 期间每 25–30 s 正常完成一个 step，projcache 显示其请求约 24 万 tokens、几乎全命中 prompt cache —— 说明父会话一切正常，只是"压缩账本"没被 fork 继承。

## 已做的修复（均带备份）

1. **代码（根治）**：`dsh-api-session-controller/lib/index.js` 的 `fork()` 里补上父会话引用：

   ```diff
    meta: {
        ...source.header.cwd === void 0 ? {} : { cwd: source.header.cwd },
   +    parentSession: source.header.id,
        isSeeded: true,
        ...composition.agentPreset === void 0 ? {} : { agentPreset: composition.agentPreset }
    },
   ```

   - 文件：`/home/xgx/.nvm/versions/node/v22.22.0/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-api-session-controller/lib/index.js:796`
   - 备份：同目录 `index.js.bak-fork-parent-20261119`
   - 已用 `node --input-type=module -e "import('./lib/index.js')"` 验证模块可正常加载。

2. **数据（救回你已经 fork 出来的那个会话）**：给它补上 header 字段

   ```json
   "parentSession": "session-5063f7d0-a63d-45e3-87cc-887012fcf9fd"
   ```

   - 文件同上 `session-0c7657e8-…/session.v4.jsonl.zstd`（解压→改第 1 行→重新 zstd -19 打包，21904 行事件完整保留）
   - 备份：`session.v4.jsonl.zstd.bak-20261119`

## 生效步骤（需要你操作）

1. **重启 DSH web 服务**（代码补丁只有重启后才加载）。如果现在是从终端跑着 `dsh web` / `bili dsh`，Ctrl-C 后重新启动即可；本对话会随重启结束。
2. 重启后，直接在左侧会话列表里打开那个 fork 会话 `session-0c7657e8…`（标题 "官方提交方案训练平台[智训引 (1)"），随便发一句话。
   - 预期：ACP 日志里出现 `fork adoption … adopted`（或 `snapshot` 成功），请求体积回到约 24 万 tokens 量级，能正常流式回答。
   - 万一还报错：先删掉这个 fork 会话在 ACP 里的脏状态再试（它在上一次失败里继承过错误链）：
     `~/.dsh/profiles/web/node_modules/.bin/bili sessions rm session-0c7657e8-5d39-45f5-bcca-a97b1fcbee5c`
3. 以后再 fork 任何大会话，都会走同一条已修复的路径。

## 回滚

```bash
D=/home/xgx/.nvm/versions/node/v22.22.0/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-api-session-controller/lib
cp "$D/index.js.bak-fork-parent-20261119" "$D/index.js"
F=~/.dsh/sessions/--home-xgx-wpp1-zhixun-harness--/session-0c7657e8-5d39-45f5-bcca-a97b1fcbee5c/session.v4.jsonl.zstd
cp "$F.bak-20261119" "$F"
```

## 备注 / 教训

- 父会话已被 ACP 压到 ~24 万 tokens，但**原始事件日志**是 62.7 MB；fork 切点取的是父会话最后一个已完成 turn（turn 101 结束），这个语义本身没问题，问题只在"压缩状态没有随 fork 继承"。
- 在该问题下，任何"上下文很大"的会话被 fork 后都会复现（payload 与模型窗口无关地膨胀到原始历史大小）。
- 这也解释了为什么"新建一个会话"（也就是当前这个）一切正常：它没有继承任何历史。
