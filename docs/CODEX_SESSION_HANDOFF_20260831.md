# Codex 会话迁移交接

## 目的

本文件用于在不继承长对话历史的情况下继续 `p+v+s` 项目。新任务应先读取本文件和仓库根目录的 `AGENTS.md`，不要默认批量读取旧任务；只有需要核对具体证据时，才按需读取源任务的少量相关回合或原始报告。

## Git 身份

- 迁移前基础提交：`95e724ae8cdfdcac5653e82ed98588688cdfe7f8`
- 快照分支：`codex/session-migration-20260831`
- 本文件随快照提交保存；实际快照 SHA 以该分支 tip 和 `git rev-parse HEAD` 为准。
- 快照只包含 Git 可见的修改。`.env`、`output/`、`src/configs/local_path.yaml`、数据集、checkpoint 和其他忽略资产不在快照中。
- 迁移前工作树为 detached HEAD，包含监测扩展、A/B 系列工具整理、GraphProbPrior 相关清理、公共队列/产物工具和已经批准的死代码删除。

## 已确认的实验结论

### A 系列

- A2 三训练 seed 平均 `H=0.4778`，相对 A0 平均提升 `0.0640`，主要来自 Unseen。
- A2 的收益主要表现为决策空间中的域内尺度变化，不能写成全局类别几何全面改善。
- 原 A2 checkpoint 上 posthoc cosine 的 `H=0.5186` 是正证据；让 cosine 参与 Prompt 与语义投影联合重训练的 winner 仅 `H=0.1946`，该重训练路线停止扩大。

### B 系列

- P0-2 不支持继续 A2 Prompt basis routing。
- P0-3 只支持 312 维稀疏局部语义关系，不支持稠密 GraphProbPrior 资格。
- P1-3a head-only compatibility 未稳定超过 dot 或简单 posthoc cosine，因此 P1-3b 的启动条件未满足。
- 不把“分类接口更复杂”当作已经成立的解决方向；简单 posthoc cosine 仍说明 A2 CLS 中存在可读取信息。

### C 系列

- C1-C4 共 12 组正式训练已完成并验收。
- 历史 `H=0.5681` 未复现；C1 三 seed 平均 `H=0.4891`。
- AM 关闭时，GPP 对 H 的影响为 3/3 seed 负向，平均 `-1.50` 个百分点；AM 开启时 GPP 跨 seed 变号。
- C5-C7 旧 GPP 组件搜索停止。若要解释失败原因，应优先做现有 checkpoint 的诊断，不先重训练。
- 历史无显式 seed、固定 seed 0/1/2 和 2026-06-29 备份快照复跑均已完成，不要重复执行。

## 监测与报告合同

- 区分 configured intent、runtime effective 和实际 observed artifact；配置开启不等于产物存在。
- 缺失证据写 `missing` 或 `insufficient_evidence`，不得补零。
- 训练 seed 是正式独立统计单位；Probe selection seed 不能冒充训练重复。
- 报告只有在证据 profile、coverage manifest 和机器门禁通过后才能声称完整。
- 当前文献审计的新主线是验证 `Prompt 参数空间 -> Prompt 功能空间 -> 分类结果` 的传播链，而不是继续增加普通 Attention 均值。

## 服务器队列

- 源任务：`019fdf41-7552-7351-b625-28164f0b00bd`
- 最后已知队列：服务器 `tmux` 会话 `c_chat_replay_0831`
- 队列日志：`/data/RuanZhaoQi/vpt-main-pvs-c-gpp-t0009-audit/output/c_gpp_t0009_chat_recovered_20260831/queue_status.log`
- 队列设计为只运行一组无显式 seed 的历史 `t0009` 复现；等待两张卡各有约 20GB 空闲显存、利用率不高于 20%，并连续两次满足条件。
- 以上仅是迁移时的最后已知状态。任何“仍在等待、已经启动或已经完成”的说法都必须先实时检查服务器进程、GPU、日志和产物。
- 未经用户确认，不启动新的服务器长任务，也不重复已经完成的实验。

## 新任务的读取与操作边界

1. 先运行 `git status --short --branch` 和 `git rev-parse HEAD`，确认位于本快照。
2. 先读本文件；不要开场就读取整个源任务或批量扫描所有报告。
3. 需要精确实验数字时，优先读取对应 A/B/C 报告，再按需读取源任务的特定回合。
4. 不覆盖用户修改，不清理历史产物，不推送远端，不启动服务器实验，除非用户明确要求。
5. 发现忽略的运行资产缺失时先报告，不把代码快照完整误写成运行环境完整。
