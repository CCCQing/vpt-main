# B系列补充实验执行协议

## 1. 命名与历史身份

本轮新增内容直接归入 **B系列实验**，原子实验只使用 `E1` 至 `E7`，不再创建额外的B1套件名。历史条件方法仍保持原始 `B1`、`B2` 身份；新产物中也直接使用 `B1`与`B2`作为方法列，不重命名旧目录或旧权重。

## 2. 实验与代码入口

| 实验 | 是否重训 | 入口 | 默认执行条件 |
|---|---:|---|---|
| `E1` 全局 residual dose-response | 否 | `replay_b_series_experiments.py` | B1主对象，B2同条件控制 |
| `E2` residual—静态Prompt几何 | 否 | 同上 | 与normal、Residual-zero共享前向 |
| `E3` grouped-depth zero/only与shortlist | 否 | 同上 | 先三组；shortlist层必须显式传入 |
| `E4` same/different-class swap | 否 | 同上 | 保存确定性donor manifest |
| `E5` 静态Prompt冻结对照 | 是 | `run_b_series_training.py --groups E5` | 每个训练seed从同seed A2 checkpoint初始化 |
| `E6` Slot-aware确定性residual | 是 | `run_b_series_training.py --groups E6` | rank-8首轮；B1/B2同构控制 |
| `E7` 样本级gate | 是 | 同上，`--groups E7-G1`起步 | 必须先通过外部precondition manifest |

`shared`仍使用同一`mu(x)`复制到16个槽位；`slot_low_rank`使用逐层低秩decoder生成不同槽位内容。样本gate默认关闭；开启时仅乘在最终residual上，不改变`mu/logvar`生成接口。所有新增概率损失、采样、KL和Graph-GP仍保持关闭。

## 3. E1—E4回放

完整测试集先对三个训练seed执行B1/B2：

```bash
python src/tools/search_plans/b_series/run_b_series_replays.py \
  --source-root output/prompt_distribution_b_series_round1 \
  --output-root output/b_series_experiments/replay/full \
  --scope full \
  --run B1:0 --run B1:1 --run B1:2 \
  --run B2:0 --run B2:1 --run B2:2 \
  --gpu-groups '0;1;2;3' --max-workers 4
```

严格三Probe使用同一入口，不把Probe种子计作独立训练：

```bash
python src/tools/search_plans/b_series/run_b_series_replays.py \
  --source-root output/prompt_distribution_b_series_round1 \
  --output-root output/b_series_experiments/replay/probe \
  --scope probe --selection-seeds 424242,424243,424244 \
  --run B1:0 --run B1:1 --run B1:2 \
  --run B2:0 --run B2:1 --run B2:2 \
  --gpu-groups '0;1;2;3' --max-workers 4
```

回放完成后必须通过原子cell完整性门，并按嵌套口径汇总：

```bash
python -m src.tools.search_plans.b_series.summarize_b_series_replays \
  --root output/b_series_experiments/replay/probe \
  --scope probe --methods B1,B2 \
  --training-seeds 0,1,2 \
  --selection-seeds 424242,424243,424244
```

该汇总先对同一checkpoint的三套Probe选样计算均值与范围，再把三个训练seed作为独立单位汇总；不得把九个run-probe cell写成`n=9`。只要有一个预期cell缺失、无效或身份不符，汇总器就以非零状态退出，报告不得写“完整”。

首轮不传`--shortlist-layers`。只有深度组结果满足预声明条件后，才用例如`--experiments E3 --shortlist-layers 9,10,11`展开逐层zero/only/dose。回放器默认只保存聚合结果与E2紧凑数组，不保存全量logits；只有审计需要时才传`--save-logits`。

## 4. E5—E6重训

E5—E7 读取的是同种子的 A2 `vpt_trainable` 权重，不读取历史优化器或调度器状态。所有配对组都从同一权重起点重新建立相同的优化器和调度器，因此它们是“权重续接、优化过程重新开始”的受控比较，不应写成原训练过程的无缝续跑。`initialization_checkpoint_manifest.json` 必须记录 `optimizer_state_reused=false`、`scheduler_state_reused=false` 和续接策略。

`--a2-root`必须能解析到每个训练seed的A2 `model_final_trainable.pth`。程序在optimizer创建前加载这份trainable-only checkpoint，并写出初始化checkpoint哈希、seed、实际加载张量与新增参数清单。

```bash
python src/tools/search_plans/b_series/run_b_series_training.py \
  --groups E5,E6 --seeds 0,1,2 \
  --a2-root output/baseline_rebuild_tok16_lr6e-4_ep15 \
  --out-root output/b_series_experiments/training \
  --gpu-groups '0;1;2;3' --max-workers 4
```

E5 freeze组同时冻结静态input/deep Prompt与`RSimilarityClassifier.prototype_proj`。训练开始和结束分别生成`residual_freeze_contract_initial.json`、`residual_freeze_contract_final.json`；任何冻结参数进入optimizer、产生梯度或发生非零位移，实验直接失败，不能登记为有效freeze对照。

E6的`E6-B1-slot`与`E6-B2-slot-control`使用完全同构的rank-8 decoder。shared控制与E5 joint-matched结构相同；若两组在同一次套件中重复请求，launcher按方法与seed去重。

## 5. E7条件入口

E7默认不能启动。先复制`E7-preconditions.example.json`并用E1—E6结果填写证据来源；三个checks必须全为`true`，`evidence_paths`不能为空且每个文件必须真实存在。G1只输出每样本一个12层共享gate；G2和G3还分别要求manifest声明`max_gate_stage=G2/G3`。每一级都会同时运行同起点的G0无样本gate、固定gate输入控制和B2非条件容量控制，不能只报告复杂gate本身。

```bash
python src/tools/search_plans/b_series/run_b_series_training.py \
  --groups E7-G1 --seeds 0,1,2 \
  --a2-root output/baseline_rebuild_tok16_lr6e-4_ep15 \
  --precondition-manifest path/to/E7-preconditions.json \
  --out-root output/b_series_experiments/training \
  --gpu-groups '0;1;2' --max-workers 3
```

若E6已通过并决定在Slot-aware承载上加gate，增加`--e7-base-mode slot`；否则保持默认`shared`。不能在结果未知时把G1、G2、G3一次性全跑。

## 6. 有效性与产物

- E1—E4：记录source checkpoint SHA-256、训练seed、scope、Probe selection seed、sample顺序、condition、命中层、paired结果和donor manifest。
- E1端点：`alpha=0`复用`all_residual_zero`，`alpha=1`复用normal，避免重复前向。
- E2分组：名称明确说明是“Residual-zero相对normal”的correctness转移，不用margin符号冒充对错转移。
- E3：group-zero与group-only成对登记，三组效应不默认可加。
- E4：same-class必须排除self，different-class必须跨类；合同不满足时停止。
- E5—E7：每个训练seed是独立训练单位；三套Probe仍是同checkpoint的关联选样。
- 所有运行目录沿用不覆盖策略；存在非空未完成目录时必须换新目录或先人工审计，launcher不会自动删除。
