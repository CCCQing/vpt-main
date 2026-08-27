# A2-Cosine 正式基线适配协议

本轮把余弦读出从 checkpoint-only 诊断升级为正式重训练候选，但搜索结果仍属于开发证据。正常 `test-seen/test-unseen` 会参与超参数选择，因此不能再把同一结果写成未参与设计的独立最终测试。

## 顺序与原子问题

1. `01_anchor`：固定 `16 token / 6e-4 / 15 epoch / fixed scale=1`，训练 seed 0/1/2。它隔离“在原 A2 超参数下正式改用余弦接口”的效果。
2. `02_scale_screen`：仅在 seed0 比较 `fixed1 / fixed10 / learnable10`，确定归一化 logits 的训练温度。
3. `03_token_lr`：固定15 epoch和胜出 scale，搜索 `token×lr`。
4. `04_epoch_refine`：只对前三个 `token×lr` 组合搜索训练时长。
5. `05_shortlist`：只给前三名补齐训练 seed 1/2；seed0来自前述搜索阶段。

排序以最终 raw H 为主，AUSUC、Unseen依次作次级依据。若候选 AUSUC 低于 fixed1 seed0 超过预声明容差，优先从通过 guard 的组合中选择；若没有组合通过，必须显式报告 guard 全部失败，不能静默隐藏。

## 成本合同

搜索阶段不执行固定 Probe、target relevance、Attention、module-effect、milestone Probe和重型表示几何。仍保留数值有限性、optimizer首次更新、loss轨迹、最终Seen/Unseen/H/AUSUC、配置与checkpoint身份。胜者选出后才重新训练完整三seed A2-Cosine并执行全方位监测。

## 后续比较合同

新基线成立后，B系列必须至少形成配对的 `A2-Cosine / B1-Cosine / B2-Cosine`。B1第一轮沿用胜出A2-Cosine基础超参数；没有B2非条件容量控制时，不得把B1变化归因于图像条件 residual。

运行示例：

```bash
python src/tools/search_plans/a_series/run_a2_cosine_search.py \
  --gpu-groups '0;1;2;3' \
  --max-workers 4 \
  --out-root output/a2_cosine_baseline_search
```
