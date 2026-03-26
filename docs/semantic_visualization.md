# Semantic Visualization Pipeline (LateSemanticSideBranch)

This project includes an optional visualization pipeline for the prompt + semantic-token ViT model.
It is disabled by default and does not change training/evaluation behavior unless enabled.

## Enable

Use config overrides:

- `SOLVER.VIS.ENABLE True`
- `SOLVER.VIS.EPOCH_LIST [1,2,3,5,8,10,15,20,30,40,50]` (recommended for 50-epoch runs)
- `SOLVER.VIS.SPLITS ['val','test']`
- `SOLVER.VIS.MAX_SAMPLES 8`
- `SOLVER.VIS.LOCAL_CONTROL True`
- `SOLVER.VIS.ROLLOUT True`
- `SOLVER.VIS.GT_HN_COMPARE True`
- `SOLVER.VIS.TRENDS True`

Rollout needs attention weights, so when visualization is enabled the trainer forces affinity `vis=True`.

Trigger priority:

- If `SOLVER.VIS.EPOCH_LIST` is non-empty, visualization runs only on those 1-based epochs.
- Otherwise, fallback to periodic trigger by `SOLVER.VIS.EVERY_EPOCH`.

## Output directory

All outputs are saved under:

- `<OUTPUT_DIR>/visualization/`

Structure:

- `visualization/<split>/epoch_XXX/`:
  - local semantic-token maps (`*_local_tokenK.png`, `*_local_panel.png`, `*_local_maps.npz`)
  - rollout overlays (`*_rollout_cls.png`, `*_rollout_shallow_prompt.png`, `*_rollout_deep_prompt.png`)
  - GT vs hardest-negative comparison (`*_gt_hn_compare.png`, `*_gt_hn_maps.npz`)
- `visualization/trends/<split>/`:
  - per-epoch trend json (`epoch_XXX_trend.json`)
  - trend csv (`trend_summary.csv`)
  - per-epoch entropy curves (`epoch_XXX_entropy_curve.png`)

## What each figure means

- Local control maps:
  - one Avs-based heatmap per semantic token (e.g., 4 tokens -> 4 panels).
- Rollout maps:
  - CLS attention rollout and prompt-token rollouts over the backbone.
- GT vs HN:
  - patch-level semantic responses for GT class vs hardest-negative class and their difference.
- Trends:
  - layer-wise Avs/Aps entropy, token specialization, token pairwise cosine, GT-HN score gap.
  - token norm statistics (`mean/std/max/p95/p99/outlier_ratio`) on visual patch tokens.
  - semantic-token utilization concentration (`token_usage_gini`, `token_monopoly_index`), and when available, anchor/free split metrics.
