# B3 normal Seen/Unseen experiment entry points

`B3` is currently a deterministic object-validation suite on the standard CUB
GZSL split. It does not enable sampling, KL, Graph-GP, slot residuals, or sample
gates. Standard `test_unseen` is used during development, so every B3 report must
label it as normal-Unseen development evidence rather than an untouched final
confirmation set.

## 1. Matched initialization

R1 loads the corresponding A2 checkpoint with the same training seed. Static
Prompt and the classifier stay frozen; only the residual statistics MLP and its
bounded layer amplitude are trained. Use the A2 training root, not a fixed-Probe
replay directory.

## 2. Run R1 with strict training seeds and three Probes

The deferred runner trains seeds `0/1/2` once, then replays selection seeds
`424242/424243/424244` independently from each final checkpoint.

```bash
python -m src.tools.search_plans.b_series.run_b3_deferred_probe_queue \
  --stages R1 --protocol final_gzsl \
  --a2-root output/baseline_rebuild_tok16_lr6e-4_ep15/baseline_rebuild_tok16_lr6e-4_ep15 \
  --out-root output/b3_series_final/training \
  --max-ratios 0.25,0.50 \
  --gpu-groups '0;3;4;5;6;7' --max-workers 6 \
  --probe-cpu-threads 8 \
  --equivalence-summary <equivalence-summary.json>
```

The first production use of a new fixed-Probe implementation still requires a
passing same-checkpoint equivalence summary. `--skip-equivalence-gate` is only
allowed for an explicitly labelled smoke run.

## 3. Fill the two checkpoint-only evidence gaps

`B3EVIDENCE` performs a normal forward, a true effective `Prompt-zero`, and
shared-`mu` class geometry. Effective `Prompt-zero` clears frozen static Prompt
content and forces the applied Deep Prompt residual to zero while retaining
Prompt slots and Attention routes.

Run the full standard Seen/Unseen sets once per checkpoint:

```bash
python -m src.tools.search_plans.b_series.run_b_series_replays \
  --source-root output/b3_series_final/training/final_gzsl \
  --output-root output/b3_series_final/evidence/full \
  --run B3-R1I-R025:0 --run B3-R1I-R025:1 --run B3-R1I-R025:2 \
  --run B3-R1I-R050:0 --run B3-R1I-R050:1 --run B3-R1I-R050:2 \
  --run B3-R1N-R025:0 --run B3-R1N-R025:1 --run B3-R1N-R025:2 \
  --run B3-R1N-R050:0 --run B3-R1N-R050:1 --run B3-R1N-R050:2 \
  --experiments B3EVIDENCE --scope full \
  --gpu-groups '0;3;4;5;6;7' --max-workers 6
```

Then run the same operation under the three strict Probe selections:

```bash
python -m src.tools.search_plans.b_series.run_b_series_replays \
  --source-root output/b3_series_final/training/final_gzsl \
  --output-root output/b3_series_final/evidence/probe \
  --run B3-R1I-R025:0 --run B3-R1I-R025:1 --run B3-R1I-R025:2 \
  --run B3-R1I-R050:0 --run B3-R1I-R050:1 --run B3-R1I-R050:2 \
  --run B3-R1N-R025:0 --run B3-R1N-R025:1 --run B3-R1N-R025:2 \
  --run B3-R1N-R050:0 --run B3-R1N-R050:1 --run B3-R1N-R050:2 \
  --experiments B3EVIDENCE --scope probe \
  --selection-seeds 424242,424243,424244 \
  --gpu-groups '0;3;4;5;6;7' --max-workers 6
```

## 4. Scientific completion conditions

A B3 method is not complete until all of the following hold:

- seeds `0/1/2` have valid final checkpoints;
- each checkpoint has valid Probes `424242/424243/424244`;
- `Prompt-zero` reports frozen static Prompt parameters and at least one residual
  module as applied targets;
- Prompt-zero residual traces are exactly zero and the paired sample identity is
  unchanged;
- shared-`mu` geometry is present for normal Seen and normal Unseen classes;
- the report states that normal Unseen participated in development decisions.

## 5. Validation

```bash
python -m src.tools.validate_b3_protocol
python -m src.tools.validate_deep_prompt_residual
python -m src.tools.validate_prompt_monitoring_extensions
python -m src.tools.validate_logit_geometry

for cfg in configs/b_series_experiments/B3-*.yaml; do
  python -m src.tools.search_plans.a_series.audit_baseline_config "$cfg" --build-model
done
```
