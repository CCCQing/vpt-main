# B3 isolated experiment entry points

`B3` is a deterministic object-validation suite. It does not enable sampling,
KL, Graph-GP, slot residuals, or sample gates. Existing A/B configs keep their
legacy behavior because every new switch defaults to off.

## 1. Generate the three locked pseudo-GZSL splits

```bash
python -m src.tools.search_plans.b_series.generate_b3_class_disjoint_manifests \
  --res101-path /path/to/xlsa17/data/CUB/res101.mat \
  --split-path /path/to/xlsa17/data/CUB/att_splits.mat \
  --output-dir output/b3_series/manifests
```

The suite file stores relative manifest paths and is portable between the local
workspace and server. Every training checkpoint records the selected manifest
SHA-256; a parent checkpoint from another pseudo split is rejected.

## 2. Run P0 and R1 as separate gated stages

```bash
python -m src.tools.search_plans.b_series.run_b3_series_training \
  --stages P0 \
  --manifest-suite output/b3_series/manifests/b3_pseudo_split_suite.json \
  --out-root output/b3_series/training \
  --gpu-groups '0;1;2' --max-workers 3

python -m src.tools.search_plans.b_series.run_b3_series_training \
  --stages R1 \
  --manifest-suite output/b3_series/manifests/b3_pseudo_split_suite.json \
  --out-root output/b3_series/training \
  --max-ratios 0.25,0.50 \
  --gpu-groups '0;1;2' --max-workers 3
```

Formal stages always use training seeds `0/1/2`. Each training run inherits the
common strict three-Probe contract (`424242/424243/424244`). Existing non-empty
incomplete output is never overwritten.

### 2.1 Long fixed-Probe runs: train once, then replay three selections

For a large B3 queue, prefer the deferred runner.  It saves the same final
trainable checkpoint, writes `training_checkpoint_ready.json`, and runs each
selection seed as an independent checkpoint-only process.  A run is not
scientifically complete merely because training returned successfully: all
three replay validators must pass and `b3_fixed_probe_collection.json` must be
valid.

The first production use must be gated by one same-checkpoint equivalence run:

```bash
python -m src.tools.search_plans.a_series.replay_probe_robustness \
  --source-run <completed-reference-run> \
  --output-run <equivalence-replay-run> \
  --selection-seed 424242 --source-kind completed_probe \
  --execution-profile final_full --cpu-threads 8 \
  --cache-transformed-images --cache-vit-cls-prepass

python -m src.tools.validate_fixed_probe_replay_equivalence \
  --reference-run <completed-reference-run> \
  --replay-run <equivalence-replay-run> \
  --output <equivalence-summary.json>
```

Only after that summary reports `valid=true` may the resumable R1 queue start:

```bash
python -m src.tools.search_plans.b_series.run_b3_deferred_probe_queue \
  --stages R1 \
  --manifest-suite output/b3_series/manifests/b3_pseudo_split_suite.json \
  --out-root output/b3_series/training --max-ratios 0.25,0.50 \
  --gpu-groups '0;3;4;5;6;7' --max-workers 6 \
  --probe-cpu-threads 8 \
  --equivalence-summary <equivalence-summary.json>
```

The image cache stores only deterministic transformed CPU tensors.  The ViT
cache stores only the frozen, no-Prompt CLS prepass for repeated forwards of
the same batch.  Prompt-conditioned CLS, Attention, relevance and logits are
always recomputed for every condition.  Partial integrated Probe artifacts are
preserved but explicitly excluded when a validated deferred collection
supersedes them.

## 3. Calibrate, then freeze, the R2 loss weights

The pilot is explicitly non-formal: first pseudo split, training seed 0, and one
already selected R1 residual ratio.

```bash
python -m src.tools.search_plans.b_series.run_b3_series_training \
  --stages R2 --max-ratios 0.25 \
  --r2-pilot-weights 0.01:0.002,0.05:0.01,0.10:0.02 \
  --manifest-suite output/b3_series/manifests/b3_pseudo_split_suite.json \
  --out-root output/b3_series/training \
  --gpu-groups '0;1;2' --max-workers 3
```

After selecting one pair without consulting pseudo-unseen final results, run
formal R2 on all split/training-seed pairs:

```bash
python -m src.tools.search_plans.b_series.run_b3_series_training \
  --stages R2 --max-ratios 0.25 \
  --formal-intra-weight 0.05 --formal-inter-weight 0.01 \
  --manifest-suite output/b3_series/manifests/b3_pseudo_split_suite.json \
  --out-root output/b3_series/training \
  --gpu-groups '0;1;2' --max-workers 3
```

`B3-R1A`, `B3-R2B`, and `B3-R3F` are aliases for the paired P0, R1I, and
pre-unfreeze R2I checkpoints. They are reused controls, not duplicate training.

## 4. Run B3-D1 donor aggregation with strict three Probes

Run this once per pseudo-split directory and list the desired B3 methods. The
replay automatically raises Probe support to six samples per available class,
so single/K2/K4/leave-one-out donors are all sampled without replacement.

```bash
python -m src.tools.search_plans.b_series.run_b_series_replays \
  --source-root output/b3_series/training/pseudo_seed31001 \
  --output-root output/b3_series/replays/pseudo_seed31001 \
  --run B3-R1I-R025:0 --run B3-R1I-R025:1 --run B3-R1I-R025:2 \
  --experiments B3D1 --scope probe \
  --selection-seeds 424242,424243,424244 \
  --gpu-groups '0;1;2' --max-workers 3
```

The replay writes the exact donor manifest, `shared_mu` class geometry, task and
calibration effects, and explicit same-single/K2/K4/LOO/different conditions.

## 5. Conditional R3 and locked final validation

R3 is run only after R2 passes. Pass the same frozen R2 weights used formally.
For final-GZSL validation, use `--protocol final_gzsl --a2-root <existing-A2-root>`;
the runner then initializes R1/R3A from the matched existing A2 checkpoint and
still refuses protocol or seed mismatch.

## 6. Validation

```bash
python -m src.tools.validate_b3_protocol
python -m src.tools.validate_deep_prompt_residual
python -m src.tools.validate_logit_geometry

for cfg in configs/b_series_experiments/B3-*.yaml; do
  python -m src.tools.search_plans.a_series.audit_baseline_config \
    "$cfg" --allow-b3-template --build-model
done
```

`--allow-b3-template` only permits the reusable YAML to leave the pseudo-split
manifest blank. It does not relax training: `run_b3_series_training.py` still
injects a locked manifest into every job, and the dataset/checkpoint code rejects
a missing or mismatched manifest.
