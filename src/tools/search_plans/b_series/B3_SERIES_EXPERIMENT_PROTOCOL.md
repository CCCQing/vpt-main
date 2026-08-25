# B3 normal Seen/Unseen experiment entry points

`B3` is currently a deterministic object-validation suite on the standard CUB
GZSL split. R1-R3 do not enable sampling, KL, Graph-GP, slot residuals, or sample
gates. T1/T2 are isolated follow-up branches that change only the residual's
slot parameterization. Standard `test_unseen` is used during development, so every B3 report must
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

## 6. D2-D4 checkpoint-only follow-up

`D2` compares the frozen prepass CLS with trained `mu` and at least five
random-untrained statistics-MLP controls. `D3` pairs normal and Residual-zero
for Prompt slot mass/allocation and raw/LN/K/V propagation, then applies the
common/role/role-permuted counterfactuals. `D4` is explicitly a true-label
oracle: same-class LOO, global LOO, and at least three deranged wrong-class
centers. D4 must never be reported as a deployable test-time method.

```bash
python -m src.tools.search_plans.b_series.replay_b3_followup_diagnostics \
  --source-run <B3-R1I run directory> \
  --output-dir <empty diagnostic directory> \
  --experiments D2,D3,D4 --scope full \
  --random-mlp-seeds 23001,23002,23003,23004,23005 \
  --wrong-class-seeds 24001,24002,24003 \
  --selected-layers 8,9,10,11
```

For strict Probe evidence, repeat the command independently with `--scope
probe` and selection seeds `424242/424243/424244`. D2 additionally includes the
deterministic train-seen evaluation loader when `--scope full` is used.

### 6.1 D2-G source-to-decision geometry completion

`D2G` is a checkpoint-only extension of D2. It does not stop at the absolute
geometry of prepass CLS, trained `mu`, and five random-untrained `mu` controls.
On the same sample manifest it also pairs normal with Residual-zero and records
the complete chain:

```text
prepass CLS -> trained mu -> final CLS -> projected semantic prototypes
            -> centered/direction-normalized/class-pattern logits
            -> Seen/Unseen/H/AUSUC
```

The implementation reuses `analyze_vector_geometry`,
`visual_semantic_alignment_metrics`, `semantic_visual_graph_metrics`, and
`analyze_logit_geometry`. The prepass CLS is cached once per batch and reused by
normal and Residual-zero. Raw scatter values are not subtracted across spaces
with different dimensions; cross-space interpretation uses standardized
metrics, paired directions, and relationship measures.

```bash
python -m src.tools.search_plans.b_series.replay_b3_followup_diagnostics \
  --source-run <B3-R1I run directory> \
  --output-dir <empty D2G directory> \
  --experiments D2G --scope full \
  --random-mlp-seeds 23001,23002,23003,23004,23005

python -m src.tools.validate_b3_d2g_result \
  --result-dir <completed D2G directory>
```

The formal matrix remains two R1I methods, training seeds `0/1/2`, one full
scope, and strict Probe selection seeds `424242/424243/424244`: 24 cells in
total. Probe selections are nested robustness checks, not independent training
seeds. A cell cannot be called complete unless source/checkpoint hashes,
sample/candidate identity, class and LOO support, cached-prepass equivalence,
semantic-reference validity, logit reconstruction/argmax equivalence, and task
result validity all pass. The single-cell validator and the 24-cell aggregate
validator enforce this boundary; missing values are never filled with zero.

## 7. T1/T2 slot-capacity training branches

T1 uses `slot_scalar`: the source is the normalized frozen-ViT prepass CLS,
without the statistics MLP. The same sample-level direction is retained, but
every Prompt slot receives a learned sample-conditional coefficient. It
initializes exactly as the direct shared S0 residual, so T1 changes only the
carrier capacity. The matched control uses one deterministic zero-mean fixed
direction and identical coefficient heads; it therefore measures added
slot-scalar capacity without image conditioning. The old all-ones direct
control is not used here because token-wise LayerNorm removes that direction.

T2 remains a separate rank-2 or rank-4 slot-specific-basis branch and must not
be mixed into the first T1 screening decision. Both branches load matched A2
checkpoints, freeze static Prompt and the classifier, and isolate layers 8-11.

The first T1 decision is a training-seed-0 screening pair only:

```bash
python -m src.tools.search_plans.b_series.run_b3_series_training \
  --stages T1 --protocol final_gzsl --screening-seed 0 \
  --a2-root <A-series root> --out-root <T1 screening root> \
  --max-ratios 0.25 --gpu-groups '0;4' --max-workers 2
```

Do not schedule an additional MLP-source T1 as an equal main branch. It changes
both source encoding and carrier capacity. Only after direct-source T1 shows a
useful signal may an MLP-source version be added as a secondary source-ablation.

```bash
python -m src.tools.search_plans.b_series.run_b3_series_training \
  --stages T1,T2 --protocol final_gzsl \
  --a2-root <A-series root> --out-root <B3 training root> \
  --max-ratios 0.25 --gpu-groups '0;3;4;5;6;7' --max-workers 6
```

The command above is the later formal T1/T2 matrix and schedules strict
training seeds `0/1/2`; fixed-Probe collection must still use
`424242/424243/424244` before either branch is called complete.
