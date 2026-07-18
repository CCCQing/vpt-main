# Baseline rebuild: A series

This directory is the isolated configuration and audit record for the clean VPT-GZSL baseline rebuild.

| ID | Artifact | Purpose |
| --- | --- | --- |
| A-01 | `A-01-common-ce.yaml` | Shared CUB final-GZSL protocol, semantic matcher, and CE-only loss boundary |
| A-02 | `A-02-A0-frozen-vit-ce.yaml` | A0: frozen ViT with no visual prompt |
| A-03 | `A-03-A1-vpt-shallow-ce.yaml` | A1: VPT-Shallow with learned input prompts |
| A-04 | `A-04-A2-vpt-deep-ce.yaml` | A2: VPT-Deep with learned prompts at every layer |
| A-05 | `src/tools/A-audit_baseline_config.py` | Static and optional constructed-model cleanliness audit |
| A-06 | `dataset_manifest.json` in every run output | Resolved split, global class mapping, attribute and source-file checksums |
| A-07 | `resolved_config.yaml` in every run output | Complete merged configuration after base, local-path, and CLI overrides |
| A-08 | `trainable_parameters.json` in every run output | Actual trainable tensors after the classifier head is attached |
| A-09 | runtime monitoring artifacts in every run output | Step/epoch/event records plus manifest and observed-runtime summary |
| A-10 | `diagnostics/` in every run output | Eval cache, per-class errors, calibration, fixed probes, and paired module effects |
| A-11 | `src/tools/validate_baseline_monitoring.py` | Synthetic monitoring and output-policy regression gate |
| A-12 | `src/tools/summarize_baseline_monitoring.py` | Offline multi-seed mean/CI and paired-delta summary |

The A series is deliberately restricted to the same attribute-prototype semantic matcher and seen-class cross-entropy. It is not a result claim and does not replace the separate protocol, split, evaluator, or best-validation-checkpoint work.

Run the static gate before training:

```powershell
C:\Users\84291\.conda\envs\prompt\python.exe src\tools\A-audit_baseline_config.py configs\baseline_rebuild\A-02-A0-frozen-vit-ce.yaml
C:\Users\84291\.conda\envs\prompt\python.exe src\tools\A-audit_baseline_config.py configs\baseline_rebuild\A-03-A1-vpt-shallow-ce.yaml
C:\Users\84291\.conda\envs\prompt\python.exe src\tools\A-audit_baseline_config.py configs\baseline_rebuild\A-04-A2-vpt-deep-ce.yaml
```

After the static gate passes, run the constructed-model gate once per configuration:

```powershell
C:\Users\84291\.conda\envs\prompt\python.exe src\tools\A-audit_baseline_config.py configs\baseline_rebuild\A-02-A0-frozen-vit-ce.yaml --build-model
```

`--build-model` loads the configured ViT checkpoint and verifies trainable parameter names, so it is intentionally separate from the fast static gate.

Run the monitoring regression gate after changing the Trainer, evaluator, registry, adapters, or diagnostic writers:

```powershell
C:\Users\84291\.conda\envs\prompt\python.exe src\tools\validate_baseline_monitoring.py
```

The A configs keep training-time affinity monitoring disabled. Attention and affinity references run only on the deterministic final fixed probe; the probe records a strict normal/affinity-forward logits equivalence check. A1/A2 additionally run a paired prompt-zero intervention against the same final checkpoint and sample manifest. A0 records the module-effect manifest with no active prompt intervention.

Future baseline work stays in this directory and uses the next top-level letter only after the A series has been frozen.
