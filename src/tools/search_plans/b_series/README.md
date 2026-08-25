# B-series deterministic Prompt Distributor experiments

`B1` uses the image-specific frozen-ViT CLS to generate a deterministic shared mean residual. `B2` keeps the same prepass, statistics MLP, gates, and parameter count but replaces the CLS input with a fixed vector.

Run a two-method seed-0 technical gate:

```bash
python src/tools/search_plans/b_series/run_prompt_distribution_series.py --methods B1,B2 --seeds 0 --gpu-groups '0;1' --max-workers 2
```

After the technical gate passes, resume and complete three training seeds:

```bash
python src/tools/search_plans/b_series/run_prompt_distribution_series.py --methods B1,B2 --seeds 0,1,2 --gpu-groups '0;1' --max-workers 2
```

Each task is single-GPU. Existing completed tasks are skipped; incomplete non-empty task directories are never overwritten.

The additional experiments remain part of the B series and use the neutral atomic labels `E1` through `E7`. Their commands and validity contracts are defined in [B_SERIES_EXPERIMENT_PROTOCOL.md](B_SERIES_EXPERIMENT_PROTOCOL.md).

After replay cells finish, `summarize_b_series_replays.py` enforces atomic-cell completeness and summarizes three Probe selections within each checkpoint before aggregating independent training seeds.

The B3 `D2G` checkpoint-only completion is documented in
[B3_SERIES_EXPERIMENT_PROTOCOL.md](B3_SERIES_EXPERIMENT_PROTOCOL.md). Use
`validate_b3_d2g_result.py` for each cell and `summarize_b3_d2g.py` only after
all 24 full/strict-Probe cells exist and pass their identity contracts.
