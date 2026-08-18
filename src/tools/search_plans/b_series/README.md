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
