# A-series baseline plan

This package keeps the A0/A1/A2 baseline experiment logic separate from the generic parameter-search framework.

The layers are:

1. `audit_baseline_config.py`: static configuration and optional constructed-model audit.
2. `run_baseline_series.py`: fixed A0/A1/A2 multi-GPU experiment launcher with completed-run skipping.
3. `search_a2_vpt_deep_small_grid.py`: A2-specific prompt-depth hyperparameter search.
4. `validate_baseline_monitoring.py`: synthetic monitoring regression gate.
5. `summarize_baseline_monitoring.py`: offline multi-run summary and paired comparison.

`progress_dashboard.py` is shared internal support for the two launchers; it is not a standalone experiment entry.

Experiment YAML files remain in `configs/baseline_rebuild/`. Generic scheduling and ETA infrastructure remains in `src/tools/parameter_search/`.

`run_baseline_series.py` and `search_a2_vpt_deep_small_grid.py` display one fixed-width ASCII total row plus one row for every active GPU worker. The parent process reads each trial's `progress.json`, combines live epoch/batch progress with compatible historical trial durations, and immediately backfills a worker when its current trial finishes.

The visualization defaults to 120 columns and refreshes every two seconds. Use `--progress-width`, `--progress-interval`, or `--no-progress` to change it. Child training output remains isolated in each launcher log, and the dashboard does not create an additional ETA state file.
