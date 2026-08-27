from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.tools.search_plans.a_series.run_a2_cosine_search import (
    BASE_EPOCHS,
    BASE_LR,
    BASE_NUM_TOKENS,
    SCALE_SPECS,
    _command,
    _job,
    _rank_rows,
    _read_result,
)


def _write_completed_run(job) -> None:
    run_dir = job.output_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "monitor_runtime_summary.json").write_text(
        json.dumps({"status": "completed", "seed": job.seed}), encoding="utf-8"
    )
    rows = [
        {
            "epoch": job.total_epoch,
            "split": "test_gzsl",
            "namespace": "classification",
            "metric": "gzsl_seen",
            "value": "0.8",
        },
        {
            "epoch": job.total_epoch,
            "split": "test_gzsl",
            "namespace": "classification",
            "metric": "gzsl_unseen",
            "value": "0.4",
        },
        {
            "epoch": job.total_epoch,
            "split": "test_gzsl",
            "namespace": "classification",
            "metric": "gzsl_h",
            "value": "0.5333333333",
        },
        {
            "epoch": job.total_epoch,
            "split": "test_gzsl",
            "namespace": "calibration_profile",
            "metric": "ausuc",
            "value": "0.11",
        },
    ]
    with (run_dir / "metrics_epoch.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        job = _job(
            root,
            "01_anchor",
            BASE_NUM_TOKENS,
            BASE_LR,
            BASE_EPOCHS,
            0,
            SCALE_SPECS["fixed1"],
        )
        _write_completed_run(job)
        result = _read_result(job, existing=True)
        assert result["status"] == "completed_existing"
        assert abs(float(result["gzsl_h"]) - 0.5333333333) < 1.0e-9
        assert result["ausuc"] == 0.11

        command = _command("python", Path("config.yaml"), job)
        rendered = " ".join(command)
        assert "MODEL.R_SIMILARITY.SCORE_MODE cosine" in rendered
        assert "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE 1.0" in rendered
        assert "MONITOR.PROBE.ENABLE False" in rendered
        assert "MONITOR.MODULE_EFFECT.ENABLE False" in rendered

        rows = [
            dict(result, trial_name="high_h_bad_ausuc", gzsl_h=0.60, ausuc=0.08),
            dict(result, trial_name="guarded", gzsl_h=0.55, ausuc=0.105),
        ]
        ranked = _rank_rows(rows, anchor_ausuc=0.11, ausuc_tolerance=0.01)
        assert ranked[0]["trial_name"] == "guarded"
        assert ranked[0]["ausuc_guard_pass"]

    assert set(SCALE_SPECS) == {"fixed1", "fixed10", "learnable10"}
    assert SCALE_SPECS["learnable10"].learnable
    assert SCALE_SPECS["learnable10"].fixed_scale == 0.0
    print("A2-Cosine search validation passed.")


if __name__ == "__main__":
    main()
