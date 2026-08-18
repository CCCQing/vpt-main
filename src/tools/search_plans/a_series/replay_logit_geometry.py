"""Checkpoint-only full-test replay for method-level logit geometry."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.logit_geometry import analyze_logit_geometry
from src.tools.search_plans.a_series.replay_decision_gain_decomposition import _run_source


def parse_args():
    parser = argparse.ArgumentParser(
        description="Checkpoint-only full-test logit geometry for one trained method and seed."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "refusing to mix logit geometry into non-empty output: {}".format(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    run_dir = Path(args.run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(str(run_dir))
    _require_output(output_dir)
    try:
        cfg, identity, outputs = _run_source(
            run_dir,
            args.batch_size,
            args.num_workers,
            monitor_config_name="LOGIT_GEOMETRY",
        )
        candidate_class_ids = outputs["test_seen"]["candidate_class_ids"]
        seen_class_ids = outputs["test_seen"]["seen_class_ids"]
        unseen_class_ids = outputs["test_seen"]["unseen_class_ids"]
        monitor_cfg = cfg.MONITOR.LOGIT_GEOMETRY
        summary, arrays = analyze_logit_geometry(
            outputs,
            candidate_class_ids,
            seen_class_ids,
            unseen_class_ids,
            eps=float(monitor_cfg.FACTORIZATION_EPS),
            reconstruction_atol=float(monitor_cfg.RECONSTRUCTION_ATOL),
        )
        summary["method"] = {
            "name": str(args.method_name),
            "training_seed": int(cfg.SEED),
        }
        manifest = {
            "format": "logit_geometry_reference_manifest_v1",
            "status": "completed" if summary["validity"]["valid"] else "invalid",
            "channel": "single_method_checkpoint_only_full_test_replay",
            "method_name": str(args.method_name),
            "source": identity,
            "candidate_class_ids": candidate_class_ids,
            "seen_class_ids": seen_class_ids,
            "unseen_class_ids": unseen_class_ids,
            "sample_order_sha256_by_split": {
                split: outputs[split]["sample_order_sha256"] for split in outputs
            },
            "runtime": {
                "batch_size": int(cfg.DATA.BATCH_SIZE),
                "num_workers": int(cfg.DATA.NUM_WORKERS),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "optimizer_created": False,
                "parameter_update": False,
                "storage_mode": "per_class_aggregate_only",
                "sample_logits_persisted": False,
                "sample_logits_retained_on_cpu_until_aggregation": True,
            },
        }
        _write_json(output_dir / "logit_geometry_reference_manifest.json", manifest)
        _write_json(output_dir / "logit_geometry_reference_summary.json", summary)
        _write_json(output_dir / "logit_geometry_reference_validity.json", summary["validity"])
        np.savez_compressed(output_dir / "logit_geometry_reference_per_class.npz", **arrays)
        if not summary["validity"]["valid"]:
            raise RuntimeError("logit geometry failed validity gates")
        print(
            "Logit geometry completed: method={} seed={} output={}".format(
                args.method_name, cfg.SEED, output_dir
            )
        )
    except Exception:
        _write_json(
            output_dir / "logit_geometry_reference_failure.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
