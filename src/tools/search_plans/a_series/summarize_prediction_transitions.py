#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fvcore.common.checkpoint import Checkpointer

from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.data.transforms import get_transforms
from src.models.build_model import build_model
from src.monitoring.prediction_transition import (
    PredictionTransitionAccumulator,
    correctness_pattern_counts,
)
from src.monitoring.probe import FixedProbeDataset
from src.tools.search_plans.a_series.summarize_baseline_monitoring import (
    _pair_definitions,
    _summary,
    load_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run paired A-series fixed probes and aggregate prediction transitions."
    )
    parser.add_argument(
        "--run", action="append", required=True, help="METHOD=OUTPUT_DIR; repeat per run"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_runs(items: Sequence[str]) -> Dict[str, list]:
    result: Dict[str, list] = defaultdict(list)
    for item in items:
        method, separator, path = str(item).partition("=")
        if not separator or not method.strip() or not path.strip():
            raise ValueError("--run must use METHOD=OUTPUT_DIR")
        result[method.strip()].append(load_run(method.strip(), Path(path).resolve()))
    return dict(result)


def _load_cfg(run_dir: Path, batch_size: int):
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.DATA.BATCH_SIZE = max(1, int(batch_size))
    cfg.freeze()
    return cfg


def _load_model_and_dataset(run_dir: Path, batch_size: int):
    cfg = _load_cfg(run_dir, batch_size)
    if bool(cfg.MODEL.SEMANTIC_TOKENS.ENABLE):
        raise ValueError(
            "A-series prediction transition currently requires semantic tokens to be disabled"
        )
    model, device = build_model(cfg)
    if str(cfg.MODEL.WEIGHT_PATH):
        checkpointer = Checkpointer(model)
        checkpointables = [
            key
            for key in checkpointer.checkpointables
            if key not in ["head.last_layer.bias", "head.last_layer.weight"]
        ]
        checkpointer.load(str(cfg.MODEL.WEIGHT_PATH), checkpointables)
    train_loader = data_loader.construct_trainval_loader(cfg)
    model.attach_r_similarity_head(train_loader.dataset.class_attributes)
    checkpoint_path = run_dir / str(cfg.SOLVER.TRAINABLE_FINAL_CHECKPOINT_NAME)
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("format") != "vpt_trainable_v1":
        raise ValueError(f"unsupported trainable checkpoint format: {checkpoint_path}")
    state = checkpoint.get("model_state", {})
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "unexpected trainable checkpoint keys: "
            + ",".join(incompatible.unexpected_keys[:20])
        )
    model.eval()
    return cfg, model, device, train_loader.dataset, checkpoint


@torch.no_grad()
def _predict_manifest(
    cfg,
    model,
    device,
    source_dataset,
    manifest: Mapping[str, Any],
    batch_size: int,
) -> Dict[str, list]:
    transform = get_transforms("test_seen", int(cfg.DATA.CROPSIZE))
    loader = torch.utils.data.DataLoader(
        FixedProbeDataset(source_dataset, manifest, transform),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    candidate_class_ids = [int(item) for item in manifest["candidate_class_ids"]]
    predictions = []
    targets = []
    sample_ids = []
    for batch in loader:
        inputs = batch["image"].to(device, non_blocking=True)
        logits = model(
            inputs,
            semantics=None,
            class_ids=candidate_class_ids,
            runtime_targets=None,
        )
        local_predictions = logits.detach().argmax(dim=1).cpu().tolist()
        predictions.extend(candidate_class_ids[int(index)] for index in local_predictions)
        targets.extend(int(item) for item in batch["label"].tolist())
        sample_ids.extend(str(item) for item in batch["sample_id"])
        model.clear_runtime_state()
    return {
        "predictions": predictions,
        "targets": targets,
        "sample_ids": sample_ids,
    }


def _manifest_by_split(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    root = run_dir / "diagnostics" / "probe_manifests"
    return {
        path.stem: _read_json(path)
        for path in sorted(root.glob("probe_*.json"))
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _numeric_metrics(payload: Mapping[str, Any]) -> Dict[str, float]:
    return {
        str(name): float(value)
        for name, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    expected_seeds = {
        int(item.strip())
        for item in str(args.expected_seeds).split(",")
        if item.strip()
    }
    try:
        by_method = _parse_runs(args.run)
    except ValueError as exc:
        raise SystemExit(str(exc))
    rows_by_seed = {
        method: {
            int(row["seed"]): row
            for row in rows
            if row.get("seed") is not None and not row.get("failed")
        }
        for method, rows in by_method.items()
    }
    pairs = _pair_definitions(by_method, "A0" if "A0" in by_method else sorted(by_method)[0])
    transition_rows = []
    per_seed = {}
    excluded = []
    cross_seed_values: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for seed in sorted(expected_seeds):
        available_rows = {
            method: rows[seed]
            for method, rows in rows_by_seed.items()
            if seed in rows
        }
        if len(available_rows) < 2:
            excluded.append({"seed": seed, "reason": "fewer_than_two_completed_methods"})
            continue
        fingerprints = {
            row.get("shared_condition_fingerprint") for row in available_rows.values()
        }
        if None in fingerprints or len(fingerprints) != 1:
            excluded.append({"seed": seed, "reason": "incompatible_shared_fingerprint"})
            continue
        manifests = {
            method: _manifest_by_split(Path(row["run_dir"]))
            for method, row in available_rows.items()
        }
        common_splits = set.intersection(
            *(set(value) for value in manifests.values())
        )
        compatible_splits = []
        for split in sorted(common_splits):
            split_manifests = [value[split] for value in manifests.values()]
            identities = {
                (
                    manifest.get("manifest_sha256"),
                    tuple(manifest.get("candidate_class_ids", [])),
                    tuple(row.get("sample_id") for row in manifest.get("samples", [])),
                )
                for manifest in split_manifests
            }
            if len(identities) == 1:
                compatible_splits.append(split)
        if not compatible_splits:
            excluded.append({"seed": seed, "reason": "no_compatible_primary_probe_split"})
            continue

        predictions_by_split: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(dict)
        checkpoint_audit = {}
        for method, row in sorted(available_rows.items()):
            run_dir = Path(row["run_dir"])
            cfg, model, device, source_dataset, checkpoint = _load_model_and_dataset(
                run_dir, args.batch_size
            )
            if int(cfg.SEED) != int(seed) or int(checkpoint.get("seed")) != int(seed):
                raise RuntimeError(
                    f"seed identity mismatch: method={method} expected={seed} "
                    f"config={cfg.SEED} checkpoint={checkpoint.get('seed')}"
                )
            checkpoint_audit[method] = {
                "run_dir": str(run_dir),
                "checkpoint": str(
                    run_dir / str(cfg.SOLVER.TRAINABLE_FINAL_CHECKPOINT_NAME)
                ),
                "checkpoint_format": checkpoint.get("format"),
                "checkpoint_seed": checkpoint.get("seed"),
            }
            for split in compatible_splits:
                predictions_by_split[split][method] = _predict_manifest(
                    cfg,
                    model,
                    device,
                    source_dataset,
                    manifests[method][split],
                    args.batch_size,
                )
            del model, source_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        seed_payload = {
            "shared_condition_fingerprint": next(iter(fingerprints)),
            "checkpoints": checkpoint_audit,
            "splits": {},
        }
        for split in compatible_splits:
            method_payloads = predictions_by_split[split]
            sample_id_sets = {
                tuple(payload["sample_ids"]) for payload in method_payloads.values()
            }
            target_sets = {
                tuple(payload["targets"]) for payload in method_payloads.values()
            }
            if len(sample_id_sets) != 1 or len(target_sets) != 1:
                raise RuntimeError(
                    f"paired probe order changed during execution: seed={seed} split={split}"
                )
            targets = next(iter(method_payloads.values()))["targets"]
            split_payload = {"pairs": {}}
            for target_method, reference_method, pair_name in pairs:
                if target_method not in method_payloads or reference_method not in method_payloads:
                    continue
                accumulator = PredictionTransitionAccumulator(
                    reference_method, target_method
                )
                accumulator.update(
                    method_payloads[reference_method]["predictions"],
                    method_payloads[target_method]["predictions"],
                    targets,
                )
                metrics = accumulator.finalize()
                split_payload["pairs"][pair_name] = metrics
                row = {
                    "seed": seed,
                    "split": split,
                    "pair": pair_name,
                    "probe_manifest_sha256": manifests[target_method][split][
                        "manifest_sha256"
                    ],
                    **metrics,
                }
                transition_rows.append(row)
                for metric_name, value in _numeric_metrics(metrics).items():
                    cross_seed_values[pair_name][split][metric_name].append(value)
            split_payload["correctness_patterns"] = correctness_pattern_counts(
                {
                    method: payload["predictions"]
                    for method, payload in method_payloads.items()
                },
                targets,
            )
            seed_payload["splits"][split] = split_payload
        per_seed[str(seed)] = seed_payload

    cross_seed = {
        pair_name: {
            split: {
                metric_name: _summary(values)
                for metric_name, values in sorted(metrics.items())
            }
            for split, metrics in sorted(splits.items())
        }
        for pair_name, splits in sorted(cross_seed_values.items())
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "prediction_transition_runs.csv", transition_rows)
    payload = {
        "format": "a_series_prediction_transition_v1",
        "status": "completed" if transition_rows else "no_valid_pairs",
        "expected_seeds": sorted(expected_seeds),
        "observed_seeds": sorted(int(seed) for seed in per_seed),
        "excluded_seeds": excluded,
        "per_seed": per_seed,
        "cross_seed": cross_seed,
        "execution": {
            "channel": "cross_run_aggregation_with_fixed_probe_replay",
            "paired_by": [
                "training_seed",
                "shared_condition_fingerprint",
                "probe_manifest_sha256",
                "candidate_class_ids",
                "sample_order",
            ],
            "storage_mode": "aggregate_only",
            "sample_predictions_persisted": False,
            "predictions_retained_in_memory_until_pair_aggregation": True,
        },
    }
    (output_dir / "prediction_transition_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "Prediction transition summary: seeds={} rows={} output={}".format(
            len(per_seed), len(transition_rows), output_dir
        )
    )


if __name__ == "__main__":
    main()
