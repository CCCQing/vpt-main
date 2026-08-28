#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


EXPECTED_COMMIT = "6e70b9e4c8fda553be7b193eace8249dd2cf5d43"
EXPECTED_GRAPH_SHA256 = "50B6AAC7F1D4DEAB1680510500D2DB308DEF213AD323C8E503163DE5C5CEA7BE"
EXPECTED_SPLITS = {
    "trainval": {"images": 7057, "classes": 150},
    "test_seen": {"images": 1764, "classes": 150},
    "test_unseen": {"images": 2967, "classes": 50},
}
GZSL_PATTERN = re.compile(
    r"\[gzsl-record\]\s+epoch=(?P<epoch>\d+)\s+"
    r"gzsl_seen=(?P<seen>[0-9.]+)\s+"
    r"gzsl_unseen=(?P<unseen>[0-9.]+)\s+"
    r"gzsl_h=(?P<h>[0-9.]+)"
)
SPLIT_PATTERN = re.compile(
    r"XLSA split=(?P<split>trainval|test_seen|test_unseen)\s+"
    r"images=(?P<images>\d+)\s+classes=(?P<classes>\d+)"
)


def _latest_file(root, name):
    files = list(Path(root).rglob(name))
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def _load_json(path, errors, label):
    if not path.is_file():
        errors.append("missing {}".format(label))
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append("invalid {}: {}".format(label, exc))
        return {}


def validate_run(
    run_root,
    expected_seed_mode,
    expected_seed=None,
    expected_epochs=10,
    expected_commit=EXPECTED_COMMIT,
):
    run_root = Path(run_root).resolve()
    errors = []
    identity_path = run_root / "legacy_identity.json"
    command_path = run_root / "runner_command.json"
    identity = _load_json(identity_path, errors, "legacy_identity.json")
    command_record = _load_json(command_path, errors, "runner_command.json")
    log_path = _latest_file(run_root, "logs.txt")
    launcher_log = run_root / "legacy_launcher.log"

    texts = []
    metric_text = ""
    if launcher_log.is_file():
        texts.append(launcher_log.read_text(encoding="utf-8", errors="replace"))
    if log_path is not None:
        metric_text = log_path.read_text(encoding="utf-8", errors="replace")
        texts.append(metric_text)
    else:
        errors.append("missing logs.txt")
    log_text = "\n".join(texts)
    if "Traceback (most recent call last)" in log_text:
        errors.append("traceback found in logs")

    records = []
    for match in GZSL_PATTERN.finditer(metric_text):
        item = {
            "epoch": int(match.group("epoch")),
            "seen": float(match.group("seen")),
            "unseen": float(match.group("unseen")),
            "h": float(match.group("h")),
        }
        if not records or item != records[-1]:
            records.append(item)
    if len(records) != int(expected_epochs):
        errors.append(
            "gzsl record count mismatch: expected {}, got {}".format(
                int(expected_epochs), len(records)
            )
        )
    if records and records[-1]["epoch"] != int(expected_epochs):
        errors.append(
            "final epoch mismatch: expected {}, got {}".format(
                int(expected_epochs), records[-1]["epoch"]
            )
        )

    observed_splits = {}
    for match in SPLIT_PATTERN.finditer(log_text):
        observed_splits[match.group("split")] = {
            "images": int(match.group("images")),
            "classes": int(match.group("classes")),
        }
    for split_name, expected in EXPECTED_SPLITS.items():
        if observed_splits.get(split_name) != expected:
            errors.append(
                "split contract mismatch for {}: expected {}, got {}".format(
                    split_name, expected, observed_splits.get(split_name)
                )
            )

    if identity:
        if str(identity.get("legacy_commit")) != str(expected_commit):
            errors.append("legacy commit mismatch")
        if bool(identity.get("legacy_dirty")):
            errors.append("legacy worktree is dirty")
        if str(identity.get("graph_sha256", "")).upper() != EXPECTED_GRAPH_SHA256:
            errors.append("historical graph SHA-256 mismatch")
        if not identity.get("local_path_config_sha256"):
            errors.append("missing local path config SHA-256")
        if not identity.get("attr_name_embed_sha256"):
            errors.append("missing attribute-name embedding SHA-256")
        if identity.get("seed_mode") != expected_seed_mode:
            errors.append("seed mode mismatch")
        if expected_seed_mode == "fixed":
            if identity.get("seed") != int(expected_seed):
                errors.append("fixed seed mismatch")
        elif identity.get("seed") is not None:
            errors.append("unseeded replay unexpectedly records a fixed seed")
        if not bool(identity.get("training_performed")):
            errors.append("training_performed is not true")
        if not bool(identity.get("optimizer_created")):
            errors.append("optimizer_created is not true")

    command = command_record.get("command", [])
    seed_positions = [index for index, token in enumerate(command) if token == "SEED"]
    if expected_seed_mode == "unseeded":
        if seed_positions:
            errors.append("unseeded command contains SEED override")
    else:
        if len(seed_positions) != 1:
            errors.append("fixed command must contain exactly one SEED override")
        elif seed_positions[0] + 1 >= len(command):
            errors.append("fixed command has malformed SEED override")
        elif command[seed_positions[0] + 1] != str(int(expected_seed)):
            errors.append("fixed command SEED value mismatch")

    required_signatures = ["Python", "PyTorch", "CUDA available", "CuDNN"]
    missing_signatures = [item for item in required_signatures if item not in log_text]
    if missing_signatures:
        errors.append("missing environment signatures: {}".format(missing_signatures))

    checkpoint_path = _latest_file(run_root, "model_final.pth")
    result = {
        "schema_version": 1,
        "run_root": str(run_root),
        "valid": not errors,
        "errors": errors,
        "expected": {
            "seed_mode": expected_seed_mode,
            "seed": int(expected_seed) if expected_seed is not None else None,
            "epochs": int(expected_epochs),
            "legacy_commit": str(expected_commit),
            "graph_sha256": EXPECTED_GRAPH_SHA256,
            "checkpoint_expected": False,
        },
        "artifacts": {
            "log": str(log_path) if log_path else "",
            "launcher_log": str(launcher_log) if launcher_log.is_file() else "",
            "identity": str(identity_path),
            "command": str(command_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        },
        "observed_splits": observed_splits,
        "final_metrics": records[-1] if records else {},
        "epoch_record_count": len(records),
        "checkpoint_present": checkpoint_path is not None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate one exact-legacy t0009 RNG replay.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed-mode", choices=["unseeded", "fixed"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    args = parser.parse_args()
    if args.seed_mode == "fixed" and args.seed is None:
        parser.error("--seed is required when --seed-mode=fixed")
    result = validate_run(
        args.run_root,
        args.seed_mode,
        args.seed,
        args.epochs,
        expected_commit=args.expected_commit,
    )
    output_path = Path(args.output_json) if args.output_json else Path(args.run_root) / "validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
