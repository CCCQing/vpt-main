#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


EXPECTED_GRAPH_SHA256 = "50B6AAC7F1D4DEAB1680510500D2DB308DEF213AD323C8E503163DE5C5CEA7BE"
GZSL_PATTERN = re.compile(
    r"\[gzsl-record\]\s+epoch=(?P<epoch>\d+)\s+"
    r"gzsl_seen=(?P<seen>[0-9.]+)\s+"
    r"gzsl_unseen=(?P<unseen>[0-9.]+)\s+"
    r"gzsl_h=(?P<h>[0-9.]+)"
)


def _latest_file(root, name):
    files = list(Path(root).rglob(name))
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def validate_run(run_root, expected_seed, expected_epochs, expected_gpp, expected_am):
    run_root = Path(run_root).resolve()
    log_path = _latest_file(run_root, "logs.txt")
    identity_path = _latest_file(run_root, "audit_identity.json")
    config_path = _latest_file(run_root, "resolved_config.yaml")
    checkpoint_path = _latest_file(run_root, "model_final.pth")
    errors = []

    if log_path is None:
        errors.append("missing logs.txt")
        log_text = ""
    else:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in log_text:
            errors.append("traceback found in logs.txt")

    records = []
    for match in GZSL_PATTERN.finditer(log_text):
        records.append(
            {
                "epoch": int(match.group("epoch")),
                "seen": float(match.group("seen")),
                "unseen": float(match.group("unseen")),
                "h": float(match.group("h")),
            }
        )
    if not records:
        errors.append("missing gzsl-record metrics")
    elif records[-1]["epoch"] != int(expected_epochs):
        errors.append(
            "final gzsl epoch mismatch: expected {}, got {}".format(
                int(expected_epochs), records[-1]["epoch"]
            )
        )

    identity = {}
    if identity_path is None:
        errors.append("missing audit_identity.json")
    else:
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid audit_identity.json: {exc}")

    if identity:
        if identity.get("seed") != int(expected_seed):
            errors.append(
                "seed mismatch: expected {}, got {}".format(
                    int(expected_seed), identity.get("seed")
                )
            )
        switches = identity.get("switches", {})
        if bool(switches.get("graph_prob_prior_enable")) != bool(expected_gpp):
            errors.append("GraphProbPrior switch mismatch")
        if bool(switches.get("attention_mediation_enable")) != bool(expected_am):
            errors.append("Attention Mediation switch mismatch")
        if bool(expected_gpp):
            graph = identity.get("graph", {})
            if not bool(graph.get("exists")):
                errors.append("GraphProbPrior graph file missing")
            if str(graph.get("sha256", "")).upper() != EXPECTED_GRAPH_SHA256:
                errors.append("GraphProbPrior graph SHA-256 mismatch")
        if not bool(identity.get("training_performed")):
            errors.append("training_performed is not true")
        if not bool(identity.get("optimizer_created")):
            errors.append("optimizer_created is not true")

    if config_path is None:
        errors.append("missing resolved_config.yaml")
    if checkpoint_path is None:
        errors.append("missing model_final.pth")

    result = {
        "schema_version": 1,
        "run_root": str(run_root),
        "valid": not errors,
        "errors": errors,
        "expected": {
            "seed": int(expected_seed),
            "epochs": int(expected_epochs),
            "graph_prob_prior_enable": bool(expected_gpp),
            "attention_mediation_enable": bool(expected_am),
        },
        "artifacts": {
            "log": str(log_path) if log_path else "",
            "identity": str(identity_path) if identity_path else "",
            "resolved_config": str(config_path) if config_path else "",
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        },
        "final_metrics": records[-1] if records else {},
        "epoch_record_count": len(records),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate one C-series t0009 training run.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--expected-gpp", type=int, choices=[0, 1], required=True)
    parser.add_argument("--expected-am", type=int, choices=[0, 1], required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = validate_run(
        args.run_root,
        args.expected_seed,
        args.expected_epochs,
        bool(args.expected_gpp),
        bool(args.expected_am),
    )
    output_path = Path(args.output_json) if args.output_json else Path(args.run_root) / "validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
