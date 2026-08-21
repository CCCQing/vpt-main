#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import report_evidence_gate


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fixture(root: Path):
    summary = root / "summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("evidence_role", "method", "metric_key", "count", "mean", "min", "max"),
        )
        writer.writeheader()
        writer.writerow({
            "evidence_role": "formal_result",
            "method": "A0",
            "metric_key": "checkpoint=epoch_1|split=test|condition=normal|domain=classification|entity_type=split|entity_id=all|metric=top1",
            "count": "3",
            "mean": "0.5",
            "min": "0.4",
            "max": "0.6",
        })
    report = root / "report.md"
    report.write_text("# Demo report\n\n## 1. Task results\n\nThis is a complete experiment report.\n\n## 2. Coverage\n", encoding="utf-8")
    coverage = root / "coverage.json"
    _write_json(coverage, {
        "schema_version": 1,
        "coverage_rules": [{
            "rule_id": "classification",
            "selector": {"source_id": "metrics", "domain": "classification"},
            "disposition": "presented",
            "report_sections": ["1"],
            "reason": "",
            "conclusion_impact": "formal task result",
        }],
    })
    profile = root / "profile.json"
    _write_json(profile, {
        "schema_version": 1,
        "experiment_id": "demo",
        "repo_root": str(root),
        "report_path": str(report),
        "report_title": "Demo report",
        "status_after_heading": "Demo report",
        "inventory_path": str(root / "inventory.json"),
        "coverage_manifest_path": str(coverage),
        "resolved_coverage_path": str(root / "resolved.json"),
        "acceptance_path": str(root / "acceptance.json"),
        "sources": [{
            "source_id": "metrics",
            "type": "metric_summary",
            "path": str(summary),
        }],
        "discovery_globs": [str(root / "*.csv")],
        "auto_tables": [{
            "marker_id": "coverage",
            "kind": "coverage_summary",
            "after_heading": "2.",
        }],
        "completion_claim_patterns": [r"complete experiment report"],
    })
    return profile, coverage, report


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        profile, coverage, report = _fixture(root)
        assert report_evidence_gate.sync(profile) == 0
        assert report_evidence_gate.check(profile) == 0
        stray = root / "unregistered_summary.csv"
        stray.write_text("value\n1\n", encoding="utf-8")
        assert report_evidence_gate.check(profile) == 1
        stray.unlink()
        manifest = json.loads(coverage.read_text(encoding="utf-8"))
        manifest["coverage_rules"] = []
        _write_json(coverage, manifest)
        assert report_evidence_gate.check(profile) == 1
        report_text = report.read_text(encoding="utf-8")
        assert "complete experiment report" in report_text
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        orphan = root / "orphan"
        orphan.mkdir()
        (orphan / "实验结果分析.md").write_text("# Orphan\n", encoding="utf-8")
        assert report_evidence_gate.run_tree("check-tree", root, "实验结果分析.md") == 1
    print("report_evidence_gate tests passed")


if __name__ == "__main__":
    main()
