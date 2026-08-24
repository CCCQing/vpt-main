#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCHEMA_VERSION = 1
GROUP_DIMENSIONS = (
    "checkpoint",
    "split",
    "condition",
    "domain",
    "entity_type",
    "probe_selection_seed",
)
ALLOWED_DISPOSITIONS = {
    "presented",
    "audit_only",
    "detail_omitted",
    "not_applicable",
    "not_requested",
    "legacy_not_available",
    "invalid",
}
REPORTING_DISPOSITIONS = {"presented", "audit_only", "detail_omitted"}
DEFAULT_COMPLETION_PATTERNS = (
    r"(?:本报告|本轮实验|该实验|实验结果|证据集|监测链|分析)[^。；\n]{0,30}(?:构成|作为|是|已形成|已经形成|共同构成)[^。；\n]{0,12}(?:完整|完备)",
    r"(?:完整|完备)(?:、独立)?(?:的)?[^。；\n]{0,24}(?:实验结果|实验报告|分析报告|证据集|监测链|实验)",
)
BLOCK_RE_TEMPLATE = r"<!-- report-evidence:{marker}:start -->.*?<!-- report-evidence:{marker}:end -->"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_path(value: str, profile_path: Path, profile: Mapping[str, Any]) -> Path:
    variables = {
        "profile_dir": str(profile_path.parent),
        "repo_root": str(profile.get("repo_root", profile_path.parent)),
        "artifact_root": str(profile.get("artifact_root", "")),
        "report_dir": str(Path(str(profile.get("report_path", profile_path.parent))).parent),
    }
    expanded = str(value).format(**variables)
    path = Path(expanded)
    if not path.is_absolute():
        path = profile_path.parent / path
    return path.resolve()


def _resolve_glob(value: str, profile_path: Path, profile: Mapping[str, Any]) -> List[Path]:
    resolved = _resolve_path(value, profile_path, profile)
    matches = [Path(item).resolve() for item in glob.glob(str(resolved), recursive=True)]
    return sorted(path for path in matches if path.is_file())


def _open_csv(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _parse_metric_key(metric_key: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in str(metric_key).split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def _make_evidence_id(source_id: str, role: str, dimensions: Mapping[str, str]) -> str:
    canonical = "|".join([source_id, role] + [f"{key}={dimensions.get(key, '')}" for key in GROUP_DIMENSIONS])
    domain = dimensions.get("domain") or "unknown"
    split = dimensions.get("split") or "all"
    condition = dimensions.get("condition") or "all"
    return f"{source_id}:{domain}:{split}:{condition}:{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:12]}"


def _build_metric_summary_source(
    source: Mapping[str, Any],
    path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    groups: Dict[Tuple[str, ...], MutableMapping[str, Any]] = {}
    source_sha = _file_sha256(path)
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        required = {"evidence_role", "method", "metric_key", "count", "mean", "min", "max"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            return {}, [], [f"source {source['source_id']} missing CSV fields: {missing}"]
        for row_number, row in enumerate(reader, start=2):
            dimensions = _parse_metric_key(row.get("metric_key", ""))
            if not dimensions.get("domain"):
                errors.append(f"source {source['source_id']} row {row_number} has no domain in metric_key")
                if len(errors) >= 20:
                    break
                continue
            role = str(row.get("evidence_role") or "unspecified")
            key = (role,) + tuple(dimensions.get(name, "") for name in GROUP_DIMENSIONS)
            group = groups.get(key)
            if group is None:
                fixed_dimensions = {name: dimensions.get(name, "") for name in GROUP_DIMENSIONS}
                group = {
                    "evidence_id": _make_evidence_id(str(source["source_id"]), role, fixed_dimensions),
                    "source_id": str(source["source_id"]),
                    "source_type": "metric_summary",
                    "evidence_role": role,
                    "state": "available",
                    "dimensions": fixed_dimensions,
                    "methods": set(),
                    "metrics": set(),
                    "entity_id_hashes": set(),
                    "entity_id_examples": [],
                    "record_count": 0,
                    "seed_observation_count": 0,
                }
                groups[key] = group
            group["methods"].add(str(row.get("method") or ""))
            metric = dimensions.get("metric", "")
            if metric:
                group["metrics"].add(metric)
            entity_id = dimensions.get("entity_id", "")
            if entity_id:
                entity_hash = int.from_bytes(hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest(), "big")
                group["entity_id_hashes"].add(entity_hash)
                if len(group["entity_id_examples"]) < 8 and entity_id not in group["entity_id_examples"]:
                    group["entity_id_examples"].append(entity_id)
            group["record_count"] += 1
            try:
                group["seed_observation_count"] += int(float(row.get("count") or 0))
            except ValueError:
                errors.append(f"source {source['source_id']} row {row_number} has invalid count={row.get('count')!r}")
    evidence: List[Dict[str, Any]] = []
    for group in groups.values():
        entity_hashes = group.pop("entity_id_hashes")
        group["entity_id_count"] = len(entity_hashes)
        group["entity_id_digest"] = hashlib.sha256(
            b"".join(value.to_bytes(8, "big") for value in sorted(entity_hashes))
        ).hexdigest()
        group["methods"] = sorted(group["methods"])
        group["metrics"] = sorted(group["metrics"])
        group["metric_count"] = len(group["metrics"])
        evidence.append(dict(group))
    evidence.sort(key=lambda item: item["evidence_id"])
    source_record = {
        "source_id": str(source["source_id"]),
        "source_type": "metric_summary",
        "required": bool(source.get("required", True)),
        "state": "available",
        "path": str(path),
        "sha256": source_sha,
        "atomic_group_count": len(evidence),
        "record_count": sum(item["record_count"] for item in evidence),
    }
    return source_record, evidence, errors


def _build_bundle_source(
    source: Mapping[str, Any],
    paths: Sequence[Path],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    expected_min = int(source.get("expected_min_files", 1))
    required = bool(source.get("required", True))
    if len(paths) < expected_min and required:
        errors.append(
            f"source {source['source_id']} matched {len(paths)} files, expected at least {expected_min}"
        )
    file_records = [{"path": str(path), "sha256": _file_sha256(path), "size": path.stat().st_size} for path in paths]
    state = "available" if len(paths) >= expected_min else "missing"
    dimensions = {name: str(source.get("dimensions", {}).get(name, "")) for name in GROUP_DIMENSIONS}
    if not dimensions["domain"]:
        dimensions["domain"] = str(source.get("domain", source["source_id"]))
    role = str(source.get("evidence_role", "mechanism_evidence"))
    evidence = [{
        "evidence_id": _make_evidence_id(str(source["source_id"]), role, dimensions),
        "source_id": str(source["source_id"]),
        "source_type": "artifact_bundle",
        "evidence_role": role,
        "state": state,
        "dimensions": dimensions,
        "methods": sorted(str(item) for item in source.get("methods", [])),
        "metrics": sorted(str(item) for item in source.get("metrics", [])),
        "metric_count": len(source.get("metrics", [])),
        "entity_id_count": len(paths),
        "entity_id_examples": [path.name for path in paths[:8]],
        "entity_id_digest": _payload_sha256([record["path"] for record in file_records]),
        "record_count": len(paths),
        "seed_observation_count": int(source.get("seed_observation_count", 0)),
    }]
    source_record = {
        "source_id": str(source["source_id"]),
        "source_type": "artifact_bundle",
        "required": required,
        "state": state,
        "paths": file_records,
        "bundle_sha256": _payload_sha256(file_records),
        "atomic_group_count": 1,
        "record_count": len(paths),
    }
    return source_record, evidence, errors


def build_inventory(profile_path: Path, profile: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str], set[Path]]:
    errors: List[str] = []
    sources: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    accounted_paths: set[Path] = set()
    source_ids: set[str] = set()
    for source in profile.get("sources", []):
        source_id = str(source.get("source_id", ""))
        if not source_id:
            errors.append("source without source_id")
            continue
        if source_id in source_ids:
            errors.append(f"duplicate source_id: {source_id}")
            continue
        source_ids.add(source_id)
        source_type = str(source.get("type", "artifact_bundle"))
        patterns = source.get("paths") or [source.get("path") or source.get("glob")]
        paths: List[Path] = []
        for pattern in patterns:
            if not pattern:
                continue
            paths.extend(_resolve_glob(str(pattern), profile_path, profile))
        paths = sorted(set(paths))
        accounted_paths.update(paths)
        if source_type == "metric_summary":
            if len(paths) != 1:
                errors.append(f"metric_summary source {source_id} must match exactly one file, got {len(paths)}")
                source_record = {
                    "source_id": source_id,
                    "source_type": source_type,
                    "required": bool(source.get("required", True)),
                    "state": "missing",
                    "atomic_group_count": 0,
                    "record_count": 0,
                }
                sources.append(source_record)
                continue
            source_record, source_evidence, source_errors = _build_metric_summary_source(source, paths[0])
        elif source_type == "artifact_bundle":
            source_record, source_evidence, source_errors = _build_bundle_source(source, paths)
        else:
            errors.append(f"unsupported source type {source_type!r} for {source_id}")
            continue
        sources.append(source_record)
        evidence.extend(source_evidence)
        errors.extend(source_errors)
    duplicate_ids = [item for item, count in _counts(entry["evidence_id"] for entry in evidence).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate atomic evidence ids: {duplicate_ids[:10]}")
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(profile.get("experiment_id", "")),
        "group_dimensions": list(GROUP_DIMENSIONS),
        "sources": sorted(sources, key=lambda item: item["source_id"]),
        "atomic_evidence": sorted(evidence, key=lambda item: item["evidence_id"]),
        "summary": {
            "source_count": len(sources),
            "atomic_group_count": len(evidence),
            "available_group_count": sum(item["state"] == "available" for item in evidence),
            "missing_group_count": sum(item["state"] != "available" for item in evidence),
        },
    }
    inventory["inventory_sha256"] = _payload_sha256(inventory)
    return inventory, errors, accounted_paths


def _counts(values: Iterable[str]) -> Dict[str, int]:
    result: Dict[str, int] = defaultdict(int)
    for value in values:
        result[value] += 1
    return dict(result)


def _selector_value(group: Mapping[str, Any], key: str) -> Any:
    if key in group:
        return group[key]
    return group.get("dimensions", {}).get(key, "")


def _matches_selector(group: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    for key, expected in selector.items():
        actual = _selector_value(group, key)
        if expected == "*":
            continue
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif isinstance(expected, Mapping):
            if "not" in expected and actual == expected["not"]:
                return False
            if "regex" in expected and re.fullmatch(str(expected["regex"]), str(actual)) is None:
                return False
        elif actual != expected:
            return False
    return True


def _headings(report_text: str) -> List[str]:
    return [match.group(1).strip() for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", report_text)]


def _section_exists(headings: Sequence[str], section: str) -> bool:
    section = str(section).strip()
    return any(
        heading == section
        or heading.startswith(section + ".")
        or heading.startswith(section + " ")
        for heading in headings
    )


def resolve_coverage(
    inventory: Mapping[str, Any],
    coverage: Mapping[str, Any],
    report_text: str,
) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    rules = coverage.get("coverage_rules", [])
    rule_ids = [str(rule.get("rule_id", "")) for rule in rules]
    if "" in rule_ids:
        errors.append("coverage rule without rule_id")
    duplicates = [item for item, count in _counts(rule_ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate coverage rule ids: {duplicates}")
    headings = _headings(report_text)
    resolved_items: List[Dict[str, Any]] = []
    rule_counts: Dict[str, int] = defaultdict(int)
    for group in inventory.get("atomic_evidence", []):
        matches = [rule for rule in rules if _matches_selector(group, rule.get("selector", {}))]
        if not matches:
            errors.append(f"unmapped atomic evidence: {group['evidence_id']}")
            continue
        if len(matches) > 1:
            errors.append(
                f"ambiguous atomic evidence {group['evidence_id']} matched rules "
                f"{[rule.get('rule_id') for rule in matches]}"
            )
            continue
        rule = matches[0]
        rule_id = str(rule["rule_id"])
        rule_counts[rule_id] += 1
        disposition = str(rule.get("disposition", ""))
        if disposition not in ALLOWED_DISPOSITIONS:
            errors.append(f"rule {rule_id} has invalid disposition {disposition!r}")
        sections = [str(item) for item in rule.get("report_sections", [])]
        if disposition in REPORTING_DISPOSITIONS and not sections:
            errors.append(f"rule {rule_id} with disposition {disposition} has no report_sections")
        for section in sections:
            if not _section_exists(headings, section):
                errors.append(f"rule {rule_id} references missing report section {section!r}")
        reason = str(rule.get("reason", "")).strip()
        conclusion_impact = str(rule.get("conclusion_impact", "")).strip()
        if disposition == "detail_omitted" and not reason:
            errors.append(f"rule {rule_id} uses detail_omitted without a reason")
        if not conclusion_impact:
            errors.append(f"rule {rule_id} has no conclusion_impact")
        resolved_items.append({
            "evidence_id": group["evidence_id"],
            "rule_id": rule_id,
            "disposition": disposition,
            "report_sections": sections,
            "reason": reason,
            "conclusion_impact": conclusion_impact,
        })
    for rule in rules:
        rule_id = str(rule.get("rule_id", ""))
        if rule_id and rule_counts.get(rule_id, 0) == 0 and not bool(rule.get("allow_empty", False)):
            errors.append(f"coverage rule {rule_id} matched no atomic evidence")
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": inventory.get("experiment_id", ""),
        "inventory_sha256": inventory.get("inventory_sha256", ""),
        "coverage_manifest_sha256": _payload_sha256(coverage),
        "items": sorted(resolved_items, key=lambda item: item["evidence_id"]),
        "rule_counts": dict(sorted(rule_counts.items())),
        "summary": {
            "mapped_count": len(resolved_items),
            "unmapped_count": len(inventory.get("atomic_evidence", [])) - len(resolved_items),
            "disposition_counts": _counts(item["disposition"] for item in resolved_items),
        },
    }
    resolved["resolved_sha256"] = _payload_sha256(resolved)
    return resolved, errors


def _discover_unaccounted(
    profile_path: Path,
    profile: Mapping[str, Any],
    accounted_paths: set[Path],
) -> List[str]:
    errors: List[str] = []
    ignore_patterns = [str(item).replace("\\", "/") for item in profile.get("discovery_ignore", [])]
    for pattern in profile.get("discovery_globs", []):
        for path in _resolve_glob(str(pattern), profile_path, profile):
            normalized = str(path).replace("\\", "/")
            ignored = any(fnmatch.fnmatch(normalized, item) or fnmatch.fnmatch(path.name, item) for item in ignore_patterns)
            if path not in accounted_paths and not ignored:
                errors.append(f"discovered artifact is not registered by any source: {path}")
    return errors


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_escape_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_escape_cell(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _coverage_table(coverage: Mapping[str, Any], resolved: Mapping[str, Any]) -> str:
    rules = {str(rule["rule_id"]): rule for rule in coverage.get("coverage_rules", [])}
    rows: List[List[Any]] = []
    for rule_id, count in resolved.get("rule_counts", {}).items():
        rule = rules[rule_id]
        rows.append([
            rule_id,
            count,
            rule.get("disposition", ""),
            "、".join(str(item) for item in rule.get("report_sections", [])) or "—",
            rule.get("reason", "") or "—",
            rule.get("conclusion_impact", ""),
        ])
    return _markdown_table(
        ["覆盖规则", "原子证据组数", "正文处置", "正文位置", "省略/限定理由", "对结论的影响"],
        rows,
    )


def _source_table(inventory: Mapping[str, Any]) -> str:
    rows = []
    for source in inventory.get("sources", []):
        rows.append([
            source.get("source_id", ""),
            source.get("source_type", ""),
            source.get("state", ""),
            source.get("atomic_group_count", 0),
            source.get("record_count", 0),
            str(source.get("sha256") or source.get("bundle_sha256") or "")[:12],
        ])
    return _markdown_table(
        ["证据源", "类型", "状态", "原子组数", "记录/文件数", "内容摘要SHA"],
        rows,
    )


def _read_metric_rows(path: Path) -> Iterable[Tuple[Dict[str, str], Dict[str, str]]]:
    with _open_csv(path) as handle:
        for row in csv.DictReader(handle):
            yield row, _parse_metric_key(row.get("metric_key", ""))


def _metric_table(
    table: Mapping[str, Any],
    profile_path: Path,
    profile: Mapping[str, Any],
) -> Tuple[str, List[str]]:
    errors: List[str] = []
    source_id = str(table.get("source_id", ""))
    sources = {str(item.get("source_id")): item for item in profile.get("sources", [])}
    source = sources.get(source_id)
    if not source or source.get("type") != "metric_summary":
        return "", [f"metric table {table.get('marker_id')} references invalid metric source {source_id!r}"]
    patterns = source.get("paths") or [source.get("path") or source.get("glob")]
    matches: List[Path] = []
    for pattern in patterns:
        if pattern:
            matches.extend(_resolve_glob(str(pattern), profile_path, profile))
    if len(set(matches)) != 1:
        return "", [f"metric table {table.get('marker_id')} source {source_id} did not resolve uniquely"]
    methods = [str(item) for item in table.get("methods", [])]
    method_labels = {str(key): str(value) for key, value in table.get("method_labels", {}).items()}
    row_specs = table.get("rows", [])
    selected: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row, dimensions in _read_metric_rows(next(iter(set(matches)))):
        for index, spec in enumerate(row_specs):
            selector = dict(spec.get("selector", {}))
            selector["metric"] = spec.get("metric", selector.get("metric", ""))
            candidate = {**dimensions, "method": row.get("method", ""), "evidence_role": row.get("evidence_role", "")}
            if row.get("method", "") in methods and _matches_selector(candidate, selector):
                selected[(row.get("method", ""), index)].append(row)
    rendered_rows: List[List[Any]] = []
    for index, spec in enumerate(row_specs):
        rendered = [str(spec.get("label", spec.get("metric", index)))]
        for method in methods:
            matches = selected.get((method, index), [])
            if len(matches) != 1:
                errors.append(
                    f"metric table {table.get('marker_id')} row {index} method {method} matched {len(matches)} rows"
                )
                rendered.append("ERROR")
                continue
            row = matches[0]
            mean = float(row["mean"])
            low = float(row["min"])
            high = float(row["max"])
            digits = int(spec.get("digits", table.get("digits", 4)))
            rendered.append(f"{mean:.{digits}f} ({low:.{digits}f}～{high:.{digits}f})")
        rendered_rows.append(rendered)
    headers = [method_labels.get(method, method) for method in methods]
    return _markdown_table([str(table.get("row_header", "指标"))] + headers, rendered_rows), errors


def _block(marker: str, body: str) -> str:
    return f"<!-- report-evidence:{marker}:start -->\n{body.rstrip()}\n<!-- report-evidence:{marker}:end -->"


def _replace_or_insert_block(report_text: str, marker: str, body: str, after_heading: str) -> Tuple[str, str | None]:
    rendered = _block(marker, body)
    pattern = re.compile(BLOCK_RE_TEMPLATE.format(marker=re.escape(marker)), re.DOTALL)
    if pattern.search(report_text):
        return pattern.sub(rendered, report_text, count=1), None
    heading_pattern = re.compile(rf"(?m)^(#{{1,6}}\s+{re.escape(after_heading)}[^\n]*)$")
    match = heading_pattern.search(report_text)
    if not match:
        return report_text, f"cannot insert auto block {marker}: heading prefix {after_heading!r} not found"
    position = match.end()
    return report_text[:position] + "\n\n" + rendered + report_text[position:], None


def _extract_block(report_text: str, marker: str) -> str | None:
    pattern = re.compile(BLOCK_RE_TEMPLATE.format(marker=re.escape(marker)), re.DOTALL)
    match = pattern.search(report_text)
    return match.group(0) if match else None


def _remove_block(report_text: str, marker: str) -> str:
    pattern = re.compile(BLOCK_RE_TEMPLATE.format(marker=re.escape(marker)), re.DOTALL)
    return pattern.sub("", report_text)


def _completion_claims(report_text: str, profile: Mapping[str, Any]) -> List[str]:
    clean = _remove_block(report_text, "acceptance-status")
    patterns = profile.get("completion_claim_patterns", DEFAULT_COMPLETION_PATTERNS)
    claims: List[str] = []
    for pattern in patterns:
        for match in re.finditer(str(pattern), clean):
            claims.append(match.group(0).strip())
    return sorted(set(claims))


def _render_auto_blocks(
    profile_path: Path,
    profile: Mapping[str, Any],
    inventory: Mapping[str, Any],
    coverage: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> Tuple[Dict[str, str], List[str]]:
    blocks: Dict[str, str] = {}
    errors: List[str] = []
    for table in profile.get("auto_tables", []):
        marker = str(table.get("marker_id", ""))
        kind = str(table.get("kind", ""))
        if not marker:
            errors.append("auto table without marker_id")
            continue
        if marker in blocks:
            errors.append(f"duplicate auto table marker_id: {marker}")
            continue
        if kind == "coverage_summary":
            body = _coverage_table(coverage, resolved)
        elif kind == "source_summary":
            body = _source_table(inventory)
        elif kind == "metric_summary":
            body, table_errors = _metric_table(table, profile_path, profile)
            errors.extend(table_errors)
        else:
            errors.append(f"unsupported auto table kind {kind!r} for marker {marker}")
            continue
        intro = str(table.get("intro", "")).strip()
        if intro:
            body = intro + "\n\n" + body
        blocks[marker] = body
    return blocks, errors


def _apply_auto_blocks(
    report_text: str,
    profile: Mapping[str, Any],
    blocks: Mapping[str, str],
) -> Tuple[str, List[str]]:
    errors: List[str] = []
    table_by_id = {str(item.get("marker_id")): item for item in profile.get("auto_tables", [])}
    for marker, body in blocks.items():
        table = table_by_id[marker]
        report_text, error = _replace_or_insert_block(
            report_text,
            marker,
            body,
            str(table.get("after_heading", "")),
        )
        if error:
            errors.append(error)
    return report_text, errors


def _validate_auto_blocks(report_text: str, blocks: Mapping[str, str]) -> List[str]:
    errors: List[str] = []
    for marker, body in blocks.items():
        actual = _extract_block(report_text, marker)
        expected = _block(marker, body)
        if actual is None:
            errors.append(f"missing generated report block: {marker}")
        elif actual != expected:
            errors.append(f"stale or manually edited generated report block: {marker}")
    return errors


def _acceptance_body(
    passed: bool,
    inventory: Mapping[str, Any],
    resolved: Mapping[str, Any],
    error_count: int,
) -> str:
    if passed:
        status = "**PASSED（允许将本报告标记为完整）**"
        action = "原子证据、正文覆盖、独立产物登记和自动表格均通过当前机器验收。"
    else:
        status = "**FAILED（本报告不得标记为完整）**"
        action = "存在未登记、歧义映射、缺失章节、过期表格或产物问题；先修复并重新运行验收。"
    return (
        f"> 机器验收状态：{status}  \n"
        f"> 原子证据组：`{inventory.get('summary', {}).get('atomic_group_count', 0)}`；"
        f"已映射：`{resolved.get('summary', {}).get('mapped_count', 0)}`；错误：`{error_count}`。  \n"
        f"> {action}  \n"
        f"> inventory SHA：`{str(inventory.get('inventory_sha256', ''))[:12]}`；"
        f"coverage SHA：`{str(resolved.get('resolved_sha256', ''))[:12]}`。"
    )


def _receipt(
    profile: Mapping[str, Any],
    inventory: Mapping[str, Any],
    coverage: Mapping[str, Any],
    resolved: Mapping[str, Any],
    report_text: str,
    errors: Sequence[str],
    warnings: Sequence[str],
    auto_blocks: Mapping[str, str],
) -> Dict[str, Any]:
    passed = not errors
    report_without_status = _remove_block(report_text, "acceptance-status")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "validator_sha256": _file_sha256(Path(__file__).resolve()),
        "experiment_id": profile.get("experiment_id", ""),
        "status": "passed" if passed else "failed",
        "report_may_claim_complete": passed,
        "inventory_sha256": inventory.get("inventory_sha256", ""),
        "coverage_manifest_sha256": _payload_sha256(coverage),
        "resolved_coverage_sha256": resolved.get("resolved_sha256", ""),
        "report_sha256_without_status": hashlib.sha256(report_without_status.encode("utf-8")).hexdigest(),
        "auto_block_sha256": {key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in sorted(auto_blocks.items())},
        "summary": {
            "source_count": inventory.get("summary", {}).get("source_count", 0),
            "atomic_group_count": inventory.get("summary", {}).get("atomic_group_count", 0),
            "mapped_count": resolved.get("summary", {}).get("mapped_count", 0),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "errors": list(errors),
        "warnings": list(warnings),
    }
    payload["receipt_sha256"] = _payload_sha256(payload)
    return payload


def _profile_paths(profile_path: Path, profile: Mapping[str, Any]) -> Dict[str, Path]:
    required = ("report_path", "inventory_path", "coverage_manifest_path", "resolved_coverage_path", "acceptance_path")
    missing = [key for key in required if not profile.get(key)]
    if missing:
        raise ValueError(f"profile missing paths: {missing}")
    return {key: _resolve_path(str(profile[key]), profile_path, profile) for key in required}


def _base_validation(
    profile_path: Path,
    profile: Mapping[str, Any],
    inventory: Mapping[str, Any],
    coverage: Mapping[str, Any],
    resolved: Mapping[str, Any],
    report_text: str,
    accounted_paths: set[Path],
    inventory_errors: Sequence[str],
    coverage_errors: Sequence[str],
    table_errors: Sequence[str],
    auto_blocks: Mapping[str, str],
) -> List[str]:
    errors = list(inventory_errors) + list(coverage_errors) + list(table_errors)
    errors.extend(_discover_unaccounted(profile_path, profile, accounted_paths))
    errors.extend(_validate_auto_blocks(report_text, auto_blocks))
    if any(item.get("state") != "available" and item.get("required", True) for item in inventory.get("sources", [])):
        errors.append("one or more required evidence sources are not available")
    return sorted(set(errors))


def sync(profile_path: Path) -> int:
    profile_path = profile_path.resolve()
    profile = _load_json(profile_path)
    paths = _profile_paths(profile_path, profile)
    coverage = _load_json(paths["coverage_manifest_path"])
    report_text = paths["report_path"].read_text(encoding="utf-8")
    inventory, inventory_errors, accounted_paths = build_inventory(profile_path, profile)
    resolved, coverage_errors = resolve_coverage(inventory, coverage, report_text)
    auto_blocks, table_errors = _render_auto_blocks(profile_path, profile, inventory, coverage, resolved)
    report_text, insertion_errors = _apply_auto_blocks(report_text, profile, auto_blocks)
    resolved, coverage_errors_after = resolve_coverage(inventory, coverage, report_text)
    base_errors = _base_validation(
        profile_path,
        profile,
        inventory,
        coverage,
        resolved,
        report_text,
        accounted_paths,
        inventory_errors,
        coverage_errors + coverage_errors_after,
        table_errors + insertion_errors,
        auto_blocks,
    )
    claims = _completion_claims(report_text, profile)
    errors = list(base_errors)
    if base_errors and claims:
        errors.append("completion claim exists while validation is failing: " + " | ".join(claims[:8]))
    passed = not errors
    status_body = _acceptance_body(passed, inventory, resolved, len(errors))
    report_text, status_error = _replace_or_insert_block(
        report_text,
        "acceptance-status",
        status_body,
        str(profile.get("status_after_heading", profile.get("report_title", ""))),
    )
    if status_error:
        errors.append(status_error)
        passed = False
        status_body = _acceptance_body(False, inventory, resolved, len(errors))
        report_text, _ = _replace_or_insert_block(
            report_text,
            "acceptance-status",
            status_body,
            str(profile.get("status_after_heading", profile.get("report_title", ""))),
        )
    paths["report_path"].write_text(report_text, encoding="utf-8")
    _write_json(paths["inventory_path"], inventory)
    _write_json(paths["resolved_coverage_path"], resolved)
    receipt = _receipt(profile, inventory, coverage, resolved, report_text, errors, [], auto_blocks)
    receipt["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(paths["acceptance_path"], receipt)
    print(json.dumps({
        "experiment_id": profile.get("experiment_id", ""),
        "status": receipt["status"],
        "atomic_groups": inventory["summary"]["atomic_group_count"],
        "mapped_groups": resolved["summary"]["mapped_count"],
        "errors": errors,
        "acceptance_path": str(paths["acceptance_path"]),
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def check(profile_path: Path) -> int:
    profile_path = profile_path.resolve()
    profile = _load_json(profile_path)
    paths = _profile_paths(profile_path, profile)
    coverage = _load_json(paths["coverage_manifest_path"])
    report_text = paths["report_path"].read_text(encoding="utf-8")
    inventory, inventory_errors, accounted_paths = build_inventory(profile_path, profile)
    resolved, coverage_errors = resolve_coverage(inventory, coverage, report_text)
    auto_blocks, table_errors = _render_auto_blocks(profile_path, profile, inventory, coverage, resolved)
    errors = _base_validation(
        profile_path,
        profile,
        inventory,
        coverage,
        resolved,
        report_text,
        accounted_paths,
        inventory_errors,
        coverage_errors,
        table_errors,
        auto_blocks,
    )
    if paths["inventory_path"].exists():
        if _load_json(paths["inventory_path"]) != inventory:
            errors.append("stored atomic evidence inventory is stale")
    else:
        errors.append("stored atomic evidence inventory is missing")
    if paths["resolved_coverage_path"].exists():
        if _load_json(paths["resolved_coverage_path"]) != resolved:
            errors.append("stored resolved coverage list is stale")
    else:
        errors.append("stored resolved coverage list is missing")
    receipt = _receipt(profile, inventory, coverage, resolved, report_text, errors, [], auto_blocks)
    stored_receipt = _load_json(paths["acceptance_path"]) if paths["acceptance_path"].exists() else None
    if stored_receipt is None:
        errors.append("acceptance receipt is missing")
    else:
        comparable = {key: value for key, value in stored_receipt.items() if key != "generated_at_utc"}
        if comparable != receipt:
            errors.append("acceptance receipt is stale")
    claims = _completion_claims(report_text, profile)
    if errors and claims:
        errors.append("completion claim exists while validation is failing: " + " | ".join(claims[:8]))
    expected_status = _block(
        "acceptance-status",
        _acceptance_body(not errors, inventory, resolved, len(errors)),
    )
    actual_status = _extract_block(report_text, "acceptance-status")
    if actual_status != expected_status:
        errors.append("acceptance status block is missing or stale")
    errors = sorted(set(errors))
    print(json.dumps({
        "experiment_id": profile.get("experiment_id", ""),
        "status": "passed" if not errors else "failed",
        "atomic_groups": inventory["summary"]["atomic_group_count"],
        "mapped_groups": resolved["summary"]["mapped_count"],
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def run_tree(command: str, root: Path, report_name: str) -> int:
    root = root.resolve()
    reports = sorted(path for path in root.rglob(report_name) if path.is_file())
    errors: List[str] = []
    if not reports:
        errors.append(f"no reports named {report_name!r} found under {root}")
    profiles: List[Path] = []
    for report in reports:
        profile = report.parent / "report_evidence_profile.json"
        if not profile.is_file():
            errors.append(f"report has no evidence profile and cannot claim completeness: {report}")
        else:
            profiles.append(profile)
    failed_profiles: List[str] = []
    for profile in profiles:
        return_code = sync(profile) if command == "sync-tree" else check(profile)
        if return_code != 0:
            failed_profiles.append(str(profile))
    if failed_profiles:
        errors.append(f"{len(failed_profiles)} report profiles failed: {failed_profiles}")
    print(json.dumps({
        "tree_root": str(root),
        "status": "passed" if not errors else "failed",
        "report_count": len(reports),
        "profile_count": len(profiles),
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate experiment-report evidence coverage.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("sync", "check"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--profile", type=Path, required=True)
    for command in ("sync-tree", "check-tree"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", type=Path, required=True)
        command_parser.add_argument("--report-name", default="实验结果分析.md")
    args = parser.parse_args(argv)
    try:
        if args.command == "sync":
            return sync(args.profile)
        if args.command == "check":
            return check(args.profile)
        return run_tree(args.command, args.root, args.report_name)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"report evidence gate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
