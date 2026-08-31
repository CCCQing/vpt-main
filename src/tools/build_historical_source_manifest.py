#!/usr/bin/env python3
"""Build or verify the checksum manifest for preserved historical source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.common import write_json


SOURCE_ROOT = ROOT / "src" / "tools" / "historical_files"
DEFAULT_OUTPUT = ROOT / "src" / "tools" / "historical_source_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        files.append(
            {
                "path": relative,
                "archive_group": path.relative_to(SOURCE_ROOT).parts[0],
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    identity = hashlib.sha256(
        "".join(
            "{}\t{}\t{}\n".format(
                item["path"], item["bytes"], item["sha256"]
            )
            for item in files
        ).encode("utf-8")
    ).hexdigest()
    return {
        "format": "historical_source_manifest_v1",
        "policy": (
            "Preserve these sources until an independently stored archive with "
            "matching checksums is verified; cache files are not part of the archive."
        ),
        "source_root": SOURCE_ROOT.relative_to(ROOT).as_posix(),
        "source_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "manifest_sha256": identity,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check:
        raise SystemExit("choose exactly one of --write or --check")

    expected = build_manifest()
    output = args.output.resolve()
    if args.write:
        write_json(output, expected, allow_nan=False, sort_keys=True)
        print(
            "historical source manifest written: files={} sha256={}".format(
                expected["source_count"], expected["manifest_sha256"]
            )
        )
        return

    if not output.is_file():
        raise FileNotFoundError(str(output))
    current = json.loads(output.read_text(encoding="utf-8"))
    if current != expected:
        raise RuntimeError("historical source manifest is stale: {}".format(output))
    print(
        "historical source manifest valid: files={} sha256={}".format(
            expected["source_count"], expected["manifest_sha256"]
        )
    )


if __name__ == "__main__":
    main()
