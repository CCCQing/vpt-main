#!/usr/bin/env python3
"""
Parse training logs and summarize forward/grad/update health for trace debugging.
"""
import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, Optional


NODE_A = "node=A.forward_one_batch"
NODE_B = "node=B.incorporate_prompt"
NODE_C = "node=C.vit_models.forward"


@dataclass
class StatusRow:
    forward: str = "N/A"
    grad: str = "N/A"
    update: str = "N/A"
    detail: str = ""


def _to_status_from_grad(payload: str) -> str:
    text = payload.strip()
    if text.startswith("MISSING"):
        return "MISSING"
    if text.startswith("None"):
        return "NONE"
    m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if not m:
        return "UNKNOWN"
    val = float(m.group(1))
    return "YES" if val > 0 else "ZERO"


def _to_status_from_step(payload: str) -> str:
    text = payload.strip()
    if "unavailable" in text:
        return "UNAVAILABLE"
    m = re.search(r"delta=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if not m:
        return "UNKNOWN"
    val = float(m.group(1))
    return "YES" if abs(val) > 0 else "ZERO"


def parse_log(path: str) -> Dict[str, StatusRow]:
    rows: Dict[str, StatusRow] = {
        "forward.trainer": StatusRow(),
        "forward.prompt_incorporate": StatusRow(),
        "forward.head_route": StatusRow(),
    }

    grad_re = re.compile(r"\[debug\]\s+grad\s+(.+?)\s*:\s*(.+)")
    step_re = re.compile(r"\[debug\]\s+step\s+(.+?)\s*:\s*(.+)")
    trace_re = re.compile(r"\[trace\]\s+(.+)$")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[trace]" in line:
                tm = trace_re.search(line)
                payload = tm.group(1) if tm else line.strip()
                if NODE_A in line:
                    rows["forward.trainer"].forward = "YES"
                    rows["forward.trainer"].detail = payload
                elif NODE_B in line:
                    rows["forward.prompt_incorporate"].forward = "YES"
                    rows["forward.prompt_incorporate"].detail = payload
                elif NODE_C in line:
                    rows["forward.head_route"].forward = "YES"
                    rows["forward.head_route"].detail = payload
                continue

            gm = grad_re.search(line)
            if gm:
                alias = gm.group(1).strip()
                payload = gm.group(2).strip()
                key = f"param.{alias}"
                row = rows.setdefault(key, StatusRow())
                row.grad = _to_status_from_grad(payload)
                row.detail = payload
                continue

            sm = step_re.search(line)
            if sm:
                alias = sm.group(1).strip()
                payload = sm.group(2).strip()
                key = f"param.{alias}"
                row = rows.setdefault(key, StatusRow())
                row.update = _to_status_from_step(payload)
                row.detail = payload
                continue

    # Default unresolved forward nodes to NO (more readable than N/A).
    for key in ("forward.trainer", "forward.prompt_incorporate", "forward.head_route"):
        if rows[key].forward == "N/A":
            rows[key].forward = "NO"

    return rows


def print_table(rows: Dict[str, StatusRow]) -> None:
    header = ["Module", "Forward", "Grad", "Update", "Detail"]
    keys = sorted(rows.keys())
    widths = {
        "Module": max(len("Module"), *(len(k) for k in keys)),
        "Forward": len("Forward"),
        "Grad": len("Grad"),
        "Update": len("Update"),
        "Detail": 80,
    }

    def cut(s: str, n: int) -> str:
        return s if len(s) <= n else s[: n - 3] + "..."

    print(
        f"{header[0]:<{widths['Module']}}  "
        f"{header[1]:<{widths['Forward']}}  "
        f"{header[2]:<{widths['Grad']}}  "
        f"{header[3]:<{widths['Update']}}  "
        f"{header[4]}"
    )
    print("-" * (sum(widths.values()) + 10))
    for key in keys:
        row = rows[key]
        print(
            f"{key:<{widths['Module']}}  "
            f"{row.forward:<{widths['Forward']}}  "
            f"{row.grad:<{widths['Grad']}}  "
            f"{row.update:<{widths['Update']}}  "
            f"{cut(row.detail, widths['Detail'])}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize trace/grad/update status from logs.")
    ap.add_argument("--log", required=True, help="Path to logs.txt")
    ap.add_argument(
        "--format",
        default="table",
        choices=["table", "json"],
        help="Output format.",
    )
    args = ap.parse_args()

    rows = parse_log(args.log)
    if args.format == "json":
        payload = {k: asdict(v) for k, v in rows.items()}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()

