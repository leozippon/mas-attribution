#!/usr/bin/env python3
"""Normalize serialized SWE-bench unified-diff predictions for official scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_patch(value: object) -> str:
    patch = str(value or "").strip()
    if patch.startswith("```diff") or patch.startswith("```patch") or patch.startswith("```text"):
        patch = patch.split("\n", 1)[1] if "\n" in patch else ""
    if patch.endswith("```"):
        patch = patch[:-3].rstrip()
    # Some final-answer JSON artifacts preserve escaped newlines literally.
    patch = patch.replace("\\r\\n", "\n").replace("\\n", "\n")
    return patch + "\n" if patch else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["model_patch"] = normalize_patch(row.get("model_patch"))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
