"""Helpers for exporting model outputs to official benchmark evaluator formats."""

from __future__ import annotations

import json
import re
from typing import Any


DIFF_RE = re.compile(r"diff --git a/.+?(?=\n(?:diff --git a/|$))", flags=re.DOTALL)
TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(?:<\|im_start\|>)?function=[^\n]*\n(?P<payload>\{.*?\})\s*(?:</tool_call>)?", flags=re.DOTALL)


def unwrap_model_answer(text: str | None) -> str:
    """Return the most likely answer artifact from a MAS final answer."""
    if not text:
        return ""
    stripped = text.strip()
    tool_match = TOOL_CALL_JSON_RE.search(stripped)
    if tool_match:
        try:
            payload = json.loads(tool_match.group("payload"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("code", "model_patch", "patch", "diff", "answer", "final_answer", "solution", "artifact"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return unwrap_model_answer(value)
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[len("json") :].strip()
            if candidate.startswith(("python", "diff", "patch")):
                candidate = re.sub(r"^(python|diff|patch)\s*", "", candidate, count=1).strip()
            nested = unwrap_model_answer(candidate)
            if nested:
                return nested
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("model_patch", "patch", "diff", "artifact", "code", "answer", "final_answer", "solution"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return unwrap_model_answer(value)
    return stripped


def extract_unified_diff(text: str | None) -> str:
    artifact = unwrap_model_answer(text)
    if not artifact:
        return ""
    if "diff --git a/" in artifact:
        first = artifact.index("diff --git a/")
        return artifact[first:].strip()
    if "--- a/" in artifact and "+++ b/" in artifact:
        return artifact.strip()
    matches = DIFF_RE.findall(artifact)
    if matches:
        return "\n".join(match.strip() for match in matches)
    return ""


def extract_python_or_text(text: str | None) -> str:
    return unwrap_model_answer(text).strip()


def normalize_instance_id(task_id: str, dataset: str) -> str:
    prefix = f"{dataset}/"
    if task_id.startswith(prefix):
        return task_id[len(prefix) :]
    return task_id
