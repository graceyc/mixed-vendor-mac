# core/metrics.py
import re
from typing import List, Dict


def _clean_dx_name(name: str) -> str:
    """
    Post-process a raw line to trim explanations and trailing decorations.
    """
    name = name.strip()
    # accept trailing semicolons
    name = name.rstrip(";").strip()
    # drop explanations after dash/colon-style separators
    parts = re.split(r"\s[-\u2013\u2014:]\s", name, maxsplit=1)
    name = parts[0].strip()
    # drop trailing (...) parentheses
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return name


def parse_topk_predictions(text: str, k: int = 5) -> List[str]:
    """
    Parse up to k diagnosis strings from a free-form LLM output.

    Priority:
      1) Lines that look like numbered list items: '1. dx', '2) dx', etc.
      2) Fallback: non-empty lines, with obvious explanatory headers filtered out.

    This is meant to avoid extracting preamble sentences like
    "Based on the case information..." as diagnoses.
    """
    if not text:
        return []

    text = text.strip()
    preds: List[str] = []

    # ---- 1) Prefer explicit numbered list items: 1. dx, 2) dx, etc. ----
    numbered_preds: List[str] = []
    for ln in text.splitlines():
        ln = ln.rstrip()
        m = re.match(r"^\s*(?:[1-9]\d*|[\u2460-\u2469])[\.\)\:\-]+\s*(.+?)\s*$", ln)
        if not m:
            continue
        name = _clean_dx_name(m.group(1))
        if name:
            numbered_preds.append(name)
        if len(numbered_preds) >= k:
            break

    if numbered_preds:
        preds = numbered_preds
    else:
        # ---- 2) Fallback: line-based heuristic, but drop obvious preambles ----
        lines = [ln.strip() for ln in text.splitlines()]
        # remove empty / bullet-only lines
        lines = [ln.strip(" \t-\u2022") for ln in lines if ln.strip()]

        filtered: List[str] = []
        for ln in lines:
            low = ln.lower()
            # Heuristics to skip boilerplate or explanation headers
            if any(kw in low for kw in [
                "here are the top",
                "here is the top",
                "here are my top",
                "based on the case",
                "based on this case",
                "based on the information",
                "most likely diagnoses",
                "top 5 most likely",
                "top five most likely",
                "differential diagnoses",
                "differential diagnosis",
            ]):
                continue

            name = _clean_dx_name(ln)
            if name:
                filtered.append(name)

        preds = filtered

    # Deduplicate while preserving order
    seen = set()
    unique_preds: List[str] = []
    for p in preds:
        if p and p not in seen:
            seen.add(p)
            unique_preds.append(p)

    return unique_preds[:k]


def metrics_from_scores(scores: List[int]) -> Dict[str, float]:
    top1 = 1.0 if (len(scores) >= 1 and scores[0] == 2) else 0.0
    top5 = 1.0 if any(s == 2 for s in scores[:5]) else 0.0
    return {"top1_acc": top1, "top5_acc": top5}
