# ===================================
# File: mac_da/utils_extract.py
# ===================================
import re
from typing import List, Optional

# Accepts "1. ", "1) ", "1- " etc., captures the item text as group(2)
_NUMBERED_ITEM_RE = re.compile(r"^\s*(?:([1-9]|10)[\.\)\-])\s*(.+?)\s*$")

def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [ln.rstrip() for ln in text.split("\n")]

def extract_numbered_list(text: str, k: int = 5) -> List[str]:
    """
    Extract the LAST k numbered items (1.., 2.. etc.) from `text`, PRESERVING parentheses/brackets.
    Strategy:
      - Scan from the bottom; collect the last k lines that look like numbered items.
      - Keep original order (1..k ascending) for the return value.
      - We do NOT strip (...) / [...] content; only trim whitespace and a trailing semicolon if present.
      - If fewer than k exist overall, return whatever we found (or fall back to best contiguous block).
    """
    lines = _split_lines(text)
    matches_rev: List[str] = []

    # Bottom-up pass: take the last k numbered items
    for ln in reversed(lines):
        m = _NUMBERED_ITEM_RE.match(ln)
        if not m:
            continue
        item = m.group(2).strip()
        if item.endswith(";"):
            item = item[:-1].rstrip()
        if item:
            matches_rev.append(item)
            if len(matches_rev) >= k:
                break

    if matches_rev:
        return list(reversed(matches_rev))

    # Fallback: find best contiguous block (top-down), still preserving brackets
    blocks: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        m = _NUMBERED_ITEM_RE.match(ln)
        if m:
            val = m.group(2).strip()
            if val.endswith(";"):
                val = val[:-1].rstrip()
            if val:
                cur.append(val)
        else:
            if len(cur) >= 2:
                blocks.append(cur[:])
            cur = []
    if len(cur) >= 2:
        blocks.append(cur)

    if blocks:
        # Choose the last block (closest to bottom) to align with "last k bullet points"
        best = blocks[-1]
        return best[-k:] if len(best) > k else best

    return []

def _messages_with_name(history, name_prefix: str):
    return [m for m in history if (m.get("name") or "").lower().startswith(name_prefix.lower())]

def parse_consensus_topk(chat_history: List[dict], k: int = 5) -> List[str]:
    """
    Prefer the latest Supervisor message that contains numbered items; otherwise fall back to any message.
    Uses `extract_numbered_list` which returns the LAST k items within a message.
    """
    # Supervisor first (bottom-up across messages)
    for m in reversed(_messages_with_name(chat_history, "supervisor")):
        preds = extract_numbered_list(m.get("content", ""), k=k)
        if len(preds) >= min(3, k):
            return preds[:k]

    # Fallback: any message
    for m in reversed(chat_history):
        preds = extract_numbered_list(m.get("content", ""), k=k)
        if len(preds) >= min(3, k):
            return preds[:k]

    return []

def force_supervisor_finalize_list(supervisor_agent, k: int = 5) -> Optional[List[str]]:
    """
    Ask the supervisor to finalize now with exactly k lines, then parse with the same extractor.
    """
    prompt = (
        f"Finalize now. Reply ONLY with a numbered list of exactly {k} diagnoses, one per line, "
        f"formatted '1. ...' through '{k}. ...'. Then append TERMINATE."
    )
    try:
        reply = supervisor_agent.generate_reply([{"role": "user", "content": prompt}])
        text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
        preds = extract_numbered_list(text, k=k)
        return preds if len(preds) == k else None
    except Exception:
        return None
