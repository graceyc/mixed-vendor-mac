# mac_runner/mac_eval_adapter.py
import re
from utils.evaluation import diagnosis_evaluate
from llm_utils.api import Openai_api_handler

def _format_top10_for_eval(top10_list_or_text):
    """Return a DeepRare-style numbered list (string: '1. ...' .. '10. ...')."""
    if isinstance(top10_list_or_text, str):
        lines = re.findall(r"^\s*\d+[\).\s-]+(.+)$", top10_list_or_text, flags=re.MULTILINE)
        if not lines:
            lines = [ln.strip() for ln in top10_list_or_text.splitlines() if ln.strip()]
    else:
        lines = list(top10_list_or_text)
    lines = lines[:10]
    cleaned = []
    for i, d in enumerate(lines, 1):
        d = re.sub(r"\(.*?\)", "", d).strip()
        cleaned.append(f"{i}. {d}")
    while len(cleaned) < 10:
        cleaned.append(f"{len(cleaned)+1}. UNKNOWN")
    return "\n".join(cleaned)

def judge_with_deeprare(consensus_top10, golden_names_str, judge_label="gpt4"):
    """
    consensus_top10: list[str] or raw text from Supervisor
    golden_names_str: comma-separated gold names (from RareDataset patient tuple)
    judge_label: 'gpt4' or 'chatgpt' (DeepRare's labels; your api.py maps to Azure)
    """
    handler = Openai_api_handler(judge_label)  # uses your Azure env if set
    pred_str = _format_top10_for_eval(consensus_top10)
    rank = diagnosis_evaluate(pred_str, golden_names_str, handler)
    return pred_str, rank
