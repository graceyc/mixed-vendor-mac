# core/judge.py
from typing import List, Optional
from .llm_handlers import OpenAIChatHandler

# ===== Evaluation Prompt =====
JUDGE_PROMPT_TEMPLATE = (
    "You are an expert in diagnosing challenging cases. You will receive a student's answer containing 5 differential\n"
    "diagnoses, as well as the reference diagnosis. You need to score each diagnosis from the student's answer\n"
    "according to the following rules:\n"
    "2 = The student's diagnosis exactly matches the reference diagnosis;\n"
    "1 = The student's diagnosis is a broad category that includes the reference diagnosis;\n"
    "0 = The student's diagnosis does not meet the criteria for a score of 1 or 2.\n"
    "Here is the student's answer:\n"
    "{student}\n"
    "Here is the reference diagnosis:\n"
    "{gold}\n"
    "Output Format: Output the scores in the following format.\n"
    "1. Disease 1 name: score X;\n"
    "2. Disease 2 name: score X;\n"
    "..."
)

def judge_scores_o4mini(gold: str, preds: List[str], o4_deployment_or_model: Optional[str] = None) -> List[int]:
    """
    Uses the exact GPT-4o evaluation prompt. We render the student's 5 diagnoses as:
      1. Dx1;
      2. Dx2;
      ...
    and parse lines containing 'score X' (X in {0,1,2}). Only '2' counts as correct downstream.
    """
    # Render student's answer exactly as required (with trailing semicolons).
    student_block = "\n".join([f"{i+1}. {p};" for i, p in enumerate(preds, start=1)])
    prompt = JUDGE_PROMPT_TEMPLATE.format(student=student_block, gold=gold)

    judge = OpenAIChatHandler(model=o4_deployment_or_model or "gpt-4o-mini", role="judge")
    out = judge.get_completion(
        system_prompt="You are a precise medical evaluation judge. Follow the format exactly.",
        prompt=prompt,
        seed=42
    ) or ""

    # Parse 'score X' per line; accept optional trailing semicolon
    scores = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        low = ln.lower()
        if "score" in low:
            if "score 2" in low:
                scores.append(2)
            elif "score 1" in low:
                scores.append(1)
            elif "score 0" in low:
                scores.append(0)

    # Pad/truncate to number of predictions for robustness
    while len(scores) < len(preds):
        scores.append(0)
    return scores[:len(preds)]
