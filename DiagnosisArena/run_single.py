# run_single.py
#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from core.data_loading import load_da_2024
from core.prompts import PROMPT_TEMPLATE
from core.llm_handlers import get_handler
from core.judge import judge_scores_o4mini
from core.metrics import parse_topk_predictions, metrics_from_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--provider", type=str, choices=["openai","anthropic","gemini"], required=True)
    ap.add_argument("--model", type=str, required=True, help="Raw model id or Azure deployment name")
    ap.add_argument("--o4mini_judge_model", type=str, default=None, help="Model/deployment for the judge (o4-mini)")
    ap.add_argument("--system", type=str, default="You are a specialist in rare diseases. Focus ONLY on diagnosis names.")
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, recompute even when a per-case JSON already exists.")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_cases = out_root / "cases"
    out_root.mkdir(parents=True, exist_ok=True)
    out_cases.mkdir(parents=True, exist_ok=True)

    recs = load_da_2024(Path(args.jsonl))
    print(f"Loaded {len(recs)} DiagnosisArena cases for year 2024.")

    h = get_handler(args.provider, model=args.model)
    rows = []

    skipped = 0

    for rec in tqdm(recs, desc="Open-ended Top-5"):
        case_path = out_cases / f"da2024_id{rec.id}.json"
        if case_path.exists() and not args.overwrite:
            skipped += 1
            continue

        user_prompt = PROMPT_TEMPLATE.format(
            case=rec.case_info or "",
            exam=rec.phys_exam or "",
            tests=rec.tests or ""
        )
        pred_text = h.get_completion(args.system, user_prompt) or ""
        preds = parse_topk_predictions(pred_text, k=5)

        # Judge with o4-mini 0/1/2 => metrics from '2'
        scores = judge_scores_o4mini(rec.final_dx, preds, o4_deployment_or_model=args.o4mini_judge_model)
        m = metrics_from_scores(scores)

        case_obj = {
            "mode": "open",
            "id": rec.id,
            "date": rec.date,
            "gold": rec.final_dx,
            "predictions": preds,
            "judge_scores_0_1_2": scores,
            **m
        }
        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(case_obj, f, indent=2, ensure_ascii=False)

        rows.append({
            "id": rec.id,
            "date": rec.date,
            "gold": rec.final_dx,
            "pred_1": preds[0] if len(preds)>0 else "",
            "pred_2": preds[1] if len(preds)>1 else "",
            "pred_3": preds[2] if len(preds)>2 else "",
            "pred_4": preds[3] if len(preds)>3 else "",
            "pred_5": preds[4] if len(preds)>4 else "",
            "score_1": scores[0] if len(scores)>0 else "",
            "score_2": scores[1] if len(scores)>1 else "",
            "score_3": scores[2] if len(scores)>2 else "",
            "score_4": scores[3] if len(scores)>3 else "",
            "score_5": scores[4] if len(scores)>4 else "",
            "top1_acc": m["top1_acc"],
            "top5_acc": m["top5_acc"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False)
    agg = {
        "N": len(rows),
        "top1_acc": float(df["top1_acc"].mean()) if len(rows) else 0.0,
        "top5_acc": float(df["top5_acc"].mean()) if len(rows) else 0.0,
        "skipped_existing": int(skipped),
    }
    with open(out_root / "metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
