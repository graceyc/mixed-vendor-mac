#!/usr/bin/env python3
import os, re, json, csv, argparse
from glob import glob
from statistics import median
from collections import defaultdict

def parse_rank(raw):
    """Return 1..10 if present, else 11 (miss). Accepts '6', 'No', None, etc."""
    if raw is None:
        return 11
    s = str(raw).strip()
    if s.lower() in {"no", "none", "na"}:
        return 11
    m = re.search(r'\b(10|[1-9])\b', s)
    return int(m.group(1)) if m else 11

def r_at_k(rank, k):
    return 1.0 if rank <= k else 0.0

def main():
    ap = argparse.ArgumentParser(
        description="Summarize avg Recall@K per dataset from <ROOT>/<DATASET>/<MODEL_DIR>/patient_*.json"
    )
    ap.add_argument("--root", required=True,
                    help="Directory that directly contains per-dataset folders (e.g., results_single_o4)")
    ap.add_argument("--datasets", nargs="+", default=["HMS","MME","LIRICAL","RAMEDIS"])
    ap.add_argument("--model_dir", default="gpt4_diagnosis",
                    help="Subfolder under each dataset (e.g., chatgpt_diagnosis or chatgpt_diagnosis_cot)")
    ap.add_argument("--csv_out", default="results_summary.csv")
    args = ap.parse_args()

    per_ds_ranks = defaultdict(list)

    for ds in args.datasets:
        # CHANGED: no hardcoded "results" segment
        patt = os.path.join(args.root, ds, args.model_dir, "patient_*.json")
        files = sorted(glob(patt))
        if not files:
            print(f"[WARN] No files for {ds}: {patt}")
            continue

        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8-sig") as f:
                    rec = json.load(f)
            except Exception as e:
                print(f"[WARN] Skipping {fp}: {e}")
                continue
            per_ds_ranks[ds].append(parse_rank(rec.get("predict_rank")))

    header = f"{'DATASET':12} {'N':>6} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MED_R':>8}"
    print(f"\n=== Single-LLM results summary (from `{args.root}/<DATASET>/{args.model_dir}`) ===")
    print(header)
    print("-" * len(header))

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset","n","recall@1","recall@3","recall@5","recall@10","median_rank"])

        all_ranks = []
        for ds in args.datasets:
            ranks = per_ds_ranks.get(ds, [])
            n = len(ranks)
            if n == 0:
                print(f"{ds:12} {0:6d} {0.000:8.3f} {0.000:8.3f} {0.000:8.3f} {0.000:8.3f} {0:8.3f}")
                w.writerow([ds, 0, "0.0000","0.0000","0.0000","0.0000","0.0000"])
                continue

            r1  = sum(r_at_k(r,1)  for r in ranks)/n
            r3  = sum(r_at_k(r,3)  for r in ranks)/n
            r5  = sum(r_at_k(r,5)  for r in ranks)/n
            r10 = sum(r_at_k(r,10) for r in ranks)/n
            med = median(ranks)

            all_ranks.extend(ranks)
            print(f"{ds:12} {n:6d} {r1:8.3f} {r3:8.3f} {r5:8.3f} {r10:8.3f} {med:8.3f}")
            w.writerow([ds, n, f"{r1:.4f}", f"{r3:.4f}", f"{r5:.4f}", f"{r10:.4f}", f"{med:.4f}"])

        if all_ranks:
            N = len(all_ranks)
            overall = (
                sum(r_at_k(r,1)  for r in all_ranks)/N,
                sum(r_at_k(r,3)  for r in all_ranks)/N,
                sum(r_at_k(r,5)  for r in all_ranks)/N,
                sum(r_at_k(r,10) for r in all_ranks)/N,
                median(all_ranks),
            )
            print("-" * len(header))
            print(f"{'OVERALL':12} {N:6d} {overall[0]:8.3f} {overall[1]:8.3f} {overall[2]:8.3f} {overall[3]:8.3f} {overall[4]:8.3f}")
            w.writerow(["OVERALL", N, *(f"{x:.4f}" for x in overall[:4]), f"{overall[4]:.4f}"])

    print(f"\nSaved CSV → {args.csv_out}")

if __name__ == "__main__":
    main()
