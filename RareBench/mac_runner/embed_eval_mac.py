#!/usr/bin/env python3
# mac_runner/embed_eval_per_round.py
import os, json, re, pathlib, argparse, pickle, hashlib
from typing import List, Dict, Optional, Tuple, Set, Any
import numpy as np

from sentence_transformers import SentenceTransformer

# Optional HF import (only used if needed to fetch gold codes)
try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

# -------------------- HF embedding model config --------------------
_HF_EMBED_MODEL: Optional[SentenceTransformer] = None


def embed_texts(client: Any, model: str, texts: List[str]) -> np.ndarray:
    """
    Embedding helper backed by a Hugging Face sentence-transformers model
    (e.g. FremyCompany/BioLORD-2023).

    `client` is ignored (kept only to avoid touching call sites).
    `model` should be a HF repo id or left empty to use BIOLORD_MODEL_NAME env var.
    """
    global _HF_EMBED_MODEL

    # Decide which HF model to use
    if not model:
        model = os.environ.get("BIOLORD_MODEL_NAME", "FremyCompany/BioLORD-2023")

    # Lazily load the sentence-transformers model once
    if _HF_EMBED_MODEL is None:
        _HF_EMBED_MODEL = SentenceTransformer(model)

    # Clean inputs
    clean: List[str] = []
    for t in texts:
        s = "" if t is None else str(t)
        s = s.strip()
        if s:
            clean.append(s)

    dim = _HF_EMBED_MODEL.get_sentence_embedding_dimension()

    # If nothing left, return an empty (0, dim) matrix
    if not clean:
        return np.zeros((0, dim), dtype=np.float32)

    # Encode with batching; returns (len(clean), dim)
    emb = _HF_EMBED_MODEL.encode(
        clean,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,  # we normalize in cosine_sim
    )

    return emb.astype(np.float32, copy=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


# -------------------- mapping / cache --------------------
def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _invert_name_to_codes(code_to_name: Dict[str, str]) -> Dict[str, set]:
    by_name: Dict[str, set] = {}
    for code, name in code_to_name.items():
        by_name.setdefault(name.strip().lower(), set()).add(code)
    return by_name


def load_or_build_disease_index(
    mapping_path: str,
    cache_dir: str,
    model: str,
    client: Any,
) -> Tuple[List[str], np.ndarray, Dict[str, set]]:
    cache_dir_path = pathlib.Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    with open(mapping_path, "r", encoding="utf-8-sig") as f:
        code_to_name: Dict[str, str] = json.load(f)

    name_to_codes = _invert_name_to_codes(code_to_name)
    st = os.stat(mapping_path)
    ident = f"{_h(model)}_{st.st_size}_{int(st.st_mtime)}"
    cache_file = cache_dir_path / f"disease_embed_{ident}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as pf:
            blob = pickle.load(pf)
        return blob["codes"], blob["mat"], name_to_codes

    codes = sorted(code_to_name.keys())
    names = [code_to_name[c] for c in codes]
    mat = embed_texts(client, model, names)
    with open(cache_file, "wb") as pf:
        pickle.dump({"codes": codes, "mat": mat}, pf)
    return codes, mat, name_to_codes


# -------------------- parsing & helpers --------------------
_NUM_LINE = re.compile(r"^\s*\d+[\).\s-]+(.+)$", re.MULTILINE)


def _clean_top10(items_or_text) -> List[str]:
    """
    Accepts either:
      - list[str]: already tokenized candidates
      - str: numbered / bulleted lines or freeform lines
    Returns up to 10 cleaned labels, parentheticals removed + deduped.
    """
    if isinstance(items_or_text, list):
        lines = [str(x).strip() for x in items_or_text if str(x).strip()]
    else:
        txt = str(items_or_text or "")
        lines = _NUM_LINE.findall(txt)
        if not lines:
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    lines = [re.sub(r"\(.*?\)", "", d).strip() for d in lines]
    seen, out = set(), []
    for x in lines:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
        if len(out) == 10:
            break
    return out


def _normalize_codes(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[,\s;/]+", str(raw))
    out = []
    for s in items:
        s = s.strip()
        if not s:
            continue
        out.append(s)  # keep namespace (e.g., OMIM:, ORPHA:)
    return out


def _extract_case_index_from_id(case_id: str) -> Optional[int]:
    """
    Try to extract an integer index from strings like 'HMS-0', 'MME-12', 'patient_7', etc.
    """
    if not case_id:
        return None
    m = re.search(r"(\d+)", str(case_id))
    return int(m.group(1)) if m else None


def _build_hf_gold_lookup(
    builder: str,
    subset: str,
    split: str,
) -> Dict[int, List[str]]:
    """
    Returns {index_in_split -> RareDisease (list[str])}
    """
    if not _HAS_DATASETS:
        raise RuntimeError("`datasets` not available. Install with `pip install datasets`.")
    ds = load_dataset(builder, subset, split=split)
    lookup: Dict[int, List[str]] = {}
    for i in range(len(ds)):
        codes = _normalize_codes(ds[i].get("RareDisease"))
        lookup[i] = codes
    return lookup


# -------- rank & matching on codes (gold codes provided) --------
def _rank_and_match(
    pred_top10: List[str],
    gold_codes: Set[str],
    codes: List[str],
    code_mat: np.ndarray,
    client: Any,
    model: str,
    min_sim: Optional[float] = None,
) -> Tuple[str, float, List[str], List[float]]:
    """
    Returns:
      rank: "1".."10" or "No"
      best_sim_to_gold: max cosine(pred, any gold code-name in index)
      pred_match_codes: for each prediction, the nearest code in the universe
      pred_match_sims:  corresponding cosine similarities
    """
    if not pred_top10:
        return "No", 0.0, [], []
    pred_mat = embed_texts(client, model, pred_top10)
    if pred_mat.size == 0:
        return "No", 0.0, [], []

    sims = cosine_sim(pred_mat, code_mat)  # (k, N)
    nn_idx = sims.argmax(axis=1)          # nearest code per pred
    nn_sim = sims.max(axis=1)

    pred_match_codes = [codes[j] for j in nn_idx]
    pred_match_sims = [float(x) for x in nn_sim]

    # best similarity to any GOLD code (reporting)
    if gold_codes:
        index = {c: i for i, c in enumerate(codes)}
        gold_idx = [index[c] for c in gold_codes if c in index]
        best_sim_to_gold = float(sims[:, gold_idx].max(axis=1).max()) if gold_idx else 0.0
    else:
        best_sim_to_gold = 0.0

    # rank: first prediction whose NN lands in gold set (and passes threshold if given)
    rank_str = "No"
    for i, (c, s) in enumerate(zip(pred_match_codes, pred_match_sims), 1):
        if (c in gold_codes) and (min_sim is None or s >= min_sim):
            rank_str = str(i)
            break
    return rank_str, best_sim_to_gold, pred_match_codes, pred_match_sims


def _recall_counts(ranks: List[Optional[int]], k: int) -> Tuple[int, int, float]:
    """
    ranks: list of int rank (1..10) or None (miss), one per evaluated case.
    Returns (hits, total, recall).
    """
    total = len(ranks)
    if total == 0:
        return 0, 0, 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits, total, hits / total


# -------------------- driver --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_dir", required=True, help="Folder with *.json case files")
    ap.add_argument(
        "--disease_mapping",
        default="mapping/disease_mapping.json",
        help="JSON mapping {code: name} used to build the embedding index",
    )
    ap.add_argument(
        "--cache_dir",
        default=".cache/embed_eval",
        help="Where to cache the disease embedding matrix",
    )
    ap.add_argument(
        "--min_sim",
        type=float,
        default=None,
        help="Optional cosine threshold; omit to apply NO threshold",
    )

    # HF embedding model
    ap.add_argument(
        "--hf_model",
        default="FremyCompany/BioLORD-2023",
        help="Hugging Face sentence-transformers model name (e.g. FremyCompany/BioLORD-2023)",
    )

    # Gold-code retrieval preferences (JSON first; optional HF fallback)
    ap.add_argument(
        "--prefer_json_fields",
        action="store_true",
        help=(
            "Prefer gold codes already present in each JSON (keys: RareDisease/gold_codes). "
            "Default: try JSON, else HF fallback."
        ),
    )
    ap.add_argument(
        "--hf_builder",
        default="chenxz/RareBench",
        help="HF dataset builder",
    )
    ap.add_argument(
        "--hf_subset",
        default=None,
        help="HF subset/config (default: use per-file 'dataset' if present)",
    )
    ap.add_argument(
        "--hf_split",
        default="test",
        help="HF split (train/validation/test)",
    )
    args = ap.parse_args()

    # We no longer use Azure; keep `client` for compatibility with embed_texts signature.
    client = None
    model = args.hf_model

    codes, code_mat, name_to_codes = load_or_build_disease_index(
        args.disease_mapping, args.cache_dir, model, client
    )
    codes_set = set(codes)

    # Lazy HF caches per subset+split: {(subset, split) -> lookup dict}
    hf_cache: Dict[Tuple[str, str], Dict[int, List[str]]] = {}

    # For supervisor-level metrics
    supervisor_ranks: List[Optional[int]] = []

    files = sorted(pathlib.Path(args.judged_dir).glob("*.json"))
    for pth in files:
        try:
            with open(pth, "r", encoding="utf-8-sig") as f:
                js = json.load(f)
        except Exception as e:
            print(f"[skip] {pth.name}: {e}")
            continue

        # ---------------- GOLD CODES (prefer JSON, else HF, else name fallback) ----------------
        gold_codes: List[str] = []

        # 1) JSON field(s)
        jd_codes = js.get("RareDisease") or js.get("gold_codes")
        if jd_codes:
            gold_codes = _normalize_codes(jd_codes)

        # 2) HF fallback (if needed)
        if not gold_codes and not args.prefer_json_fields and _HAS_DATASETS:
            subset = (args.hf_subset or js.get("dataset") or "").strip() or "HMS"
            key = (subset, args.hf_split)
            if key not in hf_cache:
                try:
                    hf_cache[key] = _build_hf_gold_lookup(args.hf_builder, subset, args.hf_split)
                except Exception as e:
                    print(f"[warn] HF lookup unavailable for {key}: {e}")
                    hf_cache[key] = {}
            # map file's case_id -> index
            cid = js.get("case_id") or js.get("patient_id") or pth.stem  # e.g., 'HMS-0'
            idx = _extract_case_index_from_id(str(cid))
            if idx is not None and idx in hf_cache[key]:
                gold_codes = _normalize_codes(hf_cache[key][idx])

        # 3) Last resort: name->code via mapping (from gold_names/golden_diagnosis)
        if not gold_codes:
            gold_names_str = js.get("gold_names") or js.get("golden_diagnosis") or ""
            names = [x.strip().lower() for x in re.split(r"[,/]+", gold_names_str) if x.strip()]
            gold_codes = sorted(set().union(*(name_to_codes.get(n, set()) for n in names)))

        # keep only codes that exist in our index
        gold_codes = [c for c in gold_codes if c in codes_set]
        gold_set: Set[str] = set(gold_codes)

        # -------- Per-round doctor predictions --------
        rounds = js.get("per_round_metrics") or []
        changed = False
        for entry in rounds:
            preds = entry.get("predictions")
            if preds is None:
                continue
            pred = _clean_top10(preds)
            rank, best_sim, match_codes, match_sims = _rank_and_match(
                pred, gold_set, codes, code_mat, client, model, args.min_sim
            )
            # Replace old embed fields with new ones
            entry["embed_gold_codes"] = gold_codes
            entry["embed_pred_top10"] = pred
            entry["embed_pred_match_codes"] = match_codes
            entry["embed_pred_match_sims"] = [round(x, 4) for x in match_sims]
            entry["embed_best_sim_to_gold"] = round(best_sim, 4)
            entry["embed_judge_at_10"] = rank
            changed = True

        # -------- Supervisor final top-10 (list or numbered string) --------
        sup_src = js.get("supervisor_consensus_top10_numbered") or js.get(
            "supervisor_consensus_top10"
        )
        if sup_src:
            sup_list = _clean_top10(sup_src)
            s_rank, s_best, s_codes, s_sims = _rank_and_match(
                sup_list, gold_set, codes, code_mat, client, model, args.min_sim
            )
            js["embed_supervisor_gold_codes"] = gold_codes
            js["embed_supervisor_pred_top10"] = sup_list
            js["embed_supervisor_pred_match_codes"] = s_codes
            js["embed_supervisor_pred_match_sims"] = [round(x, 4) for x in s_sims]
            js["embed_supervisor_best_sim_to_gold"] = round(s_best, 4)
            js["embed_supervisor_judge_at_10"] = s_rank
            changed = True

            # Only evaluate recall if we actually have gold codes
            if gold_set:
                if s_rank == "No":
                    supervisor_ranks.append(None)
                else:
                    try:
                        supervisor_ranks.append(int(s_rank))
                    except ValueError:
                        supervisor_ranks.append(None)

        if changed:
            with open(pth, "w", encoding="utf-8-sig") as wf:
                json.dump(js, wf, indent=2, ensure_ascii=False)
            print(f"[updated] {pth.name}")

    # -------------------- overall supervisor recall --------------------
    if supervisor_ranks:
        for k in (1, 3, 5, 10):
            hits, total, r = _recall_counts(supervisor_ranks, k)
            print(f"supervisor R@{k}: {r:.4f} ({hits}/{total})")
    else:
        print("No supervisor predictions with gold codes found; recall not computed.")


if __name__ == "__main__":
    main()
