#!/usr/bin/env python3
# embed_eval_single.py
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

# ------------- HF embedding model config -------------
_HF_EMBED_MODEL: Optional[SentenceTransformer] = None


def embed_texts(client: Any, model: str, texts: List[str]) -> np.ndarray:
    """
    Embedding helper backed by a Hugging Face sentence-transformers model
    (e.g. FremyCompany/BioLORD-2023).

    `client` is ignored (kept only to avoid touching call sites).
    `model` should be a HF repo id or left empty to use BIOLORD_MODEL_NAME env var.
    """
    global _HF_EMBED_MODEL

    if not model:
        model = os.environ.get("BIOLORD_MODEL_NAME", "FremyCompany/BioLORD-2023")

    if _HF_EMBED_MODEL is None:
        _HF_EMBED_MODEL = SentenceTransformer(model)

    clean: List[str] = []
    for t in texts:
        s = "" if t is None else str(t).strip()
        if s:
            clean.append(s)

    dim = _HF_EMBED_MODEL.get_sentence_embedding_dimension()

    if not clean:
        return np.zeros((0, dim), dtype=np.float32)

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

# ------------- mapping / cache -------------
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

# ------------- parsing helpers -------------
_NUM_LINE = re.compile(r"^\s*\d+[\).\s-]+(.+)$", re.MULTILINE)


def _clean_top10(items_or_text) -> List[str]:
    if isinstance(items_or_text, list):
        lines = [str(x).strip() for x in items_or_text if str(x).strip()]
    else:
        txt = str(items_or_text or "")
        lines = _NUM_LINE.findall(txt)
        if not lines:
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    # strip parentheticals and de-dup
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


def _split_gold_names(gold: str) -> List[str]:
    parts = re.split(r"[,/]+", str(gold or ""))
    return [p.strip().lower() for p in parts if p.strip()]

# ------------- gold-code lookup -------------
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
        out.append(s)
    return out


def _extract_case_id(js: dict, pth: pathlib.Path, filename_regex: Optional[str]) -> Optional[int]:
    # 1) case_id field
    for key in ("case_id", "patient_id", "index", "Id", "ID"):
        if key in js and isinstance(js[key], (int, str)):
            try:
                return int(str(js[key]))
            except Exception:
                pass
    # 2) from filename pattern
    if filename_regex:
        m = re.search(filename_regex, pth.name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def _build_hf_gold_lookup(builder: str, subset: str, split: str) -> Dict[int, List[str]]:
    """
    Returns {index_in_split -> RareDisease (list[str])}
    """
    if not _HAS_DATASETS:
        raise RuntimeError("`datasets` not available. Install with `pip install datasets`.")
    ds = load_dataset(builder, subset, split=split)
    lookup: Dict[int, List[str]] = {}
    for i in range(len(ds)):
        row = ds[i]
        codes = _normalize_codes(row.get("RareDisease"))
        lookup[i] = codes
    return lookup

# ------------- rank & matching -------------
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
      best_sim_to_gold: max cosine(pred, any gold code name)
      pred_match_codes: nearest code in the universe per pred
      pred_match_sims:  corresponding cosine similarities
    """
    if not pred_top10:
        return "No", 0.0, [], []

    pred_mat = embed_texts(client, model, pred_top10)  # (k, d)
    if pred_mat.size == 0:
        return "No", 0.0, [], []
    sims = cosine_sim(pred_mat, code_mat)              # (k, N)

    nn_idx = sims.argmax(axis=1)          # (k,)
    nn_sim = sims.max(axis=1)             # (k,)
    pred_match_codes = [codes[j] for j in nn_idx]
    pred_match_sims = [float(x) for x in nn_sim]

    if gold_codes:
        code_index = {c: i for i, c in enumerate(codes)}
        gold_idx = [code_index[c] for c in gold_codes if c in code_index]
        if gold_idx:
            best_sim_to_gold = float(sims[:, gold_idx].max(axis=1).max())
        else:
            best_sim_to_gold = 0.0
    else:
        best_sim_to_gold = 0.0

    rank_str = "No"
    for i, (c, s) in enumerate(zip(pred_match_codes, pred_match_sims), 1):
        if (c in gold_codes) and (min_sim is None or s >= min_sim):
            rank_str = str(i)
            break

    return rank_str, best_sim_to_gold, pred_match_codes, pred_match_sims


def _recall_counts(ranks: List[Optional[int]], k: int) -> Tuple[int, int, float]:
    """
    ranks: list of rank (1..10) or None (miss).
    Returns (hits, total, recall).
    """
    total = len(ranks)
    if total == 0:
        return 0, 0, 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits, total, hits / total

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_dir", required=True, help="Folder with single-LLM *.json files")
    ap.add_argument("--disease_mapping", required=True, help="JSON {code: name}")
    ap.add_argument("--cache_dir", default=".cache/embed_eval_single")
    ap.add_argument("--min_sim", type=float, default=None, help="Optional cosine threshold")

    # HF embedding model
    ap.add_argument(
        "--hf_model",
        default="FremyCompany/BioLORD-2023",
        help="Hugging Face sentence-transformers model name (e.g. FremyCompany/BioLORD-2023)",
    )

    # Gold-code retrieval preferences
    ap.add_argument(
        "--prefer_json_fields",
        action="store_true",
        help=(
            "Prefer gold codes already present in each JSON (keys: RareDisease/gold_codes). "
            "Default: try JSON, else HF."
        ),
    )
    ap.add_argument("--hf_builder", default="chenxz/RareBench", help="HF dataset builder")
    ap.add_argument("--hf_subset", default="HMS", help="HF subset/config name")
    ap.add_argument("--hf_split", default="test", help="HF split name (train/validation/test)")
    ap.add_argument(
        "--filename_id_regex",
        default=r"patient_(\d+)\.json$",
        help="Regex to extract integer case id from filename when JSON lacks an id",
    )

    args = ap.parse_args()

    client = None
    model = args.hf_model

    codes, code_mat, name_to_codes = load_or_build_disease_index(
        args.disease_mapping, args.cache_dir, model, client
    )
    codes_set = set(codes)

    # HF lookup (optional)
    hf_lookup: Optional[Dict[int, List[str]]] = None
    if not args.prefer_json_fields:
        if not _HAS_DATASETS:
            print("[warn] datasets not installed; will only use JSON fields for gold codes.")
        else:
            try:
                hf_lookup = _build_hf_gold_lookup(args.hf_builder, args.hf_subset, args.hf_split)
            except Exception as e:
                print(f"[warn] HF lookup unavailable ({e}); will only use JSON fields.")

    files = sorted(pathlib.Path(args.judged_dir).glob("*.json"))

    # collect ranks for recall
    final_ranks: List[Optional[int]] = []

    for pth in files:
        try:
            with open(pth, "r", encoding="utf-8-sig") as f:
                js = json.load(f)
        except Exception as e:
            print(f"[skip] {pth.name}: {e}")
            continue

        # ----------- GOLD CODES -----------
        gold_codes: List[str] = []

        # 1) Prefer codes already in JSON
        jd_codes = js.get("RareDisease") or js.get("gold_codes")
        if jd_codes:
            gold_codes = _normalize_codes(jd_codes)

        # 2) HF lookup by case id
        if not gold_codes and hf_lookup:
            cid = _extract_case_id(js, pth, args.filename_id_regex)
            if cid is not None and cid in hf_lookup:
                gold_codes = _normalize_codes(hf_lookup[cid])

        # 3) Name->code mapping fallback
        if not gold_codes:
            gold_names = _split_gold_names(js.get("gold_names") or js.get("golden_diagnosis"))
            if gold_names:
                gold_codes = sorted(set().union(*(name_to_codes.get(n, set()) for n in gold_names)))

        gold_codes = [c for c in gold_codes if c in codes_set]
        gold_set: Set[str] = set(gold_codes)

        # ----------- PRED TOP-10 -----------
        pred_src = js.get("predict_top10") or js.get("predict_diagnosis")
        if not pred_src:
            continue
        preds = _clean_top10(pred_src)
        if not preds:
            continue

        # ----------- RANK & MATCH -----------
        rank, best_sim, pred_match_codes, pred_match_sims = _rank_and_match(
            preds, gold_set, codes, code_mat, client, model, args.min_sim
        )

        # ----------- WRITE BACK -----------
        js["embed_single_gold_codes"] = gold_codes
        js["embed_single_pred_top10"] = preds
        js["embed_single_pred_match_codes"] = pred_match_codes
        js["embed_single_pred_match_sims"] = [round(x, 4) for x in pred_match_sims]
        js["embed_single_best_sim_to_gold"] = round(best_sim, 4)
        js["embed_single_at_10"] = rank  # "1".."10" or "No"

        with open(pth, "w", encoding="utf-8-sig") as wf:
            json.dump(js, wf, indent=2, ensure_ascii=False)
        print(f"[updated] {pth.name}: rank={rank}, gold={gold_codes}")

        # collect for recall only if we have gold codes
        if gold_set:
            if rank == "No":
                final_ranks.append(None)
            else:
                try:
                    final_ranks.append(int(rank))
                except ValueError:
                    final_ranks.append(None)

    # ----------- overall recall -----------
    if final_ranks:
        for k in (1, 3, 5, 10):
            hits, total, r = _recall_counts(final_ranks, k)
            print(f"final R@{k}: {r:.4f} ({hits}/{total})")
    else:
        print("No cases with valid gold codes and predictions; recall not computed.")


if __name__ == "__main__":
    main()
