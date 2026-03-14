# main.py
from llm_utils.api import Openai_api_handler, Zhipuai_api_handler, Gemini_api_handler
# from llm_utils.local_llm import Local_llm_handler
import argparse
from utils.mydataset import RareDataset
from utils.evaluation import diagnosis_evaluate
import os
from prompt import RarePrompt
import json
import numpy as np
import re
from typing import Optional  # Python 3.9-friendly Optional[...] instead of `| None`
from llm_utils.unified import get_handler


def _is_azure():
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


def diagnosis_metric_calculate(folder, judge_model="chatgpt"):
    """
    Judge previously generated predictions. On Azure, force the judge to use
    AZURE_OPENAI_EVAL_DEPLOYMENT via model label 'judge'. Off Azure, use the passed label.
    """
    from llm_utils.api import Openai_api_handler  # local import to avoid extra deps

    # Choose judge handler
    if _is_azure():
        handler = Openai_api_handler("judge")
    else:
        handler = Openai_api_handler(judge_model)

    CNT = 0
    metric = {}
    recall_top_k = []

    import re, json, os

    def _norm_rank(s: str):
        """Return '1'..'10' or None (meaning 'No' / unrecognized)."""
        if not isinstance(s, str):
            return None
        t = s.strip()
        if t.lower() in {"no", "否"}:
            return None
        # extract first 1..10
        m = re.findall(r"\b(10|[1-9])\b", t)
        return m[0] if m else None

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    files.sort()  # deterministic ordering for evaluation/io

    for file in files:
        # Read JSON safely
        try:
            with open(file, "r", encoding="utf-8-sig") as f:
                res = json.load(f)
        except Exception as e:
            print(file, "JSON parse error:", e)
            CNT += 1
            continue

        predict_rank = res.get("predict_rank")
        pred = res.get("predict_diagnosis")
        gold = res.get("golden_diagnosis")

        if pred is None or gold is None:
            print(file, "missing predict_diagnosis or golden_diagnosis")
            CNT += 1
            continue

        # If not judged yet, call judge now
        if predict_rank is None:
            jr = handler.get_completion(
                "You are a specialist in the field of rare diseases.",
                (
                    "I will now give you ten predicted diseases and the standard diagnosis.\n"
                    "If ANY predicted disease clearly matches ANY standard diagnosis (including obvious aliases), "
                    'answer with ONLY the best rank index as a number 1..10. Otherwise answer ONLY "No".\n'
                    "Do not include any extra text.\n\n"
                    f"Predicted diseases: {pred}\n"
                    f"Standard diagnosis: {gold}\n"
                ),
            )
            if jr is None:
                print(file, "judge returned None")
                CNT += 1
                res["predict_rank"] = None
            else:
                jr = jr.replace("\n", "").strip()
                nr = _norm_rank(jr)
                if nr is None:
                    # treat as no hit
                    res["predict_rank"] = None
                    recall_top_k.append(11)
                else:
                    res["predict_rank"] = nr
                    recall_top_k.append(int(nr))

            # write back result
            with open(file, "w", encoding="utf-8-sig") as wf:
                json.dump(res, wf, indent=4, ensure_ascii=False)

        else:
            # Already judged; normalize it and accumulate
            nr = _norm_rank(str(predict_rank))
            if nr is None:
                recall_top_k.append(11)
            else:
                recall_top_k.append(int(nr))

    # Aggregate metrics
    if recall_top_k:
        metric["recall_top_1"] = len([i for i in recall_top_k if i <= 1]) / len(recall_top_k)
        metric["recall_top_3"] = len([i for i in recall_top_k if i <= 3]) / len(recall_top_k)
        metric["recall_top_10"] = len([i for i in recall_top_k if i <= 10]) / len(recall_top_k)
        import statistics

        metric["median_rank"] = statistics.median(recall_top_k)
    else:
        metric["recall_top_1"] = metric["recall_top_3"] = metric["recall_top_10"] = 0.0
        metric["median_rank"] = None

    print(folder)
    print(metric)
    print("files skipped/errors: ", CNT)

    # Prefer total_tokens if available; keep legacy prints for compatibility
    if hasattr(handler, "total_tokens"):
        print("evaluate total tokens:", handler.total_tokens)
    else:
        print(
            "evaluate tokens: ",
            handler.gpt4_tokens,
            handler.chatgpt_tokens,
            handler.chatgpt_instruct_tokens,
        )


def generate_random_few_shot_id(
    exclude_id, total_num, k_shot=3, rng: Optional[np.random.Generator] = None
):
    """
    Randomly pick k_shot distinct indices in [0, total_num), excluding any in exclude_id.
    Uses a NumPy Generator. If rng is None, a fresh non-deterministic Generator is used.
    NOTE: This is defined for completeness; it is not used when --few_shot none.
    """
    if rng is None:
        rng = np.random.default_rng()  # non-deterministic by default
    exclude_set = set(exclude_id)
    few_shot_id = []
    while len(few_shot_id) < k_shot:
        idx = int(rng.integers(0, total_num))
        if idx not in exclude_set and idx not in few_shot_id:
            few_shot_id.append(idx)
    return few_shot_id


def generate_dynamic_few_shot_id(methods, exclude_id, dataset, k_shot=3):
    """
    Deterministic similarity-based selection; no RNG needed.
    NOTE: This is defined for completeness; it is not used when --few_shot none.
    """
    few_shot_id = []

    patient = dataset.load_hpo_code_data()
    if methods == "dynamic":
        phe2embedding = json.load(
            open("mapping/phe2embedding.json", "r", encoding="utf-8-sig")
        )
    elif methods == "medprompt":
        phe2embedding = json.load(
            open("mapping/medprompt_emb.json", "r", encoding="utf-8-sig")
        )
    ic_dict = json.load(open("mapping/ic_dict.json", "r", encoding="utf-8-sig"))
    if methods == "medprompt":
        ic_dict = {k: 1 for k, _ in ic_dict.items()}

    exclude_patient = patient[exclude_id]
    exclude_patient_embedding = np.array(
        [np.array(phe2embedding[phe]) for phe in exclude_patient[0] if phe in phe2embedding]
    )
    exclude_patient_ic = np.array(
        [ic_dict[phe] for phe in exclude_patient[0] if phe in phe2embedding]
    )
    exclude_patient_embedding = np.sum(
        exclude_patient_embedding * exclude_patient_ic.reshape(-1, 1), axis=0
    ) / np.sum(exclude_patient_ic)
    candidata_embedding_list = []
    for i, p in enumerate(patient):
        phe_embedding = np.array(
            [np.array(phe2embedding[phe]) for phe in p[0] if phe in phe2embedding]
        )
        ic_coefficient_list = np.array(
            [ic_dict[phe] for phe in p[0] if phe in phe2embedding]
        )
        phe_embedding = np.sum(
            phe_embedding * ic_coefficient_list.reshape(-1, 1), axis=0
        ) / np.sum(ic_coefficient_list)
        candidata_embedding_list.append(phe_embedding)
    candidata_embedding_list = np.array(candidata_embedding_list)
    cosine_sim = np.dot(candidata_embedding_list, exclude_patient_embedding)
    cosine_sim = np.argsort(cosine_sim)[::-1]
    for i in cosine_sim:
        if i not in few_shot_id and i != exclude_id:
            few_shot_id.append(i)
        if len(few_shot_id) == k_shot:
            break

    return few_shot_id


def run_task(
    task_type,
    dataset: RareDataset,
    handler,
    results_folder,
    few_shot,
    cot,
    judge_model,
    eval=False,
    rng: Optional[np.random.Generator] = None,
    num_shards: int = 1,
    shard_id: int = 0,
):
    few_shot_dict = {}
    rare_prompt = RarePrompt()
    if task_type == "diagnosis":
        patient_info_type = dataset.dataset_type
        os.makedirs(results_folder, exist_ok=True)
        print("Begin diagnosis.....")
        print("total patient: ", len(dataset.patient))
        if num_shards > 1:
            print(f"Sharding enabled: num_shards={num_shards}, shard_id={shard_id}")
        ERR_CNT = 0
        questions = []
        for i, patient in enumerate(dataset.patient):
            # Sharding: only handle indices assigned to this shard
            if num_shards > 1 and (i % num_shards) != shard_id:
                continue

            if handler is None:
                print("handler is None")
                break
            result_file = os.path.join(results_folder, f"patient_{i}.json")
            if os.path.exists(result_file):
                continue
            patient_info = patient[0]
            golden_diagnosis = patient[1]
            few_shot_info = []
            # No few-shot is added when --few_shot none
            if few_shot == "random":
                few_shot_id = generate_random_few_shot_id(
                    [i], len(dataset.patient), rng=rng
                )
                few_shot_dict[i] = few_shot_id
                for idx in few_shot_id:
                    few_shot_info.append(
                        (dataset.patient[idx][0], dataset.patient[idx][1])
                    )
            elif few_shot == "dynamic" or few_shot == "medprompt":
                few_shot_id = generate_dynamic_few_shot_id(few_shot, i, dataset)
                few_shot_dict[str(i)] = [str(idx) for idx in few_shot_id]
                for idx in few_shot_id:
                    few_shot_info.append(
                        (dataset.patient[idx][0], dataset.patient[idx][1])
                    )

            system_prompt, prompt = rare_prompt.diagnosis_prompt(
                patient_info_type, patient_info, cot, few_shot_info
            )

            questions.append(system_prompt + prompt)
            if few_shot == "auto-cot":
                autocot_example = json.load(
                    open("mapping/autocot_example.json", "r", encoding="utf-8-sig")
                )
                # Only used if you actually set --few_shot auto-cot
                system_prompt = (
                    "Here a some examples: "
                    + autocot_example.get(getattr(handler, "model_name", ""), "")
                    + system_prompt
                )
                prompt = prompt + "Let us think step by step.\n"

            predict_diagnosis = handler.get_completion(system_prompt, prompt)
            if predict_diagnosis is None:
                print(f"patient {i} predict diagnosis is None")
                ERR_CNT += 1
                continue

            predict_rank = None
            res = {
                "patient_info": patient_info,
                "golden_diagnosis": golden_diagnosis,
                "predict_diagnosis": predict_diagnosis,
                "predict_rank": predict_rank,
            }
            json.dump(
                res,
                open(result_file, "w", encoding="utf-8-sig"),
                indent=4,
                ensure_ascii=False,
            )
            print(f"patient {i} finished")

            # Prefer total_tokens if available
            if hasattr(handler, "total_tokens"):
                print("total tokens so far:", handler.total_tokens)
            else:
                if type(handler) == Openai_api_handler:
                    print(
                        "total tokens (legacy counters): ",
                        handler.gpt4_tokens,
                        handler.chatgpt_tokens,
                        handler.chatgpt_instruct_tokens,
                    )

        if eval:
            # On Azure force 'judge' (EVAL deployment); otherwise keep user choice
            jm = "judge" if _is_azure() else judge_model
            diagnosis_metric_calculate(results_folder, judge_model=jm)
        print("diagnosis ERR_CNT: ", ERR_CNT)
    elif task_type == "mdt":
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type", type=str, default="diagnosis", choices=["diagnosis", "mdt"]
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PUMCH_ADM",
        choices=["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"],
    )
    parser.add_argument(
        "--dataset_type", type=str, default="PHENOTYPE", choices=["EHR", "PHENOTYPE", "MDT"]
    )
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--results_folder", default="./results/PUMCH")

    # NEW: provider-first, model passthrough
    parser.add_argument(
        "--provider", type=str, default=None, choices=["openai", "anthropic", "gemini"]
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Raw model id or deployment name. Examples: gpt-4o, claude-3-5-sonnet-latest, gemini-1.5-pro. "
            "If omitted, will use provider defaults or env vars."
        ),
    )
    # Back-compat judge (we keep judge on OpenAI by default for now)
    parser.add_argument(
        "--judge_provider", type=str, default="openai", choices=["openai"]
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help=(
            "OpenAI judge model id or Azure deployment name; defaults to AZURE eval deployment or gpt-4o-mini."
        ),
    )
    parser.add_argument(
        "--few_shot",
        type=str,
        default="none",
        choices=["none", "random", "dynamic", "medprompt", "auto-cot"],
    )
    parser.add_argument(
        "--cot", type=str, default="none", choices=["none", "zero-shot"]
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (omit for non-deterministic runs)"
    )

    # Sharding args
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of parallel shards (processes).",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="This process's shard id in [0, num_shards-1].",
    )

    args = parser.parse_args()

    # Basic sanity for sharding
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(
            f"shard_id must be in [0, num_shards-1], got {args.shard_id} with num_shards={args.num_shards}"
        )

    rng = np.random.default_rng(args.seed)

    # ---- Resolve provider/model from legacy flags for back-compat ----
    legacy_to_provider: dict[str, tuple[str, Optional[str]]] = {
        "gpt4": ("openai", "gpt-4o"),
        "chatgpt": ("openai", "gpt-4o-mini"),
        "gemini_pro": ("gemini", "gemini-1.5-pro"),
        "glm4": ("openai", None),  # not supported here; left for old path
        "glm3_turbo": ("openai", None),  # same as above
        # local models omitted in unified path on purpose
    }
    if args.provider is None and args.model in legacy_to_provider:
        args.provider, mapped_model = legacy_to_provider[args.model]
        args.model = mapped_model

    # If still no provider, pick based on model hint or fall back to OpenAI:
    if args.provider is None:
        if args.model and args.model.startswith("claude"):
            args.provider = "anthropic"
        elif args.model and args.model.startswith("gemini"):
            args.provider = "gemini"
        else:
            args.provider = "openai"

    # Create reasoning handler
    role = "reasoner" if not _is_azure() else "reasoner"  # keep name; Azure routing happens in handler
    handler = get_handler(args.provider, model=args.model, role=role)

    dataset = RareDataset(args.dataset_name, args.dataset_path, args.dataset_type)

    # Folder naming (unchanged)
    few_shot = {
        "none": "",
        "random": "_few_shot",
        "dynamic": "_dynamic_few_shot",
        "medprompt": "_medprompt",
        "auto-cot": "_auto-cot",
    }.get(args.few_shot, "")
    cot = "" if args.cot == "none" else "_cot"

    # Make result folder label human-friendly (provider@model)
    model_tag = (
        args.model
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("GEMINI_MODEL")
        or os.getenv("CLAUDE_MODEL")
        or "default"
    )
    run_tag = f"{args.provider}@{model_tag}".replace("/", "_")
    results_folder = os.path.join(
        args.results_folder, args.dataset_name, run_tag + "_" + args.task_type + few_shot + cot
    )

    run_task(
        args.task_type,
        dataset,
        handler,
        results_folder,
        args.few_shot,
        args.cot,
        args.judge_model,
        args.eval,
        rng=rng,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )

    if hasattr(handler, "total_tokens"):
        print(f"{args.provider} total tokens: {handler.total_tokens}")


if __name__ == "__main__":
    main()
