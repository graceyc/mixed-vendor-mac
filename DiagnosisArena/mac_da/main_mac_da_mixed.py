# =============================
# File: mac_da/main_mac_da_mixed.py
# =============================
#!/usr/bin/env python3
import os, json, argparse, os.path as osp
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from autogen import GroupChat, GroupChatManager, AssistantAgent, config_list_from_json
from autogen.token_count_utils import count_token

# Local imports (package-relative)
from core.data_loading import load_da_2024
from core.metrics import metrics_from_scores
from core.judge import judge_scores_o4mini

from .prompts_mac_da import (
    get_doc_system_message,
    get_supervisor_system_message,
    make_initial_open_prompt,
)
from .utils_extract import (
    extract_numbered_list,
    parse_consensus_topk,
    force_supervisor_finalize_list,
)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="MAC for DiagnosisArena (mixed-vendor: Doctor1=o4mini, Doctor2=Gemini, Doctor3=Claude; Supervisor=o4mini)."
    )
    ap.add_argument("--jsonl", required=True, help="DiagnosisArena .jsonl path (we filter year 2024).")
    ap.add_argument("--config", default="mac_da/configs/config_list.json",
                    help="Autogen config_list file path (must contain all four tags).")
    ap.add_argument("--n_round", type=int, default=13)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cache_seed", default="42",
                    help='Use "none" to disable response cache reuse; otherwise integer seed string.')
    ap.add_argument("--output_dir", default="output_mac_da")
    ap.add_argument("--o4mini_judge_model", default=None,
                    help="Azure/OpenAI deployment or name for o4-mini judge.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if judged file already exists.")
    return ap.parse_args()

# ------------------------------------------------------------
# Helpers to load per-agent configs from one config_list.json
# ------------------------------------------------------------
def cfg_by_tag(config_path: str, tag: str):
    cfg = config_list_from_json(env_or_file=config_path, filter_dict={"tags": [tag]})
    if not cfg:
        raise RuntimeError(f"No entries found in {config_path} for tag: {tag}")
    return cfg

def make_llm_cfg(cfg_list, temperature: float, cache_seed_val: Optional[int]):
    return {
        "cache_seed": cache_seed_val,
        "temperature": float(temperature),
        "config_list": cfg_list,
        "timeout": 180,
    }

# ------------------------------------------------------------
# Token count helpers (static, using token_count_utils)
# ------------------------------------------------------------
def _infer_model_name_mixed(cfg_lists: List) -> str:
    for cfg_list in cfg_lists:
        if cfg_list and isinstance(cfg_list, list):
            first = cfg_list[0]
            if isinstance(first, dict):
                m = first.get("model") or first.get("model_name")
                if isinstance(m, str) and m:
                    return m
    return "gpt-4o-mini"


def _normalize_token_model(model_name: str) -> str:
    if not model_name:
        return "gpt-4o-mini"
    lower = model_name.lower()
    if "o4-mini" in lower or "gpt-4o-mini" in lower:
        return "gpt-4o-mini"
    if "gpt-4o" in lower:
        return "gpt-4o"
    if "4.1-mini" in lower:
        return "gpt-4.1-mini"
    if "4.1" in lower:
        return "gpt-4.1"
    return "gpt-4o-mini"


def _count_case_tokens(chat_history: list, model_name: str) -> Tuple[int, int, int]:
    prompt_msgs = []
    completion_msgs = []
    for m in chat_history or []:
        role = (m.get("role") or "").lower()
        if role == "assistant":
            completion_msgs.append(m)
        else:
            prompt_msgs.append(m)
    tc_model = _normalize_token_model(model_name)
    try:
        P = count_token(prompt_msgs, model=tc_model) if prompt_msgs else 0
        C = count_token(completion_msgs, model=tc_model) if completion_msgs else 0
    except (NotImplementedError, Exception):
        P, C = 0, 0
    return P, C, P + C

# ------------------------------------------------------------
# Run one case (OPEN@5)
# ------------------------------------------------------------
def run_case_open_mixed(
    case_id: str,
    case_text: str,
    exam: str,
    tests: str,
    gold_dx: str,
    cfg_doc1, cfg_doc2, cfg_doc3, cfg_super,
    n_round: int,
    temperature: float,
    cache_seed_val: Optional[int],
    raw_dir: str,
    judged_dir: str,
    o4mini_judge_model: Optional[str] = None,
):
    rec_path = osp.join(judged_dir, f"{case_id}.json")
    if osp.exists(rec_path):
        return None  # skipped

    # Per-agent LLM configs (mixed vendors)
    doc1_llm = make_llm_cfg(cfg_doc1, temperature, cache_seed_val)  # o4mini
    doc2_llm = make_llm_cfg(cfg_doc2, temperature, cache_seed_val)  # gemini
    doc3_llm = make_llm_cfg(cfg_doc3, temperature, cache_seed_val)  # claude
    sup_llm  = make_llm_cfg(cfg_super, temperature, cache_seed_val) # o4mini

    docs = [
        AssistantAgent(
            name="Doctor1",
            llm_config=doc1_llm,
            system_message=get_doc_system_message("Doctor1"),
        ),
        AssistantAgent(
            name="Doctor2",
            llm_config=doc2_llm,
            system_message=get_doc_system_message("Doctor2"),
        ),
        AssistantAgent(
            name="Doctor3",
            llm_config=doc3_llm,
            system_message=get_doc_system_message("Doctor3"),
        ),
    ]
    supervisor = AssistantAgent(
        name="Supervisor",
        llm_config=sup_llm,
        system_message=get_supervisor_system_message(),
    )

    groupchat = GroupChat(
        agents=docs + [supervisor],
        messages=[],
        max_round=n_round,
        speaker_selection_method="round_robin",
        admin_name="Supervisor",
        select_speaker_auto_verbose=False,
        allow_repeat_speaker=True,
        send_introductions=False,
        max_retries_for_selecting_speaker=max(1, n_round // (1 + len(docs))),
    )
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=sup_llm,
        is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or ""),
    )

    initial = make_initial_open_prompt(case_text or "", exam or "", tests or "")
    output = supervisor.initiate_chat(manager, message=initial)

    # --- Token usage (static count from chat history) ---
    model_name = _infer_model_name_mixed([cfg_doc1, cfg_doc2, cfg_doc3, cfg_super])
    P, C, total_tokens = _count_case_tokens(output.chat_history, model_name)

    # Save raw conversation
    conv_path = osp.join(raw_dir, f"{case_id}_conversation.json")
    convo_jsonl = osp.join(raw_dir, f"{case_id}.jsonl")
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(output.chat_history, f, indent=2, ensure_ascii=False)
    with open(convo_jsonl, "w", encoding="utf-8") as f:
        for i, m in enumerate(output.chat_history):
            f.write(json.dumps({"idx": i, **m}, ensure_ascii=False) + "\n")

    # Per-round captures (doctor messages)
    per_round = []
    doctor_msgs = [m for m in output.chat_history if (m.get("name") or "").startswith("Doctor")]
    for ridx, msg in enumerate(doctor_msgs):
        preds = extract_numbered_list(msg.get("content", ""), k=5)
        if not preds:
            continue
        try:
            scores = judge_scores_o4mini(gold_dx, preds, o4_deployment_or_model=o4mini_judge_model)
            mets = metrics_from_scores(scores)
        except Exception:
            scores, mets = [], {"top1_acc": 0.0, "top5_acc": 0.0}
        per_round.append({
            "round": ridx,
            "doctor": msg.get("name", f"Doctor?{ridx}"),
            "predictions": preds,
            "judge_scores_0_1_2": scores,
            **mets,
        })

    # Final consensus (prefer supervisor list)
    consensus = parse_consensus_topk(output.chat_history, k=5)
    if not consensus or len(consensus) < 5:
        finalized = force_supervisor_finalize_list(supervisor, k=5)
        if finalized:
            consensus = finalized

    # Judge final
    scores = judge_scores_o4mini(gold_dx, consensus, o4_deployment_or_model=o4mini_judge_model)
    mets = metrics_from_scores(scores)

    # Save record
    rec = {
        "case_id": case_id,
        "gold": gold_dx,
        "consensus_top5": consensus,
        "judge_scores_0_1_2": scores,
        **mets,
        "per_round": per_round,
        "policy": {
            "mode": "open",
            "temperature": temperature,
            "cache_seed": cache_seed_val,
            "mixed_vendors": {
                "Doctor1": "o4mini",
                "Doctor2": "gemini-2.5-pro",
                "Doctor3": "claude-4.5-sonnet",
                "Supervisor": "o4mini",
            },
        },
        "token_usage": {
            "prompt_tokens": P,
            "completion_tokens": C,
            "total_tokens": total_tokens,
        },
    }
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    return {
        "top1_acc": float(mets.get("top1_acc", 0.0)),
        "top5_acc": float(mets.get("top5_acc", 0.0)),
        "prompt_tokens": P,
        "completion_tokens": C,
        "total_tokens": total_tokens,
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    cache_seed_val = None if str(args.cache_seed).lower() == "none" else int(args.cache_seed)

    # Load config entries ONCE (the file should include all these tags)
    cfg_doc1 = cfg_by_tag(args.config, "doctor_o4mini")
    cfg_doc2 = cfg_by_tag(args.config, "doctor_gemini")
    cfg_doc3 = cfg_by_tag(args.config, "doctor_claude")
    cfg_super = cfg_by_tag(args.config, "supervisor_o4mini")

    # Data
    recs = load_da_2024(Path(args.jsonl))

    # Output layout
    seed_suffix = f"seed{cache_seed_val}" if cache_seed_val is not None else "seedNone"
    tag_pair = "docs_o4mini+gemini+claude__sup_o4mini"
    base_dir = osp.join(
        args.output_dir,
        "MAC_DA_MIXED",
        "OPEN5",
        tag_pair,
        f"3docs_{args.n_round}r_{seed_suffix}_t{args.temperature}",
    )
    raw_dir = osp.join(base_dir, "raw"); os.makedirs(raw_dir, exist_ok=True)
    judged_dir = osp.join(base_dir, "judged"); os.makedirs(judged_dir, exist_ok=True)

    # ---- Global accumulators ----
    total_P = 0
    total_C = 0
    n_used = 0
    sum_top1 = 0.0
    sum_top5 = 0.0

    for rec in tqdm(recs, desc="DiagnosisArena-2024 open (mixed)"):
        case_id = f"DA2024-{rec.id}"
        judged_path = osp.join(judged_dir, f"{case_id}.json")
        if osp.exists(judged_path) and not args.overwrite:
            continue

        out = run_case_open_mixed(
            case_id=case_id,
            case_text=rec.case_info or "",
            exam=rec.phys_exam or "",
            tests=rec.tests or "",
            gold_dx=rec.final_dx,
            cfg_doc1=cfg_doc1, cfg_doc2=cfg_doc2, cfg_doc3=cfg_doc3, cfg_super=cfg_super,
            n_round=args.n_round,
            temperature=args.temperature,
            cache_seed_val=cache_seed_val,
            raw_dir=raw_dir,
            judged_dir=judged_dir,
            o4mini_judge_model=args.o4mini_judge_model,
        )
        if out:
            n_used += 1
            sum_top1 += out["top1_acc"]
            sum_top5 += out["top5_acc"]
            total_P += out["prompt_tokens"]
            total_C += out["completion_tokens"]
            tqdm.write(
                f"[TOKENS][OPEN][MIXED] {case_id}: "
                f"prompt={out['prompt_tokens']}, "
                f"completion={out['completion_tokens']}, "
                f"total={out['total_tokens']}"
            )

    # ---- Final summary ----
    total_tokens = total_P + total_C
    top1 = (sum_top1 / n_used) if n_used else 0.0
    top5 = (sum_top5 / n_used) if n_used else 0.0
    print("\n=== RUN SUMMARY (OPEN@5, MIXED VENDOR) ===")
    print(f"Cases evaluated: {n_used}")
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")
    print(f"Total prompt tokens (approx):     {total_P}")
    print(f"Total completion tokens (approx): {total_C}")
    print(f"Total tokens (approx):            {total_tokens}")

if __name__ == "__main__":
    main()
