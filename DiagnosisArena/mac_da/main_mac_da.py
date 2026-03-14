# =============================
# File: mac_da/main_mac_da.py
# =============================
#!/usr/bin/env python3
import os, json, argparse, os.path as osp
from pathlib import Path
from typing import List, Tuple, Optional
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
        description="MAC for DiagnosisArena (single-vendor doctors; open@5 mode)."
    )
    ap.add_argument("--jsonl", required=True, help="DiagnosisArena .jsonl path (we filter year 2024).")
    ap.add_argument("--config", default="mac_da/configs/config_list.json",
                    help="Autogen config_list file path or env var name.")
    ap.add_argument("--vendor", choices=["o4mini", "gemini", "claude"], required=True,
                    help="Single-vendor for all 3 doctors.")
    ap.add_argument("--supervisor_vendor", choices=["o4mini", "gemini", "claude"], default=None,
                    help="If set, supervisor uses this vendor; otherwise same as --vendor.")
    ap.add_argument("--num_doctors", type=int, default=3)
    ap.add_argument("--n_round", type=int, default=13)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--cache_seed", default="42",
                    help='Use "none" to disable response cache reuse; otherwise integer seed string.')
    ap.add_argument("--output_dir", default="output_mac_da")
    ap.add_argument("--o4mini_judge_model", default=None,
                    help="Azure/OpenAI deployment or name for o4-mini judge.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if judged file already exists.")
    return ap.parse_args()

# ------------------------------------------------------------
# Token count helpers (static, using token_count_utils)
# ------------------------------------------------------------
def _infer_model_name(cfg_doctors, cfg_super) -> str:
    for cfg_list in (cfg_super, cfg_doctors):
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
# Tag selection
# ------------------------------------------------------------
def vendor_tags(vendor: str) -> Tuple[str, list]:
    if vendor == "o4mini":
        return "doctor_o4mini", ["supervisor_o4mini", "x_o4mini", "doctor_o4mini"]
    if vendor == "gemini":
        return "doctor_gemini", ["supervisor_gemini", "doctor_gemini"]
    if vendor == "claude":
        return "doctor_claude", ["supervisor_claude", "doctor_claude"]
    raise ValueError(f"Unknown vendor: {vendor}")


def pick_supervisor_cfg(config_path: str, sup_candidates: list):
    for tag in sup_candidates:
        cfg = config_list_from_json(env_or_file=config_path, filter_dict={"tags": [tag]})
        if cfg:
            return cfg, tag
    return [], None

# ------------------------------------------------------------
# Run one case (OPEN@5)
# ------------------------------------------------------------
def run_case_open(
    case_id: str,
    case_text: str,
    exam: str,
    tests: str,
    gold_dx: str,
    cfg_doctors,
    cfg_super,
    num_doctors: int,
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

    doctor_llm_config = {
        "cache_seed": cache_seed_val,
        "temperature": float(temperature),
        "config_list": cfg_doctors,
        "timeout": 180,
    }
    supervisor_llm_config = {
        "cache_seed": cache_seed_val,
        "temperature": float(temperature),
        "config_list": cfg_super,
        "timeout": 180,
    }

    docs = [
        AssistantAgent(
            name=f"Doctor{i+1}",
            llm_config=doctor_llm_config,
            system_message=get_doc_system_message(f"Doctor{i+1}"),
        )
        for i in range(num_doctors)
    ]
    supervisor = AssistantAgent(
        name="Supervisor",
        llm_config=supervisor_llm_config,
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
        max_retries_for_selecting_speaker=max(1, n_round // (1 + num_doctors)),
    )
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=supervisor_llm_config,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    )

    # Initial prompt
    initial = make_initial_open_prompt(case_text or "", exam or "", tests or "")
    output = supervisor.initiate_chat(manager, message=initial)

    # --- Token usage (static count from chat history) ---
    model_name = _infer_model_name(cfg_doctors, cfg_super)
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

    # Judge final consensus
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

    # Doctor tag + supervisor candidates
    doc_tag, _ = vendor_tags(args.vendor)
    sup_vendor = args.supervisor_vendor or args.vendor
    _, sup_candidates_from_vendor = vendor_tags(sup_vendor)

    # Autogen config lists
    cfg_doctors = config_list_from_json(env_or_file=args.config, filter_dict={"tags": [doc_tag]})
    if not cfg_doctors:
        raise RuntimeError(f"No entries found in {args.config} for doctor tag: {doc_tag}")

    cfg_super, chosen_sup_tag = pick_supervisor_cfg(args.config, sup_candidates_from_vendor)
    if not cfg_super:
        cfg_super = config_list_from_json(env_or_file=args.config, filter_dict={"tags": [doc_tag]})
        chosen_sup_tag = doc_tag if cfg_super else None
    if not cfg_super:
        raise RuntimeError(f"No supervisor entries found for any of tags {sup_candidates_from_vendor + [doc_tag]}")

    # Data
    recs = load_da_2024(Path(args.jsonl))

    # Output layout
    seed_suffix = f"seed{cache_seed_val}" if cache_seed_val is not None else "seedNone"
    tag_pair = f"docs_{doc_tag}__sup_{chosen_sup_tag}"
    base_dir = osp.join(
        args.output_dir,
        "MAC_DA",
        "OPEN5",
        tag_pair,
        f"{args.num_doctors}docs_{args.n_round}r_{seed_suffix}_t{args.temperature}",
    )
    raw_dir = osp.join(base_dir, "raw"); os.makedirs(raw_dir, exist_ok=True)
    judged_dir = osp.join(base_dir, "judged"); os.makedirs(judged_dir, exist_ok=True)

    # ---- Global accumulators ----
    total_P = 0
    total_C = 0
    n_used = 0
    sum_top1 = 0.0
    sum_top5 = 0.0

    for rec in tqdm(recs, desc="DiagnosisArena-2024 open"):
        case_id = f"DA2024-{rec.id}"
        judged_path = osp.join(judged_dir, f"{case_id}.json")
        if osp.exists(judged_path) and not args.overwrite:
            continue

        out = run_case_open(
            case_id=case_id,
            case_text=rec.case_info or "",
            exam=rec.phys_exam or "",
            tests=rec.tests or "",
            gold_dx=rec.final_dx,
            cfg_doctors=cfg_doctors,
            cfg_super=cfg_super,
            num_doctors=args.num_doctors,
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
                f"[TOKENS][OPEN] {case_id}: "
                f"prompt={out['prompt_tokens']}, "
                f"completion={out['completion_tokens']}, "
                f"total={out['total_tokens']}"
            )

    # ---- Final summary ----
    total_tokens = total_P + total_C
    top1 = (sum_top1 / n_used) if n_used else 0.0
    top5 = (sum_top5 / n_used) if n_used else 0.0
    print("\n=== RUN SUMMARY (OPEN@5) ===")
    print(f"Cases evaluated: {n_used}")
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")
    print(f"Total prompt tokens (approx):     {total_P}")
    print(f"Total completion tokens (approx): {total_C}")
    print(f"Total tokens (approx):            {total_tokens}")

if __name__ == "__main__":
    main()
