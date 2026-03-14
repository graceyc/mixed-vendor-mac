#!/usr/bin/env python3
# mac_runner/main_mac.py
import os, re, json, argparse, os.path as osp
from tqdm import tqdm
from autogen import GroupChat, GroupChatManager, AssistantAgent, config_list_from_json
from utils.mydataset import RareDataset
from mac_runner.prompts_mac_rare import (
    get_initial_message, get_doc_system_message, get_supervisor_system_message,
)
from mac_runner.mac_eval_adapter import judge_with_deeprare


# ============================================================
# CLI & Config
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run MAC pipeline with per-round LLM judging (doctors vs supervisor tags)."
    )
    ap.add_argument(
        "--config", type=str, default="mac_runner/configs/config_list.json"
    )
    ap.add_argument(
        "--doctor_tag", type=str, default="doctor_claude",
        help="Tag in config_list.json used for all Doctors (e.g., doctor_claude, doctor_gpt4o)."
    )
    ap.add_argument(
        "--supervisor_tag", type=str, default="x_gpt4o",
        help="Tag in config_list.json used for the Supervisor (e.g., x_gpt4o, supervisor_claude)."
    )
    ap.add_argument(
        "--dataset_name", type=str, default="HMS",
        choices=["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"]
    )
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--num_doctors", type=int, default=3)
    ap.add_argument("--n_round", type=int, default=13)
    ap.add_argument(
        "--judge_label", type=str, default="gpt4", choices=["gpt4", "chatgpt"],
        help="DeepRare judge label; OpenAI/Azure mapping handled in your api wrapper."
    )
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--cache_seed", type=str, default="none",
        help='Use "none" to disable response cache reuse; otherwise set an integer.'
    )
    # --- new: sharding for parallel runs ---
    ap.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of parallel shards (processes)."
    )
    ap.add_argument(
        "--shard_id", type=int, default=0,
        help="This process's shard id in [0, num_shards-1]."
    )
    return ap.parse_args()


# ============================================================
# Text Parsing Helpers
# ============================================================

def _extract_top10_from_text(text: str):
    """
    Extract up to 10 diagnosis names from a doctor's/supervisor's message.

    Handles two common formats:
      (A) An explicit 'Top-10' heading followed by a clean 1..10 list.
      (B) Any contiguous numbered list where each item may include reasoning (' - ' or ' : ').

    Returns a de-duplicated list (max 10).
    """
    import re

    def is_numbered_line(s: str) -> bool:
        return re.match(r"^\s*(?:10|[1-9])[\).\s-]+\s*\S", s) is not None

    def split_reasoning(name: str) -> str:
        return re.split(r"\s[-–—:]\s", name, maxsplit=1)[0].strip()

    def clean_item(s: str) -> str:
        s = re.sub(r"\*\*|\*|_", "", s)
        s = re.sub(r"\(.*?\)", "", s)
        return s.strip()

    def dedup_cap10(items):
        seen, out = set(), []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
                if len(out) == 10:
                    break
        return out

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]

    top10_pat = re.compile(
        r"^\s*(?:final|current)?\s*top\s*[-\s]*?10(?:\s*list)?\s*:?\s*$",
        flags=re.IGNORECASE
    )

    blocks_after_heading = []
    for i, ln in enumerate(lines):
        if top10_pat.match(ln):
            block = []
            for j in range(i + 1, len(lines)):
                if is_numbered_line(lines[j]):
                    block.append(lines[j])
                elif block:
                    break
            if len(block) >= 2:
                blocks_after_heading.append(block)

    if blocks_after_heading:
        best = blocks_after_heading[-1]
    else:
        blocks, cur = [], []
        for ln in lines:
            if is_numbered_line(ln):
                cur.append(ln)
            else:
                if len(cur) >= 2:
                    blocks.append(cur)
                cur = []
        if len(cur) >= 2:
            blocks.append(cur)

        if not blocks:
            return []

        def block_score(block, idx):
            n = len(block)
            ends_at_10 = 1 if re.match(r"^\s*10[\).\s-]+\s", block[-1]) else 0
            return (n, ends_at_10, idx)

        best = sorted(
            [(b, i) for i, b in enumerate(blocks)],
            key=lambda bi: block_score(bi[0], bi[1]),
            reverse=True
        )[0][0]

    extracted = []
    for ln in best:
        m = re.match(r"^\s*(?:10|[1-9])[\).\s-]+\s*(.+)$", ln)
        if not m:
            continue
        item = split_reasoning(clean_item(m.group(1)))
        low = item.lower()
        if any(tag in low for tag in ["next steps", "depriorit", "excluded", "refinement"]):
            continue
        extracted.append(item)

    return dedup_cap10(extracted)


def _parse_consensus_from_conversation(chat_history):
    """
    Return the most recent message that contains a valid 1..10 list,
    preferring Supervisor messages but falling back to any message.
    """
    sup_msgs = [m for m in chat_history if (m.get("name") or "").lower().startswith("supervisor")]
    for m in reversed(sup_msgs):
        text = (m.get("content") or "")
        lst = _extract_top10_from_text(text)
        if len(lst) >= 5:
            return lst[:10]

    for m in reversed(chat_history):
        text = (m.get("content") or "")
        lst = _extract_top10_from_text(text)
        if len(lst) >= 5:
            return lst[:10]

    return []


def _finalize_now(supervisor):
    """Force the Supervisor to finalize 10 diagnoses."""
    prompt = (
        "Finalize now. Reply ONLY with a numbered list of exactly 10 diagnoses, "
        "one per line, formatted as '1. ...' through '10. ...'. Then append TERMINATE."
    )
    try:
        reply = supervisor.generate_reply([{"role": "user", "content": prompt}])
        text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
        consensus = _extract_top10_from_text(text)
        return consensus if consensus else None
    except Exception:
        return None


# ============================================================
# Main Pipeline
# ============================================================

def main():
    args = parse_args()

    # Basic sanity for sharding
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard_id must be in [0, num_shards-1], got {args.shard_id} with num_shards={args.num_shards}")

    # Separate Autogen config lists for Doctors vs Supervisor
    cfg_doctors = config_list_from_json(
        env_or_file=args.config, filter_dict={"tags": [args.doctor_tag]}
    )
    cfg_super   = config_list_from_json(
        env_or_file=args.config, filter_dict={"tags": [args.supervisor_tag]}
    )

    if not cfg_doctors:
        raise RuntimeError(f"No entries found in {args.config} for doctor_tag={args.doctor_tag}")
    if not cfg_super:
        raise RuntimeError(f"No entries found in {args.config} for supervisor_tag={args.supervisor_tag}")

    cache_seed_val = None if str(args.cache_seed).lower() == "none" else int(args.cache_seed)

    doctor_llm_config = {
        "cache_seed": cache_seed_val,
        "temperature": float(args.temperature),
        "config_list": cfg_doctors,
        "timeout": 180,
    }
    supervisor_llm_config = {
        "cache_seed": cache_seed_val,
        "temperature": float(args.temperature),
        "config_list": cfg_super,
        "timeout": 180,
    }

    # Dataset
    ds = RareDataset(dataset_name=args.dataset_name, dataset_path=None, dataset_type="PHENOTYPE")
    N = len(ds.patient)

    # Output dirs
    seed_suffix = f"seed{cache_seed_val}" if cache_seed_val is not None else "seedNone"
    tag_pair = f"docs_{args.doctor_tag}__sup_{args.supervisor_tag}"
    base_dir = osp.join(
        args.output_dir, "MAC", args.dataset_name, tag_pair,
        f"{args.num_doctors}docs_{args.n_round}r_{seed_suffix}_t{args.temperature}"
    )
    raw_dir = osp.join(base_dir, "raw")
    judged_dir = osp.join(base_dir, "judged")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(judged_dir, exist_ok=True)

    print(
        f"[INFO] Dataset={args.dataset_name}, N={N}, "
        f"num_shards={args.num_shards}, shard_id={args.shard_id}"
    )

    # Process each case (this process only handles its shard)
    for idx in tqdm(range(N), desc=f"Processing {args.dataset_name}"):
        # Sharding: only handle indices assigned to this shard
        if idx % args.num_shards != args.shard_id:
            continue

        patient_info_str, golden_names_str = ds.patient[idx]
        case_id = f"{args.dataset_name}-{idx}"
        conv_path = osp.join(raw_dir, f"{case_id}_conversation.json")
        convo_jsonl_path = osp.join(raw_dir, f"{case_id}.jsonl")
        rec_path = osp.join(judged_dir, f"{case_id}.json")

        # Skip if already judged
        if osp.exists(rec_path):
            continue

        # 1) Resume judged conversation
        if osp.exists(conv_path) and not osp.exists(rec_path):
            with open(conv_path, "r") as f:
                chat_history = json.load(f)
            consensus = _parse_consensus_from_conversation(chat_history)
            if not consensus or len(consensus) < 10:
                consensus = _finalize_now(
                    AssistantAgent(
                        name="Supervisor",
                        llm_config=supervisor_llm_config,
                        system_message=get_supervisor_system_message()
                    )
                ) or []
            pred_str, rank = judge_with_deeprare(
                consensus, golden_names_str, judge_label=args.judge_label
            )
            rec = {
                "dataset": args.dataset_name,
                "case_id": case_id,
                "phenotypes_names": patient_info_str,
                "gold_names": golden_names_str,
                "supervisor_consensus_top10": consensus,
                "supervisor_consensus_top10_numbered": pred_str,
                "judge_rank_at_10": rank,
                "cache_resume": True,
            }
            with open(rec_path, "w") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            continue

        # 2) Run new MAC conversation
        docs = [
            AssistantAgent(
                name=f"Doctor{i+1}",
                llm_config=doctor_llm_config,
                system_message=get_doc_system_message(f"Doctor{i+1}")
            )
            for i in range(args.num_doctors)
        ]
        supervisor = AssistantAgent(
            name="Supervisor",
            llm_config=supervisor_llm_config,
            system_message=get_supervisor_system_message(),
        )

        groupchat = GroupChat(
            agents=docs + [supervisor],
            messages=[],
            max_round=args.n_round,
            speaker_selection_method="round_robin",
            admin_name="Supervisor",
            select_speaker_auto_verbose=False,
            allow_repeat_speaker=True,
            send_introductions=False,
            max_retries_for_selecting_speaker=max(1, args.n_round // (1 + args.num_doctors)),
        )
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=supervisor_llm_config,  # manager uses Supervisor backend
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        )

        # Run the multi-agent chat
        initial = get_initial_message(patient_info_str)
        output = supervisor.initiate_chat(manager, message=initial)

        # Save raw chat
        with open(conv_path, "w") as f:
            json.dump(output.chat_history, f, indent=2, ensure_ascii=False)
        with open(convo_jsonl_path, "w") as f:
            for i, m in enumerate(output.chat_history):
                f.write(json.dumps({"idx": i, **m}, ensure_ascii=False) + "\n")

        # 3) Per-round LLM judging (doctors' messages)
        per_round_metrics = []
        doctor_msgs = [m for m in output.chat_history if m.get("name", "").startswith("Doctor")]
        for ridx, msg in enumerate(doctor_msgs):
            preds = _extract_top10_from_text(msg.get("content", ""))
            if not preds:
                continue
            preds_formatted = "\n".join([f"{i+1}. {p}" for i, p in enumerate(preds)])

            try:
                pred_str, rank = judge_with_deeprare(
                    preds_formatted, golden_names_str, judge_label=args.judge_label
                )
                recalls = {
                    f"r@{k}": 1.0 if isinstance(rank, (int, float)) and rank is not None and rank <= k else 0.0
                    for k in [1, 3, 5, 10]
                }
            except Exception as e:
                print(f"[WARN] Per-round judging failed ({msg.get('name','Doctor?')} round {ridx}): {e}")
                rank = None
                pred_str = preds_formatted
                recalls = {f"r@{k}": 0.0 for k in [1, 3, 5, 10]}

            per_round_metrics.append({
                "round": ridx,
                "doctor": msg.get("name", f"Doctor?{ridx}"),
                "predictions": preds,
                "judge_rank_at_10": rank,
                **recalls,
            })

        # 4) Final supervisor judging
        consensus = _parse_consensus_from_conversation(output.chat_history)
        if not consensus or len(consensus) < 10:
            consensus = _finalize_now(supervisor) or []
        pred_str, rank = judge_with_deeprare(
            consensus, golden_names_str, judge_label=args.judge_label
        )

        # 5) Save final record
        rec = {
            "dataset": args.dataset_name,
            "case_id": case_id,
            "phenotypes_names": patient_info_str,
            "gold_names": golden_names_str,
            "supervisor_consensus_top10": consensus,
            "supervisor_consensus_top10_numbered": pred_str,
            "judge_rank_at_10": rank,
            "per_round_metrics": per_round_metrics,
            "cache_resume": False,
            "policy": {
                "speaker_selection": "round_robin",
                "allow_repeat_speaker": True,
                "temperature": args.temperature,
                "cache_seed": cache_seed_val,
                "doctor_tag": args.doctor_tag,
                "supervisor_tag": args.supervisor_tag,
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
            },
        }
        with open(rec_path, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
