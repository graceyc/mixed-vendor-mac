#!/usr/bin/env python3
# mac_mixed/main_mixed.py
import os, sys, json, argparse, os.path as osp
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

# ---- make imports work no matter where you run from (repo root = parent of this file's dir)
THIS_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.abspath(osp.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from autogen import GroupChat, GroupChatManager, AssistantAgent, config_list_from_json
from utils.mydataset import RareDataset
from mac_runner.prompts_mac_rare import (
    get_initial_message, get_doc_system_message, get_supervisor_system_message,
)
from mac_runner.mac_eval_adapter import judge_with_deeprare

# reuse the exact extractor and helpers from main_mac.py
from mac_runner.main_mac import (
    _extract_top10_from_text as extract_top10,
    _parse_consensus_from_conversation,
)

# vendor callers
from mac_mixed.vendor_clients.openai_azure import call_openai_azure
from mac_mixed.vendor_clients.gemini import call_gemini
from mac_mixed.vendor_clients.claude import call_claude


# ============================================================
# CLI & Config (identical shape to main_mac.py)
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Run MAC pipeline with per-round LLM judging.")
    ap.add_argument("--config", type=str, default="mac_runner/configs/config_list.json")
    ap.add_argument(
        "--model_name",
        type=str,
        default="x_gpt4o",
        choices=["x_gpt4o", "x_gpt4_turbo", "x_gpt35_turbo"],
    )
    ap.add_argument(
        "--dataset_name",
        type=str,
        default="HMS",
        choices=["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"],
    )
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--num_doctors", type=int, default=3)
    ap.add_argument("--n_round", type=int, default=13)
    ap.add_argument(
        "--judge_label",
        type=str,
        default="gpt4",
        choices=["gpt4", "chatgpt"],
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--cache_seed",
        type=str,
        default="none",
        help='Use "none" to disable response cache reuse; otherwise set an integer.',
    )

    # ---- sharding args ----
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards to split the dataset into.")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Which shard index this process should handle (0-based).")

    return ap.parse_args()


# ============================================================
# Mixed-vendor AssistantAgent subclasses
# ============================================================
class MixedVendorAgent(AssistantAgent):
    vendor: str = "azure"  # 'azure' | 'gemini' | 'claude'

    def __init__(self, *args, temperature: float = 1.0, model_id: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = float(temperature)
        self.model_id = model_id  # per-agent override

    def _history_to_strings(self, chat_history):
        parts = []
        for m in chat_history or []:
            name = m.get("name") or (m.get("role") or "user").title()
            content = m.get("content", "")
            parts.append(f"[{name}]\n{content}")
        return parts

    def generate_reply(self, messages=None, sender=None, **kwargs):
        # Autogen passes this turn's history as `messages`; fallback to sender's view
        if messages is None:
            messages = self.chat_messages.get(sender, []) if sender is not None else []

        system_message = self.system_message or ""
        history_texts = self._history_to_strings(messages)

        if self.vendor == "gemini":
            text = call_gemini(
                system_message,
                history_texts,
                model=self.model_id or os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro"),
                temperature=self.temperature,
            )
        elif self.vendor == "claude":
            text = call_claude(
                system_message,
                history_texts,
                model=self.model_id or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
                temperature=self.temperature,
            )
        else:  # azure
            text = call_openai_azure(
                system_message,
                history_texts,
                deployment=self.model_id or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-1120"),
                temperature=self.temperature,
            )
        return {"role": "assistant", "name": self.name, "content": text}


class AzureDoctor(MixedVendorAgent):
    vendor = "azure"


class GeminiDoctor(MixedVendorAgent):
    vendor = "gemini"


class ClaudeDoctor(MixedVendorAgent):
    vendor = "claude"


class AzureSupervisor(MixedVendorAgent):
    vendor = "azure"


class ClaudeSupervisor(MixedVendorAgent):
    vendor = "claude"


class GeminiSupervisor(MixedVendorAgent):
    vendor = "gemini"


# ============================================================
# Force finalization via Supervisor (like main_mac)
# ============================================================
def _finalize_now(supervisor: MixedVendorAgent):
    prompt = (
        "Finalize now. Reply ONLY with a numbered list of exactly 10 diagnoses, "
        "one per line, formatted as '1. ...' through '10. ...'. Then append TERMINATE."
    )
    try:
        reply = supervisor.generate_reply([{"role": "user", "content": prompt}])
        text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
        consensus = extract_top10(text)
        return consensus if consensus else None
    except Exception:
        return None


# ============================================================
# Main (mirrors main_mac.py)
# ============================================================
def main():
    load_dotenv(find_dotenv(), override=False)
    args = parse_args()

    # Print actual backends/models ONCE at startup (from env — mirrors what we pass below)
    plan = []
    if args.num_doctors >= 1:
        plan.append(
            (
                "Doctor1",
                "Azure OpenAI",
                os.getenv("AZURE_OPENAI_DOCTOR_DEPLOYMENT", "o4-mini-0416"),
            )
        )
    if args.num_doctors >= 2:
        plan.append(
            (
                "Doctor2",
                "Google Gemini",
                os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro"),
            )
        )
    if args.num_doctors >= 3:
        plan.append(
            (
                "Doctor3",
                "Anthropic Claude",
                os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
            )
        )
    sup_model = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
    plan.append(("Supervisor", "Google Gemini", sup_model))

    print("\n" + "-" * 100)
    print("Agent backends/models")
    print("-" * 100)
    for name, vendor, model in plan:
        print(f"{name:<10} → {vendor} · {model}")
    print("-" * 100 + "\n")

    # --- Model config for GroupChatManager (same shape as main_mac.py)
    cfg_list = config_list_from_json(env_or_file=args.config, filter_dict={"tags": [args.model_name]})
    cache_seed_val = None if str(args.cache_seed).lower() == "none" else int(args.cache_seed)
    model_config = {
        "cache_seed": cache_seed_val,
        "temperature": float(args.temperature),
        "config_list": cfg_list,
        "timeout": 180,
    }

    # --- Dataset ---
    ds = RareDataset(dataset_name=args.dataset_name, dataset_path=None, dataset_type="PHENOTYPE")
    N = len(ds.patient)

    # ---- sharding logic ----
    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id={shard_id} must be in [0, {num_shards - 1}]")

    indices = [i for i in range(N) if i % num_shards == shard_id]
    print(
        f"[INFO] Dataset={args.dataset_name}, N={N}, "
        f"num_shards={num_shards}, shard_id={shard_id}, local_N={len(indices)}"
    )

    # --- Output structure (identical pattern)
    seed_suffix = f"seed{cache_seed_val}" if cache_seed_val is not None else "seedNone"
    base_dir = osp.join(
        args.output_dir,
        "MAC",
        args.dataset_name,
        args.model_name,
        f"{args.num_doctors}docs_{args.n_round}r_{seed_suffix}_t{args.temperature}",
    )
    raw_dir = osp.join(base_dir, "raw")
    judged_dir = osp.join(base_dir, "judged")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(judged_dir, exist_ok=True)

    # ========================================================
    # Process Each Case (only this shard's indices)
    # ========================================================
    for idx in tqdm(indices, desc=f"Processing {args.dataset_name}"):
        patient_info_str, golden_names_str = ds.patient[idx]
        case_id = f"{args.dataset_name}-{idx}"
        conv_path = osp.join(raw_dir, f"{case_id}_conversation.json")
        convo_jsonl_path = osp.join(raw_dir, f"{case_id}.jsonl")
        rec_path = osp.join(judged_dir, f"{case_id}.json")

        if osp.exists(rec_path):
            continue

        # 1) Resume judged conversation
        if osp.exists(conv_path) and not osp.exists(rec_path):
            with open(conv_path, "r") as f:
                chat_history = json.load(f)
            consensus = _parse_consensus_from_conversation(chat_history)
            if not consensus or len(consensus) < 10:
                supervisor_tmp = GeminiSupervisor(
                    name="Supervisor",
                    llm_config=model_config,
                    system_message=get_supervisor_system_message(),
                    temperature=args.temperature,
                    model_id=os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro"),
                )
                consensus = _finalize_now(supervisor_tmp) or []
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

        # 2) New MAC conversation
        docs = [
            AzureDoctor(
                name="Doctor1",
                llm_config=model_config,
                system_message=get_doc_system_message("Doctor1"),
                temperature=args.temperature,
                model_id=os.getenv("AZURE_OPENAI_DOCTOR_DEPLOYMENT", "o4-mini-0416"),
            ),
            GeminiDoctor(
                name="Doctor2",
                llm_config=model_config,
                system_message=get_doc_system_message("Doctor2"),
                temperature=args.temperature,
                model_id=os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro"),
            ),
            ClaudeDoctor(
                name="Doctor3",
                llm_config=model_config,
                system_message=get_doc_system_message("Doctor3"),
                temperature=args.temperature,
                model_id=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
            ),
        ][: args.num_doctors]

        supervisor = GeminiSupervisor(
            name="Supervisor",
            llm_config=model_config,
            system_message=get_supervisor_system_message(),
            temperature=args.temperature,
            model_id=os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro"),
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
            llm_config=model_config,
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or ""),
        )

        # Run
        initial = get_initial_message(patient_info_str)
        output = supervisor.initiate_chat(manager, message=initial)

        # Save raw chat
        with open(conv_path, "w") as f:
            json.dump(output.chat_history, f, indent=2, ensure_ascii=False)
        with open(convo_jsonl_path, "w") as f:
            for i, m in enumerate(output.chat_history):
                f.write(json.dumps({"idx": i, **m}, ensure_ascii=False) + "\n")

        # 3) Per-round judging (doctors)
        per_round_metrics = []
        doctor_msgs = [m for m in output.chat_history if m.get("name", "").startswith("Doctor")]
        for ridx, msg in enumerate(doctor_msgs):
            preds = extract_top10(msg.get("content", ""))
            if not preds:
                continue
            preds_formatted = "\n".join([f"{i+1}. {p}" for i, p in enumerate(preds)])

            try:
                pred_str, rank = judge_with_deeprare(
                    preds_formatted, golden_names_str, judge_label=args.judge_label
                )
                recalls = {
                    f"r@{k}": 1.0 if isinstance(rank, (int, float)) and rank <= k else 0.0
                    for k in [1, 3, 5, 10]
                }
            except Exception as e:
                print(f"[WARN] Per-round judging failed ({msg['name']} round {ridx}): {e}")
                rank, pred_str = None, preds_formatted
                recalls = {f"r@{k}": 0.0 for k in [1, 3, 5, 10]}

            per_round_metrics.append(
                {
                    "round": ridx,
                    "doctor": msg["name"],
                    "predictions": preds,
                    "judge_rank_at_10": rank,
                    **recalls,
                }
            )

        # 4) Final supervisor judging
        consensus = _parse_consensus_from_conversation(output.chat_history)
        if not consensus or len(consensus) < 10:
            # use the same supervisor model for forced finalization
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
            },
        }
        with open(rec_path, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
