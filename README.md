# Mixed-Vendor Multi-Agent Consultation for Clinical Diagnosis

Code for the paper: **"Do Mixed-Vendor Multi-Agent LLM Systems Improve Clinical Diagnosis?"** ([arXiv:2603.04421](https://arxiv.org/abs/2603.04421))

This repository implements and evaluates three diagnostic configurations:
- **Single LLM**: A single model generating differential diagnoses.
- **Single-Vendor MAC**: Multiple agents from the same model family collaborating via Multi-Agent Consultation.
- **Mixed-Vendor MAC**: Agents from different vendors (OpenAI o4-mini, Google Gemini-2.5-Pro, Anthropic Claude-4.5-Sonnet) collaborating together.

Experiments are run on two benchmarks:
- **[RareBench](https://huggingface.co/datasets/chenxz/RareBench)** — Rare disease diagnosis
- **[DiagnosisArena](https://arxiv.org/abs/2505.14107)** — General clinical diagnosis (2024 cases, open-ended Top-5 task)

## Repository Structure

```
MixedVendorMAC/
├── RareBench/                          # RareBench dataset experiments
│   ├── main.py                         # Single-LLM runner
│   ├── prompt.py                       # Prompt generation
│   ├── llm_utils/
│   │   ├── unified.py                  # Unified LLM handler (OpenAI/Anthropic/Gemini)
│   │   └── api.py                      # Legacy OpenAI handler (used for judging)
│   ├── utils/
│   │   ├── mydataset.py                # RareBench dataset loader
│   │   └── evaluation.py               # LLM judge function
│   ├── mac_runner/                     # Single-vendor MAC
│   │   ├── main_mac.py                 # MAC runner (AutoGen-based)
│   │   ├── prompts_mac_rare.py         # Doctor/supervisor system prompts
│   │   ├── mac_eval_adapter.py         # Judge wrapper for MAC
│   │   ├── embed_eval_mac.py           # Embedding evaluation (BioLORD) for MAC results
│   │   └── configs/config_list.json    # AutoGen agent config (add your API keys)
│   ├── mac_mixed/                      # Mixed-vendor MAC
│   │   ├── main_mixed.py              # Mixed-vendor MAC runner
│   │   ├── vendor_agents.py           # Per-vendor agent wrappers
│   │   └── vendor_clients/            # Vendor-specific API clients
│   │       ├── openai_azure.py
│   │       ├── gemini.py
│   │       └── claude.py
│   ├── summarize_single_runner.py      # Aggregate single-LLM results (R@1/3/5/10)
│   ├── embed_eval_single.py            # Embedding evaluation (BioLORD) for single-LLM results
│   └── mapping/                        # Required data files
│       ├── phenotype_mapping.json
│       ├── disease_mapping.json
│       ├── ic_dict.json
│       └── num.json
│
└── DiagnosisArena/                     # DiagnosisArena dataset experiments
    ├── run_single.py                   # Single-LLM runner (Top-5)
    ├── requirements.txt                # Python dependencies
    ├── core/
    │   ├── data_loading.py             # Load DiagnosisArena JSONL
    │   ├── prompts.py                  # Top-5 prompt template
    │   ├── llm_handlers.py             # Unified LLM handler
    │   ├── judge.py                    # o4-mini LLM judge (0/1/2 scoring)
    │   └── metrics.py                  # Parse predictions, compute accuracy
    └── mac_da/                         # MAC framework
        ├── main_mac_da.py              # Single-vendor MAC runner
        ├── main_mac_da_mixed.py        # Mixed-vendor MAC runner
        ├── prompts_mac_da.py           # Doctor/supervisor prompts
        ├── utils_extract.py            # Extract consensus from chat
        └── configs/config_list.json    # AutoGen agent config (add your API keys)
```

## Setup

### 1. Install dependencies

```bash
# Core dependencies (both benchmarks)
pip install openai anthropic google-generativeai pandas tqdm python-dotenv

# For MAC (multi-agent) runs
pip install pyautogen

# For embedding-based evaluation (RareBench only)
pip install sentence-transformers datasets
```

### 2. Configure API keys

**Option A: Environment variables** (recommended)
```bash
# OpenAI / Azure OpenAI
export OPENAI_API_KEY="your-key"
# Or for Azure:
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google Gemini
export GEMINI_API_KEY="your-key"
```

**Option B: Config files** (for MAC runs)

Edit the `config_list.json` files in `RareBench/mac_runner/configs/` and `DiagnosisArena/mac_da/configs/` with your API keys.

## Usage

### DiagnosisArena

**Single LLM (Top-5)**
```bash
cd DiagnosisArena

# Run with OpenAI
python run_single.py --jsonl data.jsonl --provider openai --model gpt-4o-mini --outdir runs/o4mini

# Run with Anthropic
python run_single.py --jsonl data.jsonl --provider anthropic --model claude-4.5-sonnet --outdir runs/claude

# Run with Gemini
python run_single.py --jsonl data.jsonl --provider gemini --model models/gemini-2.5-pro --outdir runs/gemini
```

**Single-Vendor MAC**
```bash
python -m mac_da.main_mac_da --jsonl data.jsonl --vendor o4mini
python -m mac_da.main_mac_da --jsonl data.jsonl --vendor gemini
python -m mac_da.main_mac_da --jsonl data.jsonl --vendor claude
```

**Mixed-Vendor MAC**
```bash
python -m mac_da.main_mac_da_mixed --jsonl data.jsonl
```

### RareBench

**Single LLM**
```bash
cd RareBench

python main.py --provider openai --model gpt-4o-mini --dataset_name HMS --eval
python main.py --provider anthropic --model claude-4.5-sonnet --dataset_name HMS --eval
python main.py --provider gemini --model models/gemini-2.5-pro --dataset_name HMS --eval
```

**Single-Vendor MAC**
```bash
python -m mac_runner.main_mac --doctor_tag doctor_o4mini --supervisor_tag x_o4mini --dataset_name HMS
```

**Mixed-Vendor MAC**
```bash
python -m mac_mixed.main_mixed --dataset_name HMS
```

**Aggregate single-LLM results**
```bash
python summarize_single_runner.py --root single_results --datasets HMS MME LIRICAL --csv_out results.csv
```

**Embedding-based evaluation (BioLORD)**
```bash
# Single-LLM results
python embed_eval_single.py \
  --judged_dir single_results/HMS/openai@gpt-4o-mini_diagnosis \
  --disease_mapping mapping/disease_mapping.json

# MAC results (single-vendor or mixed-vendor)
python -m mac_runner.embed_eval_mac \
  --judged_dir output/MAC/HMS/docs_doctor_o4mini__sup_x_o4mini/3docs_13r_seedNone_t0.2/judged \
  --disease_mapping mapping/disease_mapping.json
```

## Evaluation Methods

The two benchmarks use different evaluation approaches.

### DiagnosisArena

Evaluation is performed **during the run** (in `run_single.py` and the MAC scripts). Each case goes through:

1. **Prediction**: The LLM (or MAC agents) generates 5 differential diagnoses.
2. **Judging**: Each of the 5 predictions is sent to an o4-mini LLM judge (`core/judge.py`), which scores it against the gold reference diagnosis:
   - `2` = exact match
   - `1` = broad category containing the reference (e.g., "cardiomyopathy" when gold is "dilated cardiomyopathy")
   - `0` = incorrect
3. **Accuracy**: Only score `2` (exact match) counts as correct (`core/metrics.py`):
   - **Top-1 accuracy**: 1 if the first prediction scored 2, else 0
   - **Top-5 accuracy**: 1 if any of the 5 predictions scored 2, else 0

Results are saved per-case as JSON files.

### RareBench

Two evaluation methods are available:

**1. LLM Judge** (built into the pipeline)

The LLM generates 10 differential diagnoses per case. An o4-mini judge determines which prediction (rank 1-10) matches the gold diagnosis, or returns "No" if none match. Recall is computed as R@1, R@3, R@5, and R@10.

- For **single-LLM** runs: pass `--eval` to `main.py` to judge inline, then use `summarize_single_runner.py` to aggregate.
- For **MAC** runs: judging runs automatically during the pipeline (both per-round doctor predictions and the final supervisor consensus are judged).

**2. Embedding-based (BioLORD)** (post-hoc)

Uses the [BioLORD-2023](https://huggingface.co/FremyCompany/BioLORD-2023) biomedical embedding model to re-evaluate results by computing cosine similarity between predicted diagnoses and the full disease code index. The model is automatically downloaded on first use.

- For **single-LLM** results: use `embed_eval_single.py`
- For **MAC** results (single-vendor or mixed-vendor): use `mac_runner/embed_eval_mac.py` — evaluates both per-round doctor predictions and the final supervisor consensus.

## Citation

```bibtex
@inproceedings{yuan2026mixed,
  title={Do Mixed-Vendor Multi-Agent LLM Systems Improve Clinical Diagnosis?},
  author={Yuan, Grace Chang and Zhang, Xiaoman and Kim, Sung Eun and Rajpurkar, Pranav},
  booktitle={Proceedings of the EACL 2026 Workshop on Healthcare and Language Learning (HeaLing)},
  year={2026}
}
```
