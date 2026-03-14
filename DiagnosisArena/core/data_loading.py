# core/data_loading.py
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class DARecord:
    id: int
    date: str
    year: int
    case_info: str
    phys_exam: str
    tests: str
    final_dx: str
    options: List[str]       # for MCQ; [] if unavailable
    gold_letter: Optional[str]  # 'A'..'D' if available

    def to_prompt_text(self) -> str:
        # Compose a readable case text (shared by both modes)
        sections = []
        if self.case_info: sections.append(f"Case Information: {self.case_info.strip()}")
        if self.phys_exam: sections.append(f"Physical Examination: {self.phys_exam.strip()}")
        if self.tests:     sections.append(f"Diagnostic Tests: {self.tests.strip()}")
        return "\n".join(sections)

def _extract_year(date_val):
    year = None
    if isinstance(date_val, str):
        m = re.search(r"(19|20|21)\d{2}", date_val)
        if m:
            year = int(m.group(0))
    elif isinstance(date_val, int):
        year = date_val
    return year

def _get_options(obj) -> List[str]:
    # 1) Dict form: {"A": "...", "B": "...", "C": "...", "D": "..."}
    for k in ["Options", "options", "MCQ Options", "choices", "mcq_options"]:
        v = obj.get(k)
        if isinstance(v, dict):
            # normalize keys to upper and pull in A..D order
            ad = {str(kk).strip().upper(): str(vv).strip() for kk, vv in v.items()}
            opts = [ad.get("A",""), ad.get("B",""), ad.get("C",""), ad.get("D","")]
            if all(s for s in opts):
                return opts
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
        if isinstance(v, str) and v.strip():
            parts = [p.strip(" ;") for p in re.split(r"[\n;]", v) if p.strip()]
            if len(parts) >= 4:
                return parts[:4]
    return []

def _get_gold_letter(obj) -> Optional[str]:
    # Common “letter” keys
    for k in ["Right Option", "RightOption", "right_option",
              "Answer", "answer", "correct_option", "gold_letter", "label"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            s = v.strip().upper()
            if s in {"A", "B", "C", "D"}:
                return s
            m = re.search(r"\b([ABCD])\b", s)
            if m:
                return m.group(1)
    return None


def load_da_2024(jsonl_path: Path) -> List[DARecord]:
    out: List[DARecord] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            date = obj.get("Date", "") or obj.get("date", "")
            year = _extract_year(date)
            if year != 2024:
                continue
            rec = DARecord(
                id        = int(obj.get("id")),
                date      = date,
                year      = year,
                case_info = obj.get("Case Information", ""),
                phys_exam = obj.get("Physical Examination", ""),
                tests     = obj.get("Diagnostic Tests", ""),
                final_dx  = obj.get("Final Diagnosis", ""),
                options   = _get_options(obj),
                gold_letter = _get_gold_letter(obj),
            )
            out.append(rec)
    out.sort(key=lambda r: r.id)
    return out
