# core/prompts.py
# ===== Open-ended (Top-5) prompt =====
PROMPT_TEMPLATE = (
    "As a medical expert, please make a diagnosis for the patient's disease based on the case information, "
    "physical examination, and diagnostic tests. Please enumerate the top 5 most likely diagnoses for the "
    "following patient in order, with the most likely disease listed first.\n"
    "Case Information:\n"
    "{case}\n"
    "Physical Examination:\n"
    "{exam}\n"
    "Diagnostic tests:\n"
    "{tests}\n"
    "Output the diagnosis in numeric order, one per line. For example:\n"
    "1. Disease A;\n"
    "2. Disease B;"
)
