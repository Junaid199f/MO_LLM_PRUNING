"""Default calibration locations and downstream task names."""

from pathlib import Path

_DEFAULT_BO_EVAL_DIR = Path(__file__).resolve().parent / "data" / "calibration"


_DEFAULT_STAGE1_IMPORTANCE_DIR = (
    Path(__file__).resolve().parent / "data" / "calibration200"
)


BO_INVALID_SCORE_PENALTY = 1e6


DEFAULT_BO_EVAL_PROMPT_FILES = [
    str(_DEFAULT_BO_EVAL_DIR / "arc_easy_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "arc_challenge_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "biomix_qa_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "gpqa_diamond_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "gsm8k_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "hellaswag_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "humaneval_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "logiqa_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "math_500_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "med_mcqa_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "mmlu_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "ontonotes5_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "piqa_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "pubmedqa_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "truthful_qa_25.json"),
    str(_DEFAULT_BO_EVAL_DIR / "winogrande_25.json"),
]


DEFAULT_STAGE1_IMPORTANCE_PROMPT_FILES = [
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "arc_easy_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "arc_challenge_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "biomix_qa_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "gpqa_diamond_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "gsm8k_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "hellaswag_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "humaneval_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "logiqa_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "math_500_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "med_mcqa_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "mmlu_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "ontonotes5_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "piqa_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "pubmedqa_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "truthful_qa_200.json"),
    str(_DEFAULT_STAGE1_IMPORTANCE_DIR / "winogrande_200.json"),
]


DEFAULT_DOWNSTREAM_TASKS = [
    "hellaswag",
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
]
