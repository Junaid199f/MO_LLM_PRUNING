"""Resolve calibration sources for each pruning stage."""

from llm_pruning.config import (
    DEFAULT_BO_EVAL_PROMPT_FILES,
    DEFAULT_STAGE1_IMPORTANCE_PROMPT_FILES,
    _DEFAULT_STAGE1_IMPORTANCE_DIR,
)
import logging
from pathlib import Path
from typing import List


def _resolve_stage1_importance_prompt_files(args) -> List[str]:
    """Prefer the dedicated calibration200 prompt set for stage-1 importance scoring."""
    preferred_files = [
        path for path in DEFAULT_STAGE1_IMPORTANCE_PROMPT_FILES if Path(path).is_file()
    ]
    if preferred_files:
        logging.info(
            "Stage-1 importance penalty will use calibration200 prompt files from %s.",
            _DEFAULT_STAGE1_IMPORTANCE_DIR,
        )
        missing_files = [
            path
            for path in DEFAULT_STAGE1_IMPORTANCE_PROMPT_FILES
            if not Path(path).is_file()
        ]
        if missing_files:
            logging.warning(
                "Stage-1 importance penalty found only %d/%d calibration200 files; "
                "using the available subset.",
                len(preferred_files),
                len(DEFAULT_STAGE1_IMPORTANCE_PROMPT_FILES),
            )
        return preferred_files

    logging.warning(
        "No calibration200 prompt files were found for stage-1 importance penalty. "
        "Falling back to --bo_eval_prompt_files."
    )
    return list(args.bo_eval_prompt_files)


def _get_second_stage_prompt_source(args):
    calibration_prompt_files = [
        path for path in DEFAULT_BO_EVAL_PROMPT_FILES if Path(path).is_file()
    ] or list(args.bo_eval_prompt_files)
    scoring_method = (
        str(getattr(args, "second_stage_method", "") or "")
        .strip()
        .lower()
        .replace("_", "-")
    )
    if (
        scoring_method in {"importance", "wanda", "wanda-sp", "wandasp", "flap"}
        and args.second_stage_prompt_set == "calibration_mix"
    ):
        return calibration_prompt_files

    prompt_source = "yahma/alpaca-cleaned"
    if args.second_stage_prompt_set == "calibration_mix":
        prompt_source = calibration_prompt_files
    elif args.second_stage_prompt_set == "c4_train":
        prompt_source = {"path": "allenai/c4", "config": "en", "split": "train"}
    elif args.second_stage_prompt_set == "wikitext2":
        prompt_source = {
            "path": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "train",
        }
    elif args.second_stage_prompt_set == "fineweb_edu":
        prompt_source = {
            "path": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "split": "train",
        }
    return prompt_source
