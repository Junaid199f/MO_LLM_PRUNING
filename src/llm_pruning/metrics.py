"""Structured run summaries and CSV logging."""

from llm_pruning.export import _get_candidate_bo_score
from llm_pruning.prompt_sources import _get_second_stage_prompt_source
from llm_pruning.config import (
    DEFAULT_DOWNSTREAM_TASKS,
    DEFAULT_STAGE1_IMPORTANCE_PROMPT_FILES,
)
from llm_pruning.reporting import append_timestamp_to_path
from llm_pruning.checkpoints import _build_run_artifact_filename, _make_json_safe
import csv
import json
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


def _build_training_log_csv_path(args) -> str:
    """Build a per-run CSV log path when CSV logging is enabled."""
    if not bool(getattr(args, "training_log_csv", True)):
        return ""

    configured_path = getattr(args, "training_log_csv_path", None)
    if configured_path:
        base_path = Path(configured_path)
        if base_path.suffix.lower() != ".csv":
            base_path = base_path / "training_log.csv"
    else:
        base_path = Path("logs") / _build_run_artifact_filename(
            args,
            suffix="training_log",
            extension=".csv",
        )
    return append_timestamp_to_path(str(base_path))


def _parse_downstream_tasks(raw_tasks) -> List[str]:
    """Normalize downstream task configuration from CLI input."""
    if raw_tasks is None:
        return list(DEFAULT_DOWNSTREAM_TASKS)
    if isinstance(raw_tasks, str):
        raw_values = raw_tasks.split(",")
    elif isinstance(raw_tasks, (list, tuple)):
        raw_values = []
        for item in raw_tasks:
            if item is None:
                continue
            raw_values.extend(str(item).split(","))
    else:
        raise ValueError(f"Unsupported downstream task format: {type(raw_tasks)}")

    tasks = []
    seen = set()
    for value in raw_values:
        task = str(value).strip()
        if not task or task in seen:
            continue
        seen.add(task)
        tasks.append(task)
    return tasks or list(DEFAULT_DOWNSTREAM_TASKS)


TRAINING_LOG_DOWNSTREAM_TASK_COLUMNS = (
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "mmlu",
)

TRAINING_LOG_FIELDNAMES = [
    "timestamp_utc",
    "model_name",
    "method",
    "sparsity_rate",
    "sparsity_alpha",
    "seed",
    "stage1_objective",
    "stage1_secondary_objective",
    "stage1_objective_dataset",
    "stage1_calibration_dataset",
    "bo_eval_prompt_files",
    "stage1_importance_prompt_files",
    "penalty_constraint_added",
    "stage1_importance_penalty_weight",
    "stage2_method",
    "stage2_pruning_method",
    "stage2_components",
    "stage2_allocation_method",
    "stage2_bayes_objective",
    "stage2_importance_metric",
    "stage2_flap_metric",
    "stage2_prompt_set",
    "stage2_num_prompts",
    "stage2_prompt_max_length",
    "stage2_calibration_dataset",
    "stage2_calibration_prompt_files",
    "stage2_bayes_enabled",
    "stage2_bayes_parallel_gpus",
    "stage2_bayes_candidate_parallel_gpus",
    "stage2_constraints",
    "candidate_idx",
    "candidate_selection",
    "stage1_active_params",
    "final_active_params",
    "stage1_pruned_ratio",
    "stage2_pruned_ratio",
    "stage2_keep_ratio",
    "stage2_layer_keep_ratios",
    "stage2_layer_prune_ratios",
    "stage2_layer_stats",
    "stage1_primary_score",
    "stage1_kl",
    "stage1_ppl",
    "stage1_ppl_label",
    "bo_calibration_score",
    "calibration_mix_ppl",
    "calibration_mix_ppl_breakdown",
    "final_plot_ppl",
    "final_plot_ppl_label",
    "ppl_wikitext2",
    "ppl_c4",
    "ppl_fineweb_edu",
    "ppl_gsm8k",
    "ppl_results",
    "perplexity_parallel_gpus",
    "perplexity_batch_size",
    "downstream_enabled",
    "downstream_tasks",
    "downstream_parallel_gpus",
    "downstream_batch_size",
    "downstream_avg_acc",
    "downstream_task_accs",
    "latency",
    "final_latency",
    "postprocess_latency_time_mean_sec",
    "postprocess_latency_th_mean_tokens_s",
    "training_log_csv_path",
    "pipeline_checkpoint_path",
    "proposed_save_pareto",
]

TRAINING_LOG_FIELDNAMES.extend(
    f"downstream_{task}_acc" for task in TRAINING_LOG_DOWNSTREAM_TASK_COLUMNS
)


def _stage1_objective_dataset(args) -> str:
    first_objective = str(getattr(args, "proposed_first_objective", "") or "")
    if first_objective in {"kl_mix", "ppl_mix"}:
        return "calibration_prompt_files"
    return str(getattr(args, "calibration_dataset", ""))


def _stage1_importance_prompt_files_for_log(args) -> List[str]:
    preferred_files = [
        path for path in DEFAULT_STAGE1_IMPORTANCE_PROMPT_FILES if Path(path).is_file()
    ]
    return preferred_files or list(getattr(args, "bo_eval_prompt_files", []) or [])


def _stage2_components(args) -> List[str]:
    components = []
    if getattr(args, "second_stage_heads", False):
        components.append("heads")
    if getattr(args, "second_stage_mlp", False):
        components.append("mlp")
    if getattr(args, "second_stage_embedding", False):
        components.append("embedding")
    if getattr(args, "second_stage_norm", False):
        components.append("norm")
    return components


def _format_stage2_prompt_source(source) -> Tuple[str, List[str]]:
    if isinstance(source, (list, tuple)):
        files = [str(item) for item in source]
        labels = [Path(path).name for path in files]
        return "local_calibration:" + ",".join(labels), files
    if isinstance(source, dict):
        path = str(source.get("path", ""))
        config = source.get("config")
        split = source.get("split")
        parts = [path]
        if config:
            parts.append(str(config))
        if split:
            parts.append(str(split))
        return ":".join(part for part in parts if part), []
    return str(source), []


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, (bool, int, str)):
        return value
    return json.dumps(_make_json_safe(value), sort_keys=True)


def _append_training_log_csv(
    path_value: str, rows: Sequence[Dict[str, object]]
) -> None:
    if not path_value:
        return
    rows = [row for row in rows if row]
    if not rows:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _csv_cell(row.get(key)) for key in TRAINING_LOG_FIELDNAMES}
            )
    logging.info("Appended %d training summary row(s) to %s", len(rows), path)


def _build_training_log_row(
    args,
    candidate: Dict[str, object],
    *,
    candidate_selection: str,
    checkpoint_path: Path,
    final_ppl_results: Optional[Dict[str, float]] = None,
    downstream_avg_acc: Optional[float] = None,
    downstream_task_accs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    stage2_source, stage2_prompt_files = _format_stage2_prompt_source(
        _get_second_stage_prompt_source(args)
    )
    ppl_results = dict(final_ppl_results or {})
    candidate_ppl_results = candidate.get("final_ppl_results")
    if isinstance(candidate_ppl_results, dict):
        ppl_results.update(candidate_ppl_results)

    task_accs = downstream_task_accs
    if task_accs is None and isinstance(candidate.get("downstream_task_accs"), dict):
        task_accs = candidate.get("downstream_task_accs")
    task_accs = dict(task_accs or {})
    if downstream_avg_acc is None and candidate.get("downstream_avg_acc") is not None:
        downstream_avg_acc = candidate.get("downstream_avg_acc")

    stage2_allocation_method = candidate.get("stage2_allocation_method") or getattr(
        args, "second_stage_allocation_method", ""
    )
    stage2_importance_metric = candidate.get("stage2_importance_metric") or getattr(
        args, "second_stage_importance_metric", ""
    )
    stage1_primary_score = candidate.get("fitness")
    stage1_kl = None
    if str(args.proposed_first_objective).lower().startswith("kl"):
        stage1_kl = stage1_primary_score
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "model_name": args.model,
        "method": args.method,
        "sparsity_rate": float(args.sparsity),
        "sparsity_alpha": float(args.sparsity_alpha),
        "seed": int(args.seed),
        "stage1_objective": args.proposed_first_objective,
        "stage1_secondary_objective": args.proposed_second_objective,
        "stage1_objective_dataset": _stage1_objective_dataset(args),
        "stage1_calibration_dataset": args.calibration_dataset,
        "bo_eval_prompt_files": list(args.bo_eval_prompt_files),
        "stage1_importance_prompt_files": _stage1_importance_prompt_files_for_log(args),
        "penalty_constraint_added": bool(args.stage1_importance_penalty),
        "stage1_importance_penalty_weight": float(
            args.stage1_importance_penalty_weight
        ),
        "stage2_method": args.second_stage_method,
        "stage2_pruning_method": args.second_stage_method,
        "stage2_components": _stage2_components(args),
        "stage2_allocation_method": stage2_allocation_method,
        "stage2_bayes_objective": args.second_stage_bayes_objective,
        "stage2_importance_metric": stage2_importance_metric,
        "stage2_flap_metric": args.second_stage_flap_metric,
        "stage2_prompt_set": args.second_stage_prompt_set,
        "stage2_num_prompts": int(args.second_stage_num_prompts),
        "stage2_prompt_max_length": int(args.second_stage_prompt_max_length),
        "stage2_calibration_dataset": stage2_source,
        "stage2_calibration_prompt_files": stage2_prompt_files,
        "stage2_bayes_enabled": bool(args.second_stage_bayes),
        "stage2_bayes_parallel_gpus": args.second_stage_bayes_parallel_gpus,
        "stage2_bayes_candidate_parallel_gpus": args.second_stage_bayes_candidate_parallel_gpus,
        "stage2_constraints": {
            "min_keep_ratio": args.second_stage_bayes_min_keep_ratio,
            "early_layer_count": args.second_stage_bayes_early_layer_count,
            "early_min_keep_ratio": args.second_stage_bayes_early_min_keep_ratio,
            "early_max_prune_factor": args.second_stage_bayes_early_max_prune_factor,
        },
        "candidate_idx": candidate.get("candidate_idx"),
        "candidate_selection": candidate_selection,
        "stage1_active_params": candidate.get("active_params"),
        "final_active_params": candidate.get(
            "final_active_params", candidate.get("active_params")
        ),
        "stage1_pruned_ratio": candidate.get("stage1_pruned_ratio"),
        "stage2_pruned_ratio": candidate.get("stage2_pruned_ratio"),
        "stage2_keep_ratio": candidate.get("stage2_keep_ratio"),
        "stage2_layer_keep_ratios": candidate.get("stage2_layer_keep_ratios"),
        "stage2_layer_prune_ratios": candidate.get("stage2_layer_prune_ratios"),
        "stage2_layer_stats": candidate.get("stage2_layer_stats"),
        "stage1_primary_score": stage1_primary_score,
        "stage1_kl": stage1_kl,
        "stage1_ppl": candidate.get("stage1_ppl"),
        "stage1_ppl_label": candidate.get("stage1_ppl_label"),
        "bo_calibration_score": _get_candidate_bo_score(candidate),
        "calibration_mix_ppl": candidate.get("calibration_mix_ppl"),
        "calibration_mix_ppl_breakdown": candidate.get("calibration_mix_ppl_breakdown"),
        "final_plot_ppl": candidate.get("final_plot_ppl"),
        "final_plot_ppl_label": candidate.get("final_plot_ppl_label"),
        "ppl_wikitext2": ppl_results.get("wikitext2"),
        "ppl_c4": ppl_results.get("c4"),
        "ppl_fineweb_edu": ppl_results.get("fineweb_edu"),
        "ppl_gsm8k": ppl_results.get("gsm8k"),
        "ppl_results": ppl_results,
        "perplexity_parallel_gpus": args.perplexity_parallel_gpus,
        "perplexity_batch_size": args.perplexity_batch_size,
        "downstream_enabled": bool(args.evaluate_downstream),
        "downstream_tasks": _parse_downstream_tasks(args.downstream_tasks),
        "downstream_parallel_gpus": args.downstream_parallel_gpus,
        "downstream_batch_size": args.downstream_batch_size,
        "downstream_avg_acc": downstream_avg_acc,
        "downstream_task_accs": task_accs,
        "latency": candidate.get("latency"),
        "final_latency": candidate.get("final_latency"),
        "postprocess_latency_time_mean_sec": candidate.get(
            "postprocess_latency_time_mean_sec"
        ),
        "postprocess_latency_th_mean_tokens_s": candidate.get(
            "postprocess_latency_th_mean_tokens_s"
        ),
        "training_log_csv_path": getattr(args, "training_log_csv_path", ""),
        "pipeline_checkpoint_path": str(checkpoint_path),
        "proposed_save_pareto": args.proposed_save_pareto,
    }
    for task in TRAINING_LOG_DOWNSTREAM_TASK_COLUMNS:
        row[f"downstream_{task}_acc"] = task_accs.get(task)
    return row


def _training_log_candidate_rows(
    args,
    mask_candidates: Sequence[Dict[str, object]],
    *,
    checkpoint_path: Path,
    skip_single_candidate_postprocessing: bool,
    report_sorted_final_candidates: bool,
    report_all_final_candidates: bool,
    last_idx: Optional[int],
    final_ppl_results: Optional[Dict[str, float]],
    downstream_avg_acc: Optional[float],
    downstream_task_accs: Optional[Dict[str, float]],
) -> List[Dict[str, object]]:
    rows = []
    if report_sorted_final_candidates:
        selected = sorted(mask_candidates, key=_candidate_latency_sort_key)
        selection_label = "sorted_final_candidate"
    elif report_all_final_candidates:
        selected = list(mask_candidates)
        selection_label = "all_final_candidate"
    elif last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
        selected = [mask_candidates[int(last_idx) - 1]]
        selection_label = "selected_candidate"
    elif skip_single_candidate_postprocessing and len(mask_candidates) == 1:
        selected = [mask_candidates[0]]
        selection_label = "selected_candidate"
    else:
        selected = []

    for candidate in selected:
        selected_idx = candidate.get("candidate_idx")
        use_standalone_downstream = (
            len(selected) == 1
            and downstream_task_accs is not None
            and not candidate.get("downstream_task_accs")
        )
        use_standalone_ppl = (
            len(selected) == 1
            and final_ppl_results is not None
            and not candidate.get("final_ppl_results")
        )
        rows.append(
            _build_training_log_row(
                args,
                candidate,
                candidate_selection=(
                    f"{selection_label}:{selected_idx}"
                    if selected_idx is not None
                    else selection_label
                ),
                checkpoint_path=checkpoint_path,
                final_ppl_results=final_ppl_results if use_standalone_ppl else None,
                downstream_avg_acc=(
                    downstream_avg_acc if use_standalone_downstream else None
                ),
                downstream_task_accs=(
                    downstream_task_accs if use_standalone_downstream else None
                ),
            )
        )
    return rows


def _candidate_latency_sort_key(
    candidate: Dict[str, object],
) -> Tuple[float, float, int]:
    latency = candidate.get("final_latency")
    ppl = _get_candidate_bo_score(candidate)
    idx = int(candidate.get("candidate_idx", 0) or 0)
    latency_key = float(latency) if latency is not None else float("inf")
    ppl_key = float(ppl) if ppl is not None else float("inf")
    return latency_key, ppl_key, idx
