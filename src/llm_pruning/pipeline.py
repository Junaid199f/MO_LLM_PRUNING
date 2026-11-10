"""Two-stage pruning orchestration, candidate replay, and parallel workers."""

from llm_pruning.selection import (
    _evaluate_all_final_candidates_downstream,
    _log_all_final_candidates,
    _log_final_pareto_summary,
)

from llm_pruning.recovery import (
    _resolve_recovery_finetune_gpu_ids,
    _run_recovery_finetune_ddp,
)

from llm_pruning.stage2_workers import _evaluate_stage2_bo_candidates_parallel

from llm_pruning.metrics import (
    _append_training_log_csv,
    _build_training_log_csv_path,
    _candidate_latency_sort_key,
    _parse_downstream_tasks,
    _training_log_candidate_rows,
)

from llm_pruning.evaluation_workers import (
    _evaluate_candidate_perplexity_parallel,
    _evaluate_model_dir_downstream_parallel,
    _evaluate_model_dir_perplexity_parallel,
    _extract_accuracy,
)

from llm_pruning.export import (
    _resolve_selected_stage2_export_candidate_idx,
    _save_selected_stage2_model,
)

from llm_pruning.replay import (
    _load_replayed_recovery_model,
    _replay_candidate_on_fresh_model,
)

from llm_pruning.prompt_sources import (
    _get_second_stage_prompt_source,
    _resolve_stage1_importance_prompt_files,
)


from llm_pruning.advanced_cli import parse_args
from llm_pruning.reporting import (
    _layer_prune_ratios_from_keep_ratios,
    ensure_directory_exists,
    save_layer_component_pruning_plot,
    save_layerwise_allocation_plot,
    save_stage2_pareto_refinement_plot,
    save_stagewise_pruning_plot,
)
from llm_pruning.allocation import (
    _build_stage2_keep_ratios_for_method,
    _resolve_model_compute_device,
)
from llm_pruning.bayesian import (
    _bayes_optimize_layer_keep_ratios,
    _bayes_optimize_recovery_lora,
    load_tokenizer,
)
from llm_pruning.candidates import (
    _parse_layer_keep_ratios,
    build_mask_candidates,
    load_proposed_history_points,
)
from llm_pruning.calibration import (
    _compute_stage1_block_importance_scores,
    _evaluate_perplexity_suite_local,
    build_calibration,
    build_prompt_mix_corpus,
    prepare_bo_prompt_eval_inputs,
    prepare_c4_validation_inputs,
    prepare_fineweb_edu_inputs,
    prepare_gsm8k_inputs,
    prepare_prompt_mix_inputs,
    prepare_wikitext_inputs,
)
from llm_pruning.checkpoints import (
    _build_pipeline_checkpoint_path,
    _build_pipeline_checkpoint_signature,
    _candidate_stage2_completed,
    _load_pipeline_checkpoint,
    _new_pipeline_checkpoint,
    _persist_pipeline_checkpoint,
)
import copy
import gc
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from typing import Dict, List, Optional, Sequence, Tuple

from llm_pruning.latency import measure_latency
from llm_pruning.refinement.second_stage import second_stage_optimize_heads
from llm_pruning.fine_tune import fine_tune_model
from llm_pruning.core.evaluation import (
    evaluate_perplexity,
    evaluate_inference_time,
    evaluation_downstream,
    evaluation_ppl,
)
from llm_pruning.core.structural_pruning import apply_structural_pruning
from llm_pruning.core.utilities import (
    count_params_with_masks,
    get_model_layers,
    get_parameter_counts,
    loadModel,
    maskModel,
    printModelStats,
    set_seed,
    unmaskModel,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_candidate_latency_score(candidate: Dict[str, object]) -> Optional[float]:
    """Return the best available latency score for candidate selection."""
    for key in ("bo_latency_score", "final_latency", "latency"):
        value = candidate.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _get_candidate_ppl_selection_score(candidate: Dict[str, object]) -> Optional[float]:
    """Return a PPL/NLL-like score for best-PPL candidate selection."""
    for key in (
        "calibration_mix_ppl",
        "final_plot_ppl",
        "bo_calibration_ppl",
        "bo_calibration_score",
    ):
        value = candidate.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _get_text_field(dataset, candidates):
    for field in candidates:
        if field in dataset.column_names:
            return field
    return None


def convert_sparsity_to_blocks(sparsity: float, num_blocks: int) -> int:
    """Convert sparsity ratio into a concrete number of blocks."""
    if sparsity < 0 or sparsity > 1:
        raise ValueError(f"Sparsity must be between 0 and 1; got {sparsity}.")
    target_blocks = sparsity * num_blocks
    if not target_blocks.is_integer():
        logging.warning(
            "Requested sparsity does not align with block count; rounding %.4f blocks up to nearest integer.",
            target_blocks,
        )
        target_blocks = math.ceil(target_blocks)
    return int(target_blocks)


def _generate_random_layer_keep_ratios(num_layers: int, seed: int) -> List[float]:
    """Generate a deterministic random keep ratio per layer."""
    if num_layers <= 0:
        return []
    rng = np.random.default_rng(seed)
    return [float(value) for value in rng.uniform(0.0, 1.0, size=num_layers)]


def _resolve_second_stage_allocation_method(args, layer_keep_ratios) -> str:
    """Resolve the stage-2 allocation strategy with BO compatibility."""
    requested = (
        str(getattr(args, "second_stage_allocation_method", "auto") or "auto")
        .strip()
        .lower()
    )
    if requested == "auto":
        if getattr(args, "second_stage_bayes", False):
            return "proposed"
        if layer_keep_ratios == "random":
            return "uniform"
        if layer_keep_ratios is not None:
            return "manual"
        return "uniform"
    if requested == "proposed":
        return "proposed"
    if requested == "manual" and layer_keep_ratios is None:
        raise ValueError(
            "second_stage_allocation_method=manual requires --second_stage_layer_keep_ratios."
        )
    return requested


def _resolve_stage2_beta_value(args) -> float:
    """Resolve a beta/alpha scalar for 1D layerwise redistribution baselines."""
    beta = (
        args.second_stage_allocation_alpha
        if args.second_stage_allocation_alpha is not None
        else args.second_stage_allocation_beta
    )
    if args.second_stage_allocation_alpha is not None:
        logging.info(
            "Using second_stage_allocation_alpha=%.4f as the beta scalar for stage-2 beta allocation.",
            float(beta),
        )
    return min(1.0, max(0.0, float(beta)))


def describe_candidate(idx: int, total: int, candidate: dict) -> str:
    """Generate a concise text description for a mask candidate."""
    parts = [f"Candidate {idx}/{total}"]
    fitness = candidate.get("fitness")
    active = candidate.get("active_params")
    if fitness is not None:
        parts.append(f"fitness={fitness:.6f}")
    if active is not None:
        parts.append(f"active_params={int(active)}")
    return " | ".join(parts)


def _summarize_downstream_results(
    results: Optional[Dict[str, object]],
    tasks: Sequence[str],
) -> Tuple[Optional[float], Dict[str, float]]:
    if not isinstance(results, dict):
        return None, {}
    task_results = results.get("results", {})
    if not isinstance(task_results, dict):
        return None, {}
    per_task: Dict[str, float] = {}
    scores: List[float] = []
    for task in tasks:
        task_res = task_results.get(task)
        if isinstance(task_res, dict):
            score = _extract_accuracy(task_res)
        else:
            prefix = f"{task}_"
            grouped_scores = [
                _extract_accuracy(value)
                for key, value in task_results.items()
                if isinstance(key, str)
                and key.startswith(prefix)
                and isinstance(value, dict)
            ]
            if not grouped_scores:
                continue
            score = sum(grouped_scores) / len(grouped_scores)
        per_task[task] = score
        scores.append(score)
    avg_acc = sum(scores) / len(scores) if scores else None
    return avg_acc, per_task


@torch.no_grad()
def main(args=None):
    """Entry point for the CLI."""
    args = args if args is not None else parse_args()
    logging.basicConfig(
        level=getattr(logging, args.logging.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_directory_exists("graphs")
    args.training_log_csv_path = _build_training_log_csv_path(args)
    if args.training_log_csv and args.training_log_csv_path:
        logging.info(
            "Training summary CSV for this run: %s", args.training_log_csv_path
        )

    checkpoint_path = _build_pipeline_checkpoint_path(args)
    checkpoint_signature = _build_pipeline_checkpoint_signature(args)
    checkpoint_state = None
    if not args.pipeline_checkpoint and args.checkpoint_eval_only:
        raise RuntimeError("--checkpoint_eval_only requires --pipeline_checkpoint.")
    if not args.pipeline_checkpoint:
        logging.info(
            "Pipeline checkpoint disabled; only CSV/plot artifacts will be written."
        )
    elif args.force_fresh_run:
        logging.info(
            "force_fresh_run enabled: ignoring any existing pipeline checkpoint at %s.",
            checkpoint_path.resolve(),
        )
    else:
        checkpoint_state = _load_pipeline_checkpoint(
            checkpoint_path, checkpoint_signature
        )
    if checkpoint_state is None:
        checkpoint_state = _new_pipeline_checkpoint(
            checkpoint_path, checkpoint_signature
        )
    else:
        logging.info("Resuming from pipeline checkpoint %s", checkpoint_path.resolve())
        if not checkpoint_state.pop("_stage2_signature_matches", True):
            restored_candidates = copy.deepcopy(
                checkpoint_state.get("mask_candidates", []) or []
            )
            for candidate in restored_candidates:
                candidate["_stage2_completed"] = False
            checkpoint_state["mask_candidates"] = restored_candidates
            checkpoint_state["stage2"]["completed"] = False
            checkpoint_state["stage2"]["completed_candidate_indices"] = []
            checkpoint_state["stage2"]["stagewise_plot_records"] = []
            checkpoint_state["selection"]["completed"] = False
            checkpoint_state["selection"]["selected_candidate_idx"] = None
            checkpoint_state["selection"]["best_idx"] = None
            checkpoint_state["selection"]["last_idx"] = None
            _persist_pipeline_checkpoint(
                checkpoint_state,
                checkpoint_path,
                args=args,
                mask_candidates=restored_candidates,
                stage1_completed=bool(
                    checkpoint_state.get("stage1", {}).get("completed")
                ),
                stage1_importance_scores=checkpoint_state.get("stage1", {}).get(
                    "importance_scores"
                ),
                stage2_completed=False,
                stagewise_plot_records=[],
            )
        saved_paths = checkpoint_state.get("paths", {})
        if saved_paths.get("proposed_save_pareto"):
            args.proposed_save_pareto = saved_paths["proposed_save_pareto"]
        if saved_paths.get("proposed_pareto_plot"):
            args.proposed_pareto_plot = saved_paths["proposed_pareto_plot"]

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)

    calibration_corpus, calibration_dataset = build_calibration(
        tokenizer,
        args.local_datasets,
        args.num_calibration_samples,
        args.seq_len,
        args.calibration_dataset,
    )
    first_calibration_sample = calibration_dataset[0]
    wikitext_input_ids = None
    c4_val_input_ids = None
    fineweb_edu_input_ids = None
    bo_eval_datasets = None
    stage1_importance_eval_datasets = None
    gsm8k_input_ids = None
    stage1_calibration_corpus = calibration_corpus
    stage1_first_objective = args.proposed_first_objective
    stage1_ppl_input_ids = None

    need_wikitext = (
        args.method == "proposed"
        or args.evaluate_perplexity
        or args.evaluate_perplexity_stage1
        or args.evaluate_perplexity_all
        or args.proposed_first_objective == "perplexity"
    )
    if need_wikitext:
        wikitext_input_ids = prepare_wikitext_inputs(tokenizer, args.local_datasets)
    if args.method == "proposed" or args.second_stage_bayes or args.recovery_bayes:
        bo_eval_datasets = prepare_bo_prompt_eval_inputs(
            tokenizer, args.bo_eval_prompt_files
        )
    if args.evaluate_perplexity_all:
        c4_val_input_ids = prepare_c4_validation_inputs(tokenizer, args.local_datasets)
        fineweb_edu_input_ids = prepare_fineweb_edu_inputs(
            tokenizer, args.local_datasets
        )
        gsm8k_input_ids = prepare_gsm8k_inputs(tokenizer, args.local_datasets)

    if args.proposed_first_objective == "kl_mix":
        stage1_calibration_corpus = build_prompt_mix_corpus(args.bo_eval_prompt_files)
        stage1_first_objective = "kl"
    elif args.proposed_first_objective == "ppl_mix":
        stage1_ppl_input_ids = prepare_prompt_mix_inputs(
            tokenizer, args.bo_eval_prompt_files
        )
        stage1_first_objective = "perplexity"
    else:
        stage1_ppl_input_ids = wikitext_input_ids

    model = loadModel(args.model, args.cache_dir)
    printModelStats(model, "Original model")
    num_blocks = len(get_model_layers(model))
    second_stage_layer_keep_ratios = _parse_layer_keep_ratios(
        args.second_stage_layer_keep_ratios
    )
    if second_stage_layer_keep_ratios == "random":
        second_stage_layer_keep_ratios = _generate_random_layer_keep_ratios(
            num_blocks, args.seed
        )
        logging.info(
            "Generated random second-stage keep ratios for %d layers (seed=%d).",
            num_blocks,
            args.seed,
        )
    second_stage_allocation_method = _resolve_second_stage_allocation_method(
        args,
        second_stage_layer_keep_ratios,
    )
    if args.second_stage_bayes and second_stage_allocation_method != "proposed":
        logging.warning(
            "Legacy --second_stage_bayes is ignored because second_stage_allocation_method=%s was set.",
            second_stage_allocation_method,
        )
        args.second_stage_bayes = False
        args.second_stage_bayes_parallel_gpus = ""
        args.second_stage_bayes_candidate_parallel_gpus = ""
    if args.second_stage_bayes_objective == "latency" and not torch.cuda.is_available():
        raise RuntimeError("--second_stage_bayes_objective=latency requires CUDA.")
    if args.second_stage_bayes_latency_batch_size < 1:
        logging.warning(
            "second_stage_bayes_latency_batch_size must be >= 1; clamping to 1."
        )
        args.second_stage_bayes_latency_batch_size = 1
    if args.second_stage_bayes_latency_seq_len < 1:
        logging.warning(
            "second_stage_bayes_latency_seq_len must be >= 1; clamping to 1."
        )
        args.second_stage_bayes_latency_seq_len = 1
    if args.second_stage_bayes_latency_runs < 1:
        logging.warning("second_stage_bayes_latency_runs must be >= 1; clamping to 1.")
        args.second_stage_bayes_latency_runs = 1
    if args.second_stage_bayes_latency_warmup < 0:
        logging.warning(
            "second_stage_bayes_latency_warmup must be >= 0; clamping to 0."
        )
        args.second_stage_bayes_latency_warmup = 0
    logging.info(
        "Resolved stage-2 allocation method: %s.",
        second_stage_allocation_method,
    )

    if not (0.0 <= args.sparsity <= 1.0):
        raise ValueError(
            f"Total sparsity must be between 0 and 1; got {args.sparsity}."
        )
    if args.sparsity_alpha < 0.0 or args.sparsity_alpha > 1.0:
        logging.warning("sparsity_alpha outside [0,1]; clamping to bounds.")
        args.sparsity_alpha = min(1.0, max(0.0, args.sparsity_alpha))
    if (
        args.second_stage_bayes_min_keep_ratio < 0.0
        or args.second_stage_bayes_min_keep_ratio >= 1.0
    ):
        logging.warning(
            "second_stage_bayes_min_keep_ratio outside [0,1); clamping to valid range."
        )
        args.second_stage_bayes_min_keep_ratio = min(
            0.999999,
            max(0.0, args.second_stage_bayes_min_keep_ratio),
        )
    if args.second_stage_bayes_early_min_keep_ratio is not None:
        if (
            args.second_stage_bayes_early_min_keep_ratio < 0.0
            or args.second_stage_bayes_early_min_keep_ratio >= 1.0
        ):
            logging.warning(
                "second_stage_bayes_early_min_keep_ratio outside [0,1); clamping to valid range."
            )
            args.second_stage_bayes_early_min_keep_ratio = min(
                0.999999,
                max(0.0, args.second_stage_bayes_early_min_keep_ratio),
            )
        if (
            args.second_stage_bayes_early_min_keep_ratio
            < args.second_stage_bayes_min_keep_ratio
        ):
            logging.warning(
                "second_stage_bayes_early_min_keep_ratio is lower than second_stage_bayes_min_keep_ratio; "
                "raising it to the global BO minimum."
            )
            args.second_stage_bayes_early_min_keep_ratio = (
                args.second_stage_bayes_min_keep_ratio
            )
    if args.second_stage_bayes_early_layer_count < 0:
        logging.warning(
            "second_stage_bayes_early_layer_count must be >= 0; clamping to 0."
        )
        args.second_stage_bayes_early_layer_count = 0
    if (
        args.second_stage_bayes_early_max_prune_factor < 0.0
        or args.second_stage_bayes_early_max_prune_factor > 1.0
    ):
        logging.warning(
            "second_stage_bayes_early_max_prune_factor outside [0,1]; clamping to valid range."
        )
        args.second_stage_bayes_early_max_prune_factor = min(
            1.0,
            max(0.0, args.second_stage_bayes_early_max_prune_factor),
        )

    stage1_checkpoint_completed = bool(
        checkpoint_state.get("stage1", {}).get("completed")
    )
    saved_mask_candidates = checkpoint_state.get("mask_candidates", []) or []
    if args.checkpoint_eval_only:
        if not checkpoint_state.get("stage2", {}).get("completed"):
            raise RuntimeError(
                "checkpoint_eval_only requires a completed stage-2 checkpoint."
            )
        selected_candidate_idx = checkpoint_state.get("selection", {}).get(
            "selected_candidate_idx"
        )
        if selected_candidate_idx is None:
            if len(saved_mask_candidates) == 1:
                selected_candidate_idx = 1
            else:
                raise RuntimeError(
                    "Checkpoint does not record a selected candidate. Complete the full run first "
                    "or use a checkpoint containing only one final candidate."
                )
        if selected_candidate_idx < 1 or selected_candidate_idx > len(
            saved_mask_candidates
        ):
            raise RuntimeError(
                f"Checkpoint selected_candidate_idx={selected_candidate_idx} is out of range."
            )
        logging.info(
            "Checkpoint eval-only enabled: loading selected candidate %d from %s.",
            selected_candidate_idx,
            checkpoint_path.resolve(),
        )
        mask_candidates = [
            copy.deepcopy(saved_mask_candidates[selected_candidate_idx - 1])
        ]
        mask_candidates[0].setdefault("candidate_idx", selected_candidate_idx)
        mask_candidates[0]["_stage2_completed"] = True
    elif saved_mask_candidates and (
        checkpoint_state.get("stage1", {}).get("completed")
        or any(
            _candidate_stage2_completed(candidate)
            for candidate in saved_mask_candidates
        )
    ):
        mask_candidates = copy.deepcopy(saved_mask_candidates)
        logging.info(
            "Checkpointed candidate state detected at %s; skipping stage-1 search and resuming with %d saved candidate(s).",
            checkpoint_path.resolve(),
            len(mask_candidates),
        )
    # Stage-2 only mode: skip stage-1 pruning/search and start from an unpruned mask.
    elif args.stage2_only:
        logging.info(
            "Stage-2 only enabled: skipping stage-1 pruning/search and using full budget %.4f.",
            args.sparsity,
        )
        logging.info(
            "Stage-2 only mode ignores stage-1 search settings such as --proposed_pop_size and --proposed_n_gen."
        )
        mask_candidates = [{"attn": [0] * num_blocks, "mlp": [0] * num_blocks}]
    else:
        stage1_sparsity = args.sparsity * args.sparsity_alpha
        target_blocks = convert_sparsity_to_blocks(stage1_sparsity, num_blocks)
        logging.info(
            (
                "Using total sparsity %.4f with stage-1 target %.4f (blocks=%d/%d). "
                "Stage-2 keep ratio is derived from remaining budget after stage-1."
            ),
            args.sparsity,
            stage1_sparsity,
            target_blocks,
            num_blocks,
        )
        stage1_importance_scores = None
        if (
            args.method == "proposed"
            and args.stage1_importance_penalty
            and not args.proposed_eval_saved_pareto
            and not args.proposed_load_solutions
        ):
            cached_stage1_importance_scores = checkpoint_state.get("stage1", {}).get(
                "importance_scores"
            )
            cached_scores_valid = (
                isinstance(cached_stage1_importance_scores, dict)
                and len(cached_stage1_importance_scores.get("attn", [])) == num_blocks
                and len(cached_stage1_importance_scores.get("mlp", [])) == num_blocks
            )
            if cached_scores_valid:
                stage1_importance_scores = copy.deepcopy(
                    cached_stage1_importance_scores
                )
                logging.info(
                    "Reusing cached stage-1 global importance scores from %s.",
                    checkpoint_path.resolve(),
                )
            else:
                if stage1_importance_eval_datasets is None:
                    stage1_importance_eval_datasets = prepare_bo_prompt_eval_inputs(
                        tokenizer,
                        _resolve_stage1_importance_prompt_files(args),
                    )
                if not stage1_importance_eval_datasets:
                    raise RuntimeError(
                        "stage1_importance_penalty requires calibration prompt datasets. "
                        "Ensure calibration200 or --bo_eval_prompt_files points to valid prompt JSON files."
                    )
                stage1_importance_scores = _compute_stage1_block_importance_scores(
                    model,
                    stage1_importance_eval_datasets,
                    num_blocks=num_blocks,
                    seq_len=args.seq_len,
                    device=str(_resolve_model_compute_device(model)),
                )
                _persist_pipeline_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    args=args,
                    mask_candidates=saved_mask_candidates,
                    stage1_completed=stage1_checkpoint_completed,
                    stage1_importance_scores=stage1_importance_scores,
                )
                logging.info(
                    "Cached stage-1 global importance scores to %s for resume safety.",
                    checkpoint_path.resolve(),
                )
        mask_candidates = build_mask_candidates(
            args,
            model,
            tokenizer,
            stage1_calibration_corpus,
            first_calibration_sample,
            target_blocks,
            stage1_first_objective,
            stage1_ppl_input_ids,
            stage1_importance_scores,
        )
        for candidate in mask_candidates:
            candidate.setdefault("_stage2_completed", False)
        _persist_pipeline_checkpoint(
            checkpoint_state,
            checkpoint_path,
            args=args,
            mask_candidates=mask_candidates,
            stage1_completed=True,
            stage1_importance_scores=stage1_importance_scores,
            stage2_completed=False,
            stagewise_plot_records=[],
        )
        stage1_checkpoint_completed = True
        logging.info(
            "Stage-1 completed; checkpoint saved to %s.",
            checkpoint_path.resolve(),
        )

    for candidate in mask_candidates:
        candidate.setdefault("_stage2_completed", False)

    if args.stage1_only:
        _persist_pipeline_checkpoint(
            checkpoint_state,
            checkpoint_path,
            args=args,
            mask_candidates=mask_candidates,
            stage1_completed=stage1_checkpoint_completed,
            stage2_completed=False,
        )
        if args.pipeline_checkpoint:
            logging.info(
                "stage1_only enabled: saved %d Stage-1 candidate(s) to %s and exiting before Stage 2.",
                len(mask_candidates),
                checkpoint_path.resolve(),
            )
        else:
            logging.info(
                "stage1_only enabled: wrote Stage-1 CSV/plot artifacts for %d candidate(s) and exiting before Stage 2.",
                len(mask_candidates),
            )
        return

    all_stage2_candidates_restored = bool(mask_candidates) and all(
        _candidate_stage2_completed(candidate) for candidate in mask_candidates
    )

    if (
        not args.checkpoint_eval_only
        and not args.stage2_only
        and not all_stage2_candidates_restored
        and (
            args.second_stage_heads
            or args.second_stage_mlp
            or args.second_stage_embedding
            or args.second_stage_norm
        )
    ):
        logging.info(
            "Reloading the base model after stage-1 search so stage-2 starts from a clean memory state."
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        model = loadModel(args.model, args.cache_dir)

    if not mask_candidates:
        logging.error("No pruning masks were generated.")
        return

    structural_pruned = False
    # Optionally apply irreversible structural pruning based on the selected mask.
    if args.method == "proposed" and args.proposed_structural_prune:
        if len(mask_candidates) > 1:
            logging.warning(
                "Structural pruning only supports a single candidate; using the first mask."
            )
            mask_candidates = [mask_candidates[0]]
        else:
            logging.info("Applying structural pruning based on proposed mask.")
            apply_structural_pruning(
                model,
                mask_candidates[0]["attn"],
                mask_candidates[0]["mlp"],
            )
            structural_pruned = True

    logging.info("Evaluating %d candidate(s).", len(mask_candidates))
    report_sorted_final_candidates = (
        args.method == "proposed"
        and args.proposed_pareto_selection == "sorted"
        and len(mask_candidates) > 1
    )
    report_all_final_candidates = (
        args.method == "proposed"
        and args.proposed_pareto_selection == "all"
        and len(mask_candidates) > 1
    )
    evaluate_pareto_candidate_ppl = (
        args.method == "proposed"
        and len(mask_candidates) > 1
        and bo_eval_datasets is not None
    )
    measure_final_candidate_latency = (
        args.method == "proposed" and len(mask_candidates) > 1
    )
    measure_stage1_plot_ppl = evaluate_pareto_candidate_ppl
    select_best_final_candidate = (
        args.method == "proposed"
        and args.proposed_pareto_selection in {"best_ppl", "best_latency"}
        and len(mask_candidates) > 1
    )
    selection_metric = (
        "latency" if args.proposed_pareto_selection == "best_latency" else "ppl"
    )
    skip_single_candidate_postprocessing = False
    performed_mode_downstream_eval = False
    pending_selected_candidate_downstream_eval = False
    auto_resume_single_candidate_downstream_eval = (
        all_stage2_candidates_restored and len(mask_candidates) == 1
    )
    final_ppl_results: Dict[str, float] = {}
    standalone_downstream_avg_acc: Optional[float] = None
    standalone_downstream_task_accs: Optional[Dict[str, float]] = None
    warned_final_latency_skipped = False
    stagewise_plot_records = copy.deepcopy(
        checkpoint_state.get("stage2", {}).get("stagewise_plot_records", [])
    )
    best_score = float("inf")
    best_idx = None
    best_attn_mask = None
    best_mlp_mask = None
    last_idx = None
    last_attn_mask = None
    last_mlp_mask = None
    last_tuned_model = None
    last_tuned_dir = None
    model_matches_last_candidate = False
    second_stage_enabled = (
        args.second_stage_heads
        or args.second_stage_mlp
        or args.second_stage_embedding
        or args.second_stage_norm
    )
    candidate_parallel_stage2_bo = (
        args.method == "proposed"
        and len(mask_candidates) > 1
        and second_stage_enabled
        and second_stage_allocation_method == "proposed"
        and bool(str(args.second_stage_bayes_candidate_parallel_gpus).strip())
        and not all_stage2_candidates_restored
        and not structural_pruned
    )
    if candidate_parallel_stage2_bo:
        if str(args.second_stage_bayes_parallel_gpus).strip():
            logging.info(
                "Ignoring trial-level --second_stage_bayes_parallel_gpus during candidate-parallel BO; "
                "each Pareto candidate owns one GPU and runs an independent BO loop."
            )
        logging.info(
            "Candidate-parallel stage-2 BO requested; freeing the main model before launching BO workers."
        )
        try:
            del model
        except UnboundLocalError:
            pass
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        try:
            parallel_stage2_results = _evaluate_stage2_bo_candidates_parallel(
                mask_candidates,
                args,
                num_blocks=num_blocks,
                bo_eval_datasets=bo_eval_datasets,
                prompt_source=_get_second_stage_prompt_source(args),
                evaluate_pareto_candidate_ppl=evaluate_pareto_candidate_ppl,
                measure_stage1_plot_ppl=measure_stage1_plot_ppl,
                measure_final_candidate_latency=measure_final_candidate_latency,
            )
            completed_parallel_indices = {
                int(result["candidate_idx"]) for result in parallel_stage2_results
            }
            for result in parallel_stage2_results:
                result_idx = int(result["candidate_idx"])
                mask_candidates[result_idx - 1] = result["candidate"]
            stagewise_plot_records = [
                record
                for record in stagewise_plot_records
                if int(record.get("candidate_idx", -1))
                not in completed_parallel_indices
            ]
            stagewise_plot_records.extend(
                result["stagewise_record"] for result in parallel_stage2_results
            )
            stagewise_plot_records = sorted(
                stagewise_plot_records,
                key=lambda record: int(record.get("candidate_idx", 0)),
            )
            all_stage2_candidates_restored = bool(mask_candidates) and all(
                _candidate_stage2_completed(candidate) for candidate in mask_candidates
            )
            last_idx = (
                max(completed_parallel_indices)
                if completed_parallel_indices
                else last_idx
            )
            _persist_pipeline_checkpoint(
                checkpoint_state,
                checkpoint_path,
                args=args,
                mask_candidates=mask_candidates,
                stage1_completed=stage1_checkpoint_completed,
                stage2_completed=all_stage2_candidates_restored,
                stagewise_plot_records=stagewise_plot_records,
                last_idx=last_idx,
            )
            logging.info(
                "Candidate-parallel stage-2 BO completed for %d candidate(s).",
                len(parallel_stage2_results),
            )
            if not all_stage2_candidates_restored:
                logging.warning(
                    "Candidate-parallel stage-2 BO did not complete every candidate; reloading the main model for sequential fallback."
                )
                model = loadModel(args.model, args.cache_dir)
                model_matches_last_candidate = False
        except Exception as exc:
            logging.exception(
                "Candidate-parallel stage-2 BO failed; falling back to sequential candidate evaluation: %s",
                exc,
            )
            model = loadModel(args.model, args.cache_dir)
            model_matches_last_candidate = False
    for restored_idx, restored_candidate in enumerate(mask_candidates, start=1):
        restored_selection_score = (
            _get_candidate_latency_score(restored_candidate)
            if args.proposed_pareto_selection == "best_latency"
            else _get_candidate_ppl_selection_score(restored_candidate)
        )
        if (
            select_best_final_candidate
            and _candidate_stage2_completed(restored_candidate)
            and restored_selection_score is not None
        ):
            restored_score = float(restored_selection_score)
            if restored_score < best_score:
                best_score = restored_score
                best_idx = restored_idx
                best_attn_mask = restored_candidate.get("attn")
                best_mlp_mask = restored_candidate.get("mlp")
    if all_stage2_candidates_restored:
        restored_last_idx = checkpoint_state.get("selection", {}).get("last_idx")
        if restored_last_idx is None:
            restored_last_idx = checkpoint_state.get("selection", {}).get(
                "selected_candidate_idx"
            )
        if restored_last_idx is None and mask_candidates:
            restored_last_idx = len(mask_candidates)
        if restored_last_idx is not None:
            restored_last_idx = int(restored_last_idx)
            if 1 <= restored_last_idx <= len(mask_candidates):
                last_idx = restored_last_idx
                last_attn_mask = None
                last_mlp_mask = None
                model_matches_last_candidate = False
    for idx, candidate in enumerate(mask_candidates, start=1):
        if all_stage2_candidates_restored:
            logging.info(
                "All candidates already have checkpointed stage-2 results; skipping candidate evaluation and continuing with downstream/post-processing."
            )
            break
        logging.info(describe_candidate(idx, len(mask_candidates), candidate))
        attn_mask = candidate["attn"]
        mlp_mask = candidate["mlp"]

        if _candidate_stage2_completed(candidate):
            logging.info(
                "Skipping candidate %d because stage-2 results were restored from the checkpoint.",
                idx,
            )
            last_idx = idx
            last_attn_mask = None
            last_mlp_mask = None
            model_matches_last_candidate = False
            continue

        if idx > 1:
            if (
                not structural_pruned
                and model_matches_last_candidate
                and last_attn_mask is not None
                and last_mlp_mask is not None
            ):
                unmaskModel(model, last_attn_mask, last_mlp_mask)
            if second_stage_enabled:
                logging.info(
                    "Reloading the base model before evaluating candidate %d so stage-2 pruning does not accumulate across candidates.",
                    idx,
                )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = loadModel(args.model, args.cache_dir)

        if not structural_pruned:
            maskModel(model, attn_mask, mlp_mask)
        model_matches_last_candidate = True

        if structural_pruned:
            total_params, active_params = get_parameter_counts(model)
        else:
            total_params, active_params = count_params_with_masks(
                model, attn_mask, mlp_mask
            )
        pruned_params = total_params - active_params
        stage1_pruned_params = float(pruned_params)
        stage2_pruned_params = 0.0
        logging.info(
            "Parameters — total: %d | active: %d | pruned: %.2f%%",
            total_params,
            active_params,
            (pruned_params / total_params * 100) if total_params else 0.0,
        )
        before_total_ss, before_nonzero_ss = get_parameter_counts(model)
        logging.info(
            "Second-stage (candidate %d)debug_random_targets parameters before head search — total: %d, nonzero: %d",
            idx,
            before_total_ss,
            before_nonzero_ss,
        )
        if measure_stage1_plot_ppl and bo_eval_datasets is not None:
            logging.info(
                "Candidate %d: evaluating pre-stage-2 calibration prompt perplexity across %d dataset(s).",
                idx,
                len(bo_eval_datasets),
            )
            stage1_plot_ppl, stage1_plot_breakdown = _evaluate_perplexity_suite_local(
                model,
                bo_eval_datasets,
                seq_len=args.seq_len,
                device=str(_resolve_model_compute_device(model)),
            )
            candidate["stage1_ppl"] = float(stage1_plot_ppl)
            candidate["stage1_ppl_label"] = "Stage-1 Calibration-Mix PPL"
            candidate["stage1_ppl_breakdown"] = stage1_plot_breakdown
            logging.info(
                "Perplexity (calibration prompt files, stage-1) = %.4f [%s]",
                stage1_plot_ppl,
                ", ".join(
                    f"{name}={value:.4f}"
                    for name, value in stage1_plot_breakdown.items()
                ),
            )
        elif args.evaluate_perplexity_stage1 and wikitext_input_ids is not None:
            ppl_stage1 = evaluate_perplexity(
                model, wikitext_input_ids, seq_len=args.seq_len
            )
            candidate["stage1_ppl"] = float(ppl_stage1)
            candidate["stage1_ppl_label"] = "Stage-1 Wikitext2 PPL"
            logging.info("Perplexity (Wikitext2, stage-1) = %.4f", ppl_stage1)
        elif (
            args.proposed_first_objective in {"perplexity", "ppl_mix"}
            and candidate.get("fitness") is not None
        ):
            candidate["stage1_ppl"] = float(candidate["fitness"])
            candidate["stage1_ppl_label"] = (
                "Stage-1 BO Calibration PPL"
                if args.proposed_first_objective == "ppl_mix"
                else "Stage-1 Wikitext2 PPL"
            )
        elif candidate.get("fitness") is not None:
            candidate["stage1_ppl"] = float(candidate["fitness"])
            candidate["stage1_ppl_label"] = "Stage-1 Fitness"
        # Stage-2 pruning runs only when at least one component is enabled.
        candidate_bo_ppl = None
        candidate_bo_objective = str(args.second_stage_bayes_objective)
        stage2_layer_stats = []
        if second_stage_enabled:
            pruned_ratio_stage1 = (
                (pruned_params / total_params) if total_params else 0.0
            )
            # Remaining budget is based on how much was already pruned in stage-1.
            stage1_consumed = pruned_ratio_stage1
            remaining_budget = max(0.0, args.sparsity - stage1_consumed)
            # Convert remaining budget into a keep ratio for the stage-2 search.
            derived_second_stage_keep_ratio = max(0.0, min(1.0, 1.0 - remaining_budget))
            if remaining_budget <= 0.0:
                logging.info(
                    "Stage-2 pruning skipped for candidate %d: stage-1 pruned %.4f >= target %.4f.",
                    idx,
                    pruned_ratio_stage1,
                    args.sparsity,
                )
            else:
                logging.info(
                    "Stage-2 remaining budget %.4f (keep ratio %.4f) after stage-1 pruned %.4f.",
                    remaining_budget,
                    derived_second_stage_keep_ratio,
                    stage1_consumed,
                )
            prompt_source = _get_second_stage_prompt_source(args)
            if remaining_budget > 0.0:
                logging.info(
                    "Running second-stage pruning for mask candidate %d (heads=%s, mlp=%s, embedding=%s, norm=%s).",
                    idx,
                    args.second_stage_heads,
                    args.second_stage_mlp,
                    args.second_stage_embedding,
                    args.second_stage_norm,
                )
                layer_keep_ratios_for_stage2 = second_stage_layer_keep_ratios
                layer_component_masks_for_stage2 = None
                best_importance_metric = args.second_stage_importance_metric
                stage2_allocation_params = None
                if second_stage_allocation_method == "proposed":
                    logging.info(
                        "Bayesian optimization enabled for stage-2: trials=%d warmup=%d "
                        "objective=%s target_prune_ratio=%.4f min_keep=%.4f early_layers=%d "
                        "early_min_keep=%s early_max_prune_factor=%.4f.",
                        args.second_stage_bayes_trials,
                        args.second_stage_bayes_warmup,
                        args.second_stage_bayes_objective,
                        remaining_budget,
                        args.second_stage_bayes_min_keep_ratio,
                        args.second_stage_bayes_early_layer_count,
                        (
                            f"{args.second_stage_bayes_early_min_keep_ratio:.4f}"
                            if args.second_stage_bayes_early_min_keep_ratio is not None
                            else "auto"
                        ),
                        args.second_stage_bayes_early_max_prune_factor,
                    )
                    (
                        layer_keep_ratios_for_stage2,
                        bo_best_ppl,
                        bo_best_params,
                    ) = _bayes_optimize_layer_keep_ratios(
                        model,
                        num_layers=num_blocks,
                        target_prune_ratio=remaining_budget,
                        trials=args.second_stage_bayes_trials,
                        warmup=args.second_stage_bayes_warmup,
                        seed=args.seed,
                        max_ratio=derived_second_stage_keep_ratio,
                        prune_heads=args.second_stage_heads,
                        prune_mlp=args.second_stage_mlp,
                        prune_embedding=args.second_stage_embedding,
                        prune_norm=args.second_stage_norm,
                        dataset_name=prompt_source,
                        max_prompts=args.second_stage_num_prompts,
                        dataset_max_length=args.second_stage_prompt_max_length,
                        bo_eval_datasets=bo_eval_datasets,
                        seq_len=args.seq_len,
                        bo_objective=args.second_stage_bayes_objective,
                        importance_metric=args.second_stage_importance_metric,
                        scoring_method=args.second_stage_method,
                        flap_metric=args.second_stage_flap_metric,
                        min_keep_ratio=args.second_stage_bayes_min_keep_ratio,
                        early_layer_min_keep_ratio=args.second_stage_bayes_early_min_keep_ratio,
                        early_layer_count=args.second_stage_bayes_early_layer_count,
                        early_layer_max_prune_factor=args.second_stage_bayes_early_max_prune_factor,
                        tokenizer=tokenizer,
                        latency_batch_size=args.second_stage_bayes_latency_batch_size,
                        latency_seq_len=args.second_stage_bayes_latency_seq_len,
                        latency_num_runs=args.second_stage_bayes_latency_runs,
                        latency_num_warmup_runs=args.second_stage_bayes_latency_warmup,
                        latency_input_prompt=args.second_stage_bayes_latency_prompt,
                        parallel_gpus=args.second_stage_bayes_parallel_gpus,
                        model_name=args.model,
                        cache_dir=args.cache_dir,
                        stage1_attn_mask=attn_mask,
                        stage1_mlp_mask=mlp_mask,
                        logging_level=args.logging,
                    )
                    logging.info(
                        "Bayes-opt best params k=%.4f c=%.4f with bo_score=%.4f.",
                        bo_best_params[0],
                        bo_best_params[1],
                        bo_best_ppl,
                    )
                    candidate_bo_ppl = bo_best_ppl
                    stage2_allocation_params = {
                        "bo_k": float(bo_best_params[0]),
                        "bo_c": float(bo_best_params[1]),
                    }
                else:
                    if second_stage_allocation_method != "manual":
                        layer_keep_ratios_for_stage2, stage2_allocation_params = (
                            _build_stage2_keep_ratios_for_method(
                                model,
                                allocation_method=second_stage_allocation_method,
                                target_keep_ratio=derived_second_stage_keep_ratio,
                                target_prune_ratio=remaining_budget,
                                dataset_name=prompt_source,
                                dataset_split="train",
                                max_prompts=args.second_stage_num_prompts,
                                dataset_max_length=args.second_stage_prompt_max_length,
                                importance_metric=args.second_stage_importance_metric,
                                scoring_method=args.second_stage_method,
                                min_keep_ratio=args.second_stage_bayes_min_keep_ratio,
                                early_layer_min_keep_ratio=args.second_stage_bayes_early_min_keep_ratio,
                                early_layer_count=args.second_stage_bayes_early_layer_count,
                                early_layer_max_prune_factor=args.second_stage_bayes_early_max_prune_factor,
                                beta_value=_resolve_stage2_beta_value(args),
                            )
                        )
                        logging.info(
                            "Stage-2 allocation method '%s' produced per-layer keep ratios.",
                            second_stage_allocation_method,
                        )
                    else:
                        logging.info(
                            "Stage-2 allocation method 'manual' will use provided per-layer keep ratios."
                        )
                if layer_keep_ratios_for_stage2 is not None:
                    candidate["layerwise_allocation_plot"] = (
                        save_layerwise_allocation_plot(
                            layer_keep_ratios_for_stage2,
                            candidate_idx=idx,
                            model_name=args.model,
                            graphs_dir="graphs",
                        )
                    )
                second_stage_result = second_stage_optimize_heads(
                    model,
                    max_ratio=derived_second_stage_keep_ratio,
                    layer_keep_ratios=layer_keep_ratios_for_stage2,
                    layer_component_masks=layer_component_masks_for_stage2,
                    prune_heads=args.second_stage_heads,
                    prune_mlp=args.second_stage_mlp,
                    prune_embedding=args.second_stage_embedding,
                    prune_norm=args.second_stage_norm,
                    dataset_name=prompt_source,
                    max_prompts=args.second_stage_num_prompts,
                    dataset_max_length=args.second_stage_prompt_max_length,
                    importance_metric=best_importance_metric,
                    scoring_method=args.second_stage_method,
                    seed=args.seed,
                    flap_metric=args.second_stage_flap_metric,
                )
                stage2_layer_stats = copy.deepcopy(
                    second_stage_result.get("layer_stats", [])
                )
            after_total, after_nonzero = get_parameter_counts(model)
            pruned_total = before_total_ss - after_total
            pruned_nonzero = before_nonzero_ss - after_nonzero
            stage2_pruned_params = float(max(0, pruned_nonzero))
            logging.info(
                "Second-stage (candidate %d) total parameters — before: %d | after: %d | removed: %.2f%% (%d / %d)",
                idx,
                before_total_ss,
                after_total,
                (pruned_total / before_total_ss * 100) if before_total_ss else 0.0,
                pruned_total,
                before_total_ss,
            )
            logging.info(
                "Second-stage (candidate %d) nonzero parameters — before: %d | after: %d | removed: %.2f%% (%d / %d)",
                idx,
                before_nonzero_ss,
                after_nonzero,
                (
                    (pruned_nonzero / before_nonzero_ss * 100)
                    if before_nonzero_ss
                    else 0.0
                ),
                pruned_nonzero,
                before_nonzero_ss,
            )
        else:
            logging.info(
                "Skipping second-stage pruning for candidate %d (all components disabled).",
                idx,
            )
        candidate["candidate_idx"] = idx
        candidate["stage1_pruned_params"] = stage1_pruned_params
        candidate["stage1_pruned_ratio"] = (
            stage1_pruned_params / float(total_params) if total_params else 0.0
        )
        candidate["stage2_pruned_params"] = stage2_pruned_params
        candidate["stage2_pruned_ratio"] = (
            stage2_pruned_params / float(total_params) if total_params else 0.0
        )
        candidate["stage2_keep_ratio"] = None
        candidate["stage2_layer_keep_ratios"] = None
        candidate["stage2_layer_prune_ratios"] = None
        candidate["stage2_layer_stats"] = copy.deepcopy(stage2_layer_stats)
        candidate["stage2_importance_metric"] = None
        candidate["stage2_allocation_method"] = None
        candidate["stage2_allocation_params"] = None
        candidate["final_plot_ppl"] = None
        candidate["final_plot_ppl_label"] = None
        candidate["final_active_params"] = float(
            after_nonzero if second_stage_enabled else active_params
        )
        if candidate_bo_ppl is not None:
            candidate["bo_objective"] = candidate_bo_objective
            candidate["bo_objective_score"] = float(candidate_bo_ppl)
            candidate["bo_calibration_score"] = float(candidate_bo_ppl)
            if candidate_bo_objective == "latency":
                candidate["bo_latency_score"] = float(candidate_bo_ppl)
            else:
                candidate["bo_calibration_ppl"] = float(candidate_bo_ppl)
            if (
                candidate_bo_objective != "latency"
                and args.proposed_first_objective == "ppl_mix"
            ):
                candidate["final_plot_ppl"] = float(candidate_bo_ppl)
                candidate["final_plot_ppl_label"] = "Post-Stage-2 BO Calibration Score"
        if second_stage_enabled and remaining_budget > 0.0:
            candidate["stage2_keep_ratio"] = float(derived_second_stage_keep_ratio)
            candidate["stage2_layer_keep_ratios"] = copy.deepcopy(
                layer_keep_ratios_for_stage2
            )
            candidate["stage2_layer_prune_ratios"] = (
                _layer_prune_ratios_from_keep_ratios(layer_keep_ratios_for_stage2)
                if layer_keep_ratios_for_stage2 is not None
                else None
            )
            candidate["stage2_importance_metric"] = best_importance_metric
            candidate["stage2_allocation_method"] = second_stage_allocation_method
            candidate["stage2_allocation_params"] = copy.deepcopy(
                stage2_allocation_params
            )
        candidate["layer_component_pruning_plot"] = save_layer_component_pruning_plot(
            candidate_idx=idx,
            model_name=args.model,
            stage1_attn_mask=attn_mask,
            stage1_mlp_mask=mlp_mask,
            stage2_layer_stats=stage2_layer_stats,
            graphs_dir="graphs",
        )
        stagewise_plot_records.append(
            {
                "candidate_idx": float(idx),
                "total_params": float(total_params),
                "stage1_pruned_params": stage1_pruned_params,
                "stage2_pruned_params": stage2_pruned_params,
            }
        )
        if args.method == "proposed" and len(mask_candidates) > 1:
            if (
                select_best_final_candidate
                and args.proposed_pareto_selection == "best_ppl"
                and candidate_bo_ppl is not None
                and candidate_bo_objective != "latency"
                and candidate_bo_ppl < best_score
            ):
                best_score = candidate_bo_ppl
                best_idx = idx
                best_attn_mask = attn_mask
                best_mlp_mask = mlp_mask

            if evaluate_pareto_candidate_ppl and bo_eval_datasets is not None:
                ppl_candidate, ppl_breakdown = _evaluate_perplexity_suite_local(
                    model,
                    bo_eval_datasets,
                    seq_len=args.seq_len,
                    device=str(_resolve_model_compute_device(model)),
                )
                candidate["calibration_mix_ppl"] = ppl_candidate
                candidate["calibration_mix_ppl_breakdown"] = ppl_breakdown
                candidate["final_plot_ppl"] = float(ppl_candidate)
                candidate["final_plot_ppl_label"] = "Post-Stage-2 Calibration-Mix PPL"
                logging.info(
                    "Perplexity (candidate %d, calibration prompt files) = %.4f [%s]",
                    idx,
                    ppl_candidate,
                    ", ".join(
                        f"{name}={value:.4f}" for name, value in ppl_breakdown.items()
                    ),
                )
                if (
                    select_best_final_candidate
                    and args.proposed_pareto_selection == "best_ppl"
                    and float(ppl_candidate) < best_score
                ):
                    best_score = float(ppl_candidate)
                    best_idx = idx
                    best_attn_mask = attn_mask
                    best_mlp_mask = mlp_mask

        if (
            candidate.get("stage1_ppl") is not None
            and candidate.get("final_plot_ppl") is None
        ):
            candidate["final_plot_ppl"] = float(candidate["stage1_ppl"])
            candidate["final_plot_ppl_label"] = "Fallback: Stage-1 metric reused"

        if measure_final_candidate_latency:
            latency_device = str(_resolve_model_compute_device(model))
            if "cuda" not in latency_device:
                candidate["final_latency"] = None
                if not warned_final_latency_skipped:
                    logging.warning(
                        "Skipping final per-candidate latency measurement: CUDA device required (found %s).",
                        latency_device,
                    )
                    warned_final_latency_skipped = True
            else:
                try:
                    latency_metrics, _, _ = measure_latency(
                        model=model,
                        tokenizer=tokenizer,
                        device=latency_device,
                        seed=args.seed,
                    )
                    candidate["final_latency"] = float(
                        latency_metrics["time_mean(sec)"]
                    )
                    logging.info(
                        "Final latency (candidate %d) = %.4f sec",
                        idx,
                        candidate["final_latency"],
                    )
                except Exception as exc:
                    candidate["final_latency"] = None
                    logging.exception(
                        "Final latency measurement failed for candidate %d: %s",
                        idx,
                        exc,
                    )
        if (
            args.method == "proposed"
            and len(mask_candidates) > 1
            and select_best_final_candidate
            and args.proposed_pareto_selection == "best_latency"
        ):
            latency_score = _get_candidate_latency_score(candidate)
            if latency_score is not None and latency_score < best_score:
                best_score = float(latency_score)
                best_idx = idx
                best_attn_mask = attn_mask
                best_mlp_mask = mlp_mask
        last_idx = idx
        last_attn_mask = attn_mask
        last_mlp_mask = mlp_mask
        candidate["_stage2_completed"] = True
        _persist_pipeline_checkpoint(
            checkpoint_state,
            checkpoint_path,
            args=args,
            mask_candidates=mask_candidates,
            stage1_completed=stage1_checkpoint_completed,
            stage2_completed=False,
            stagewise_plot_records=stagewise_plot_records,
            best_idx=best_idx,
            last_idx=last_idx,
        )
        logging.info(
            "Saved stage-2 progress after candidate %d to %s.",
            idx,
            checkpoint_path.resolve(),
        )

    stagewise_plot_path = save_stagewise_pruning_plot(
        stagewise_plot_records,
        model_name=args.model,
        graphs_dir="graphs",
    )
    if stagewise_plot_path is not None:
        logging.info("Stagewise pruning summary plot saved at %s", stagewise_plot_path)
    stage1_history_points = []
    if args.method == "proposed":
        stage1_history_path = Path(str(args.proposed_save_pareto)).with_name(
            f"{Path(str(args.proposed_save_pareto)).stem}_all.csv"
        )
        stage1_history_points = load_proposed_history_points(str(stage1_history_path))
    pareto_refinement_plot_path = save_stage2_pareto_refinement_plot(
        mask_candidates,
        model_name=args.model,
        stage1_all_solutions=stage1_history_points,
        graphs_dir="graphs",
    )
    if pareto_refinement_plot_path is not None:
        logging.info(
            "Stage-2 Pareto refinement plot saved at %s", pareto_refinement_plot_path
        )
    _persist_pipeline_checkpoint(
        checkpoint_state,
        checkpoint_path,
        args=args,
        mask_candidates=mask_candidates,
        stage1_completed=stage1_checkpoint_completed,
        stage2_completed=True,
        stagewise_plot_records=stagewise_plot_records,
        best_idx=best_idx,
        last_idx=last_idx,
    )

    if select_best_final_candidate and best_idx is None:
        for fallback_idx, fallback_candidate in enumerate(mask_candidates, start=1):
            fallback_score = (
                _get_candidate_latency_score(fallback_candidate)
                if args.proposed_pareto_selection == "best_latency"
                else _get_candidate_ppl_selection_score(fallback_candidate)
            )
            if fallback_score is None or fallback_score >= best_score:
                continue
            best_score = float(fallback_score)
            best_idx = fallback_idx
            best_attn_mask = fallback_candidate.get("attn")
            best_mlp_mask = fallback_candidate.get("mlp")
        if best_idx is not None:
            logging.info(
                "Selected candidate %d from saved final candidate metrics because no stage-2 BO score was recorded.",
                best_idx,
            )

    if select_best_final_candidate and best_idx:
        if (
            best_idx != last_idx or not model_matches_last_candidate
        ) and not structural_pruned:
            best_candidate = mask_candidates[best_idx - 1]
            if second_stage_enabled or not model_matches_last_candidate:
                logging.info(
                    "Replaying the selected best candidate %d on a fresh base model so downstream steps use the correct stage-2-pruned model.",
                    best_idx,
                )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = _replay_candidate_on_fresh_model(best_candidate, args)
                model_matches_last_candidate = True
            else:
                unmaskModel(model, last_attn_mask, last_mlp_mask)
                maskModel(model, best_attn_mask, best_mlp_mask)
                model_matches_last_candidate = True
        last_idx = best_idx
        last_attn_mask = best_attn_mask
        last_mlp_mask = best_mlp_mask
        logging.info(
            "Selected best Pareto candidate %d based on %s=%.6f.",
            best_idx,
            selection_metric,
            best_score,
        )
        pending_selected_candidate_downstream_eval = True
        _persist_pipeline_checkpoint(
            checkpoint_state,
            checkpoint_path,
            args=args,
            mask_candidates=mask_candidates,
            stage1_completed=stage1_checkpoint_completed,
            stage2_completed=True,
            stagewise_plot_records=stagewise_plot_records,
            selected_candidate_idx=best_idx,
            best_idx=best_idx,
            last_idx=last_idx,
        )
    elif select_best_final_candidate:
        raise RuntimeError(
            f"No {selection_metric} scores were recorded for the Pareto candidates. "
            "Run with stage-2 BO enabled or switch --proposed_pareto_selection=sorted."
        )

    if args.save_selected_stage2_model:
        export_candidate_idx = _resolve_selected_stage2_export_candidate_idx(
            mask_candidates,
            checkpoint_state,
            best_idx=best_idx,
            last_idx=last_idx,
        )
        if export_candidate_idx is not None:
            _save_selected_stage2_model(
                mask_candidates[int(export_candidate_idx) - 1],
                args,
                model_name=args.model,
                candidate_idx=int(export_candidate_idx),
            )
        elif mask_candidates:
            logging.info(
                "Skipping selected stage-2 model export because no unique final candidate is selected. "
                "Use --proposed_pareto_selection best_ppl (or run with a single final candidate) to export one model."
            )

    if report_sorted_final_candidates:
        _log_final_pareto_summary(mask_candidates)
        _evaluate_all_final_candidates_downstream(
            sorted(mask_candidates, key=_candidate_latency_sort_key),
            tokenizer,
            args.model,
            args,
        )
        performed_mode_downstream_eval = True
        skip_single_candidate_postprocessing = True
        logging.info(
            "Skipping single-candidate post-processing because --proposed_pareto_selection=sorted keeps the full final candidate list."
        )
    elif report_all_final_candidates:
        _log_all_final_candidates(mask_candidates)
        _evaluate_all_final_candidates_downstream(
            mask_candidates,
            tokenizer,
            args.model,
            args,
        )
        performed_mode_downstream_eval = True
        skip_single_candidate_postprocessing = True
        logging.info(
            "Skipping single-candidate post-processing because --proposed_pareto_selection=all evaluates all final candidates."
        )

    if not skip_single_candidate_postprocessing and last_idx is not None:
        if not checkpoint_state.get("selection", {}).get("completed"):
            _persist_pipeline_checkpoint(
                checkpoint_state,
                checkpoint_path,
                args=args,
                mask_candidates=mask_candidates,
                stage1_completed=stage1_checkpoint_completed,
                stage2_completed=True,
                stagewise_plot_records=stagewise_plot_records,
                selected_candidate_idx=last_idx,
                best_idx=best_idx,
                last_idx=last_idx,
            )
        if not model_matches_last_candidate:
            logging.info(
                "Replaying candidate %d from the checkpoint on a fresh base model for post-processing.",
                last_idx,
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = _replay_candidate_on_fresh_model(
                mask_candidates[last_idx - 1], args
            )
            model_matches_last_candidate = True

    if not skip_single_candidate_postprocessing and args.recovery_finetune:
        candidate_dir = Path(args.recovery_output_dir) / f"candidate_{last_idx}"
        pruned_dir = candidate_dir / "pruned_model"
        tuned_dir = candidate_dir / "tuned_model"
        pruned_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Saving pruned model for candidate %d to %s", last_idx, pruned_dir)
        model.save_pretrained(pruned_dir)
        tokenizer.save_pretrained(pruned_dir)

        target_modules = (
            [
                module.strip()
                for module in args.recovery_target_modules.split(",")
                if module.strip()
            ]
            if args.recovery_target_modules
            else None
        )
        logging.info(
            "Starting recovery fine-tuning (candidate %d) via fine_tune.py [epochs=%s, save_steps=%s, logging_steps=%s, num_samples=%s].",
            last_idx,
            args.recovery_epochs,
            args.recovery_save_steps,
            args.recovery_logging_steps,
            args.recovery_num_samples,
        )
        with torch.enable_grad():
            if args.recovery_bayes:
                if bo_eval_datasets is None:
                    bo_eval_datasets = prepare_bo_prompt_eval_inputs(
                        tokenizer,
                        args.bo_eval_prompt_files,
                    )
                bo_root = candidate_dir / "recovery_bo"
                tuned_model, tuned_dir, bo_payload = _bayes_optimize_recovery_lora(
                    pruned_model_dir=pruned_dir,
                    tuned_root_dir=bo_root,
                    arch=args.recovery_arch,
                    num_train_epochs=args.recovery_epochs,
                    save_steps=args.recovery_save_steps,
                    logging_steps=args.recovery_logging_steps,
                    num_samples=args.recovery_num_samples,
                    target_modules=target_modules,
                    report_to=args.recovery_report_to,
                    bo_eval_datasets=bo_eval_datasets,
                    seq_len=args.seq_len,
                    seed=args.seed,
                    trials=args.recovery_bayes_trials,
                    warmup=args.recovery_bayes_warmup,
                )
                logging.info(
                    "Recovery BO selected r=%d alpha=%d dropout=%.3f lr=%.6f warmup=%.3f "
                    "with bo_score=%.4f (%s).",
                    bo_payload["r"],
                    bo_payload["alpha"],
                    bo_payload["dropout"],
                    bo_payload["learning_rate"],
                    bo_payload["warmup_ratio"],
                    bo_payload["ppl"],
                    ", ".join(
                        f"{name}={value:.4f}"
                        for name, value in bo_payload["ppl_breakdown"].items()
                    ),
                )
            else:
                recovery_gpu_ids = _resolve_recovery_finetune_gpu_ids(args)
                logging.info(
                    "Resolved recovery fine-tuning GPUs from --recovery_finetune_gpus=%r, "
                    "--downstream_parallel_gpus=%r, CUDA_VISIBLE_DEVICES=%r: %s",
                    args.recovery_finetune_gpus,
                    args.downstream_parallel_gpus,
                    os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    recovery_gpu_ids,
                )
                if len(recovery_gpu_ids) > 1:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _run_recovery_finetune_ddp(
                        pruned_model_dir=pruned_dir,
                        tuned_model_dir=tuned_dir,
                        candidate=mask_candidates[int(last_idx) - 1],
                        args=args,
                        target_modules=target_modules,
                        gpu_ids=recovery_gpu_ids,
                    )
                    tuned_model = _load_replayed_recovery_model(
                        tuned_dir,
                        args.model,
                        args.cache_dir,
                    )
                else:
                    tuned_model = fine_tune_model(
                        pruned_model_dir=str(pruned_dir),
                        tuned_model_dir=str(tuned_dir),
                        arch=args.recovery_arch,
                        num_train_epochs=args.recovery_epochs,
                        save_steps=args.recovery_save_steps,
                        logging_steps=args.recovery_logging_steps,
                        num_samples=args.recovery_num_samples,
                        target_modules=target_modules,
                        warmup_ratio=args.recovery_warmup_ratio,
                        report_to=args.recovery_report_to,
                        model=model,
                        tokenizer=tokenizer,
                    )
            if tuned_model is not None:
                last_tuned_model = tuned_model
                last_tuned_dir = tuned_dir

    if not skip_single_candidate_postprocessing and last_tuned_model is not None:
        model = last_tuned_model
        tokenizer = load_tokenizer(str(last_tuned_dir))
        if args.evaluate_perplexity or args.evaluate_perplexity_all:
            wikitext_input_ids = prepare_wikitext_inputs(tokenizer, args.local_datasets)
        if args.evaluate_perplexity_all:
            c4_val_input_ids = prepare_c4_validation_inputs(
                tokenizer, args.local_datasets
            )
            fineweb_edu_input_ids = prepare_fineweb_edu_inputs(
                tokenizer, args.local_datasets
            )
            gsm8k_input_ids = prepare_gsm8k_inputs(tokenizer, args.local_datasets)
        logging.info(
            "Recovery fine-tuning complete for candidate %d; tuned model stored at %s",
            last_idx,
            last_tuned_dir,
        )
    if not skip_single_candidate_postprocessing and args.evaluate_perplexity_all:
        final_ppl_datasets = [
            ("wikitext2", wikitext_input_ids),
            ("c4", c4_val_input_ids),
            ("fineweb_edu", fineweb_edu_input_ids),
            ("gsm8k", gsm8k_input_ids),
        ]
        can_replay_for_parallel_ppl = (
            bool(args.perplexity_parallel_gpus)
            and last_idx is not None
            and not args.recovery_finetune
        )
        can_parallel_tuned_model_ppl = (
            bool(args.perplexity_parallel_gpus) and last_tuned_dir is not None
        )
        if can_parallel_tuned_model_ppl:
            try:
                final_ppl_results = (
                    _evaluate_model_dir_perplexity_parallel(
                        Path(last_tuned_dir),
                        final_ppl_datasets,
                        args,
                    )
                    or {}
                )
            except Exception as exc:
                logging.exception(
                    "Dataset-parallel perplexity evaluation for tuned model failed; "
                    "falling back to single-process evaluation: %s",
                    exc,
                )
                final_ppl_results = {}
        elif can_replay_for_parallel_ppl:
            try:
                final_ppl_results = (
                    _evaluate_candidate_perplexity_parallel(
                        mask_candidates[int(last_idx) - 1],
                        args.model,
                        final_ppl_datasets,
                        args,
                    )
                    or {}
                )
            except Exception as exc:
                logging.exception(
                    "Dataset-parallel perplexity evaluation failed for candidate %s; "
                    "falling back to single-process evaluation: %s",
                    last_idx,
                    exc,
                )
                final_ppl_results = {}
        elif args.perplexity_parallel_gpus:
            logging.warning(
                "Ignoring --perplexity_parallel_gpus for in-memory perplexity evaluation; "
                "parallel replay is unavailable for this candidate."
            )
        if not final_ppl_results:
            final_ppl_results = (
                evaluation_ppl(
                    model=model,
                    wikitext_input_ids=wikitext_input_ids,
                    c4_val_input_ids=c4_val_input_ids,
                    fineweb_edu_input_ids=fineweb_edu_input_ids,
                    gsm8k_input_ids=gsm8k_input_ids,
                    seq_len=args.seq_len,
                    batch_size=args.perplexity_batch_size,
                )
                or {}
            )
        else:
            for dataset_name, ppl_value in final_ppl_results.items():
                logging.info("Perplexity (%s) = %.4f", dataset_name, ppl_value)
        if last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
            mask_candidates[int(last_idx) - 1]["final_ppl_results"] = dict(
                final_ppl_results
            )
    elif (
        not skip_single_candidate_postprocessing
        and args.evaluate_perplexity
        and wikitext_input_ids is not None
    ):
        ppl = evaluate_perplexity(
            model,
            wikitext_input_ids,
            seq_len=args.seq_len,
            batch_size=args.perplexity_batch_size,
        )
        final_ppl_results = {"wikitext2": float(ppl)}
        if last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
            mask_candidates[int(last_idx) - 1]["final_ppl_results"] = dict(
                final_ppl_results
            )
        logging.info("Perplexity (Wikitext2) = %.4f", ppl)
    if not skip_single_candidate_postprocessing and args.evaluate_inference:
        device_str = str(_resolve_model_compute_device(model))
        if "cuda" not in device_str:
            logging.warning(
                "Skipping inference-time measurement for candidate %d: CUDA device required (found %s).",
                last_idx,
                device_str,
            )
        else:
            try:
                evaluate_inference_time(
                    model,
                    first_calibration_sample,
                    calibration_dataset=calibration_dataset,
                    batch_size=1,
                )
                evaluate_inference_time(
                    model,
                    first_calibration_sample,
                    calibration_dataset=calibration_dataset,
                    batch_size=8,
                )
                evaluate_inference_time(
                    model,
                    first_calibration_sample,
                    calibration_dataset=calibration_dataset,
                    batch_size=16,
                )
                evaluate_inference_time(
                    model,
                    first_calibration_sample,
                    calibration_dataset=calibration_dataset,
                    batch_size=32,
                )
                evaluate_inference_time(
                    model,
                    first_calibration_sample,
                    calibration_dataset=calibration_dataset,
                    batch_size=64,
                )
                evaluate_inference_time(
                    model,
                    first_calibration_sample,
                    calibration_dataset=calibration_dataset,
                    batch_size=128,
                )
                evaluate_inference_time(
                    model,
                    first_calibration_sample,
                    calibration_dataset=calibration_dataset,
                    batch_size=256,
                )
            except Exception as exc:  # keep run alive if measurement fails
                logging.exception(
                    "Inference-time measurement failed for candidate %d: %s",
                    last_idx,
                    exc,
                )
    if (
        not skip_single_candidate_postprocessing
        and args.evaluate_downstream
        and not performed_mode_downstream_eval
    ):
        can_replay_for_parallel_downstream = (
            bool(args.downstream_parallel_gpus)
            and last_idx is not None
            and not args.recovery_finetune
        )
        can_parallel_tuned_model_downstream = (
            bool(args.downstream_parallel_gpus) and last_tuned_dir is not None
        )
        if can_parallel_tuned_model_downstream:
            parallel_downstream = _evaluate_model_dir_downstream_parallel(
                Path(last_tuned_dir),
                _parse_downstream_tasks(args.downstream_tasks),
                args,
            )
            if parallel_downstream is not None:
                (
                    standalone_downstream_avg_acc,
                    standalone_downstream_task_accs,
                    downstream_results,
                ) = parallel_downstream
                if last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
                    selected_candidate = mask_candidates[int(last_idx) - 1]
                    selected_candidate["downstream_avg_acc"] = (
                        standalone_downstream_avg_acc
                    )
                    selected_candidate["downstream_task_accs"] = (
                        standalone_downstream_task_accs
                    )
                    selected_candidate["downstream_eval_results"] = downstream_results
                performed_mode_downstream_eval = True
            else:
                downstream_results = evaluation_downstream(
                    model,
                    args.model,
                    task_list=_parse_downstream_tasks(args.downstream_tasks),
                    batch_size=args.downstream_batch_size,
                )
                standalone_downstream_avg_acc, standalone_downstream_task_accs = (
                    _summarize_downstream_results(
                        downstream_results,
                        _parse_downstream_tasks(args.downstream_tasks),
                    )
                )
                if last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
                    selected_candidate = mask_candidates[int(last_idx) - 1]
                    selected_candidate["downstream_avg_acc"] = (
                        standalone_downstream_avg_acc
                    )
                    selected_candidate["downstream_task_accs"] = (
                        standalone_downstream_task_accs
                    )
                    selected_candidate["downstream_eval_results"] = downstream_results
        elif can_replay_for_parallel_downstream:
            _evaluate_all_final_candidates_downstream(
                [mask_candidates[int(last_idx) - 1]],
                tokenizer,
                args.model,
                args,
            )
            performed_mode_downstream_eval = True
        else:
            if args.downstream_parallel_gpus:
                logging.warning(
                    "Ignoring --downstream_parallel_gpus for in-memory downstream evaluation; "
                    "task-parallel replay is unavailable for this candidate."
                )
            downstream_results = evaluation_downstream(
                model,
                args.model,
                task_list=_parse_downstream_tasks(args.downstream_tasks),
                batch_size=args.downstream_batch_size,
            )
            standalone_downstream_avg_acc, standalone_downstream_task_accs = (
                _summarize_downstream_results(
                    downstream_results,
                    _parse_downstream_tasks(args.downstream_tasks),
                )
            )
            if last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
                selected_candidate = mask_candidates[int(last_idx) - 1]
                selected_candidate["downstream_avg_acc"] = standalone_downstream_avg_acc
                selected_candidate["downstream_task_accs"] = (
                    standalone_downstream_task_accs
                )
                selected_candidate["downstream_eval_results"] = downstream_results
    elif (
        not skip_single_candidate_postprocessing
        and pending_selected_candidate_downstream_eval
        and not performed_mode_downstream_eval
    ):
        _evaluate_all_final_candidates_downstream(
            [mask_candidates[best_idx - 1]],
            tokenizer,
            args.model,
            args,
        )
        performed_mode_downstream_eval = True
    elif (
        not skip_single_candidate_postprocessing
        and auto_resume_single_candidate_downstream_eval
        and not performed_mode_downstream_eval
    ):
        selected_candidate_idx = checkpoint_state.get("selection", {}).get(
            "selected_candidate_idx"
        )
        if selected_candidate_idx is None:
            selected_candidate_idx = last_idx or 1
        _evaluate_all_final_candidates_downstream(
            [mask_candidates[int(selected_candidate_idx) - 1]],
            tokenizer,
            args.model,
            args,
        )
        performed_mode_downstream_eval = True

    if not skip_single_candidate_postprocessing:
        latency_device = str(_resolve_model_compute_device(model))
        if "cuda" not in latency_device:
            logging.warning(
                "Skipping latency measurement for candidate %d: CUDA device required (found %s).",
                last_idx,
                latency_device,
            )
        else:
            latency_metrics, _, _ = measure_latency(
                model=model,
                tokenizer=tokenizer,
                device=latency_device,
                seed=args.seed,
            )
            if last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
                selected_candidate = mask_candidates[int(last_idx) - 1]
                selected_candidate.setdefault(
                    "final_latency", latency_metrics.get("time_mean(sec)")
                )
                selected_candidate["postprocess_latency_time_mean_sec"] = (
                    latency_metrics.get("time_mean(sec)")
                )
                selected_candidate["postprocess_latency_th_mean_tokens_s"] = (
                    latency_metrics.get("th_mean(tokens/s)")
                )
            logging.info(
                (
                    "Latency (candidate %d) — time_mean=%.4f sec, time_std=%.4f, "
                    "th_mean=%.2f tok/s, th_std=%.2f, mem=%.2f MiB, out_len=%s, in_len=%s, "
                    "nbatches=%s, batchsz=%s, nparam=%s"
                ),
                last_idx,
                latency_metrics["time_mean(sec)"],
                latency_metrics["time_std"],
                latency_metrics["th_mean(tokens/s)"],
                latency_metrics["th_std"],
                latency_metrics["mem"],
                latency_metrics["out_len"],
                latency_metrics["in_len"],
                latency_metrics["nbatches"],
                latency_metrics["batchsz"],
                latency_metrics["nparam"],
            )

    training_log_rows = _training_log_candidate_rows(
        args,
        mask_candidates,
        checkpoint_path=checkpoint_path,
        skip_single_candidate_postprocessing=skip_single_candidate_postprocessing,
        report_sorted_final_candidates=report_sorted_final_candidates,
        report_all_final_candidates=report_all_final_candidates,
        last_idx=last_idx,
        final_ppl_results=final_ppl_results or None,
        downstream_avg_acc=standalone_downstream_avg_acc,
        downstream_task_accs=standalone_downstream_task_accs,
    )
    try:
        _append_training_log_csv(args.training_log_csv_path, training_log_rows)
    except Exception as exc:
        logging.exception("Failed to append training summary CSV: %s", exc)

    if (
        not structural_pruned
        and model is not None
        and model_matches_last_candidate
        and last_attn_mask is not None
        and last_mlp_mask is not None
    ):
        unmaskModel(model, last_attn_mask, last_mlp_mask)

    logging.info("Done.")
