"""Distribute retained Pareto candidates across GPU workers."""

from llm_pruning.reporting import (
    _layer_prune_ratios_from_keep_ratios,
    save_layer_component_pruning_plot,
    save_layerwise_allocation_plot,
)
from llm_pruning.bayesian import (
    _bayes_optimize_layer_keep_ratios,
    _parse_downstream_gpu_ids,
    load_tokenizer,
)
from llm_pruning.calibration import _evaluate_perplexity_suite_local
from llm_pruning.checkpoints import _candidate_stage2_completed
import copy
import gc
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from typing import Dict, List, Sequence, Tuple
from llm_pruning.latency import measure_latency
from llm_pruning.refinement.second_stage import second_stage_optimize_heads
from llm_pruning.core.utilities import (
    count_params_with_masks,
    get_parameter_counts,
    loadModel,
    maskModel,
)


def _stage2_bo_candidate_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Run one complete stage-2 BO refinement for one Pareto candidate on one GPU."""
    gpu_id = int(payload["gpu_id"])
    idx = int(payload["candidate_idx"])
    candidate = copy.deepcopy(payload["candidate"])
    model_name = str(payload["model_name"])
    cache_dir = payload.get("cache_dir")
    bo_eval_datasets = payload.get("bo_eval_datasets") or []
    device = f"cuda:{gpu_id}"

    logging.basicConfig(
        level=getattr(
            logging, str(payload.get("logging", "INFO")).upper(), logging.INFO
        ),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for candidate-parallel stage-2 BO.")
    torch.cuda.set_device(gpu_id)
    logging.info(
        "Stage-2 candidate-parallel worker on %s running BO for Pareto candidate %d.",
        device,
        idx,
    )

    attn_mask = candidate.get("attn")
    mlp_mask = candidate.get("mlp")
    if attn_mask is None or mlp_mask is None:
        raise ValueError(f"Candidate {idx} is missing attn/mlp masks.")

    model = None
    try:
        model = loadModel(model_name, cache_dir, device_map={"": device})
        maskModel(model, attn_mask, mlp_mask)
        total_params, active_params = count_params_with_masks(
            model, attn_mask, mlp_mask
        )
        pruned_params = total_params - active_params
        stage1_pruned_params = float(pruned_params)
        before_total_ss, before_nonzero_ss = get_parameter_counts(model)

        if bool(payload.get("measure_stage1_plot_ppl")) and bo_eval_datasets:
            stage1_plot_ppl, stage1_plot_breakdown = _evaluate_perplexity_suite_local(
                model,
                bo_eval_datasets,
                seq_len=int(payload["seq_len"]),
                device=device,
            )
            candidate["stage1_ppl"] = float(stage1_plot_ppl)
            candidate["stage1_ppl_label"] = "Stage-1 Calibration-Mix PPL"
            candidate["stage1_ppl_breakdown"] = stage1_plot_breakdown
        elif candidate.get("fitness") is not None:
            candidate["stage1_ppl"] = float(candidate["fitness"])
            candidate["stage1_ppl_label"] = "Stage-1 Fitness"

        pruned_ratio_stage1 = (pruned_params / total_params) if total_params else 0.0
        remaining_budget = max(0.0, float(payload["sparsity"]) - pruned_ratio_stage1)
        derived_second_stage_keep_ratio = max(0.0, min(1.0, 1.0 - remaining_budget))
        candidate_bo_ppl = None
        candidate_bo_objective = str(
            payload.get("second_stage_bayes_objective", "ppl") or "ppl"
        )
        layer_keep_ratios_for_stage2 = None
        stage2_allocation_params = None
        stage2_layer_stats = []

        if remaining_budget > 0.0:
            (
                layer_keep_ratios_for_stage2,
                bo_best_ppl,
                bo_best_params,
            ) = _bayes_optimize_layer_keep_ratios(
                model,
                num_layers=int(payload["num_blocks"]),
                target_prune_ratio=remaining_budget,
                trials=int(payload["second_stage_bayes_trials"]),
                warmup=int(payload["second_stage_bayes_warmup"]),
                seed=int(payload["seed"]),
                max_ratio=derived_second_stage_keep_ratio,
                prune_heads=bool(payload["second_stage_heads"]),
                prune_mlp=bool(payload["second_stage_mlp"]),
                prune_embedding=bool(payload["second_stage_embedding"]),
                prune_norm=bool(payload["second_stage_norm"]),
                dataset_name=payload["prompt_source"],
                max_prompts=int(payload["second_stage_num_prompts"]),
                dataset_max_length=int(payload["second_stage_prompt_max_length"]),
                bo_eval_datasets=bo_eval_datasets,
                seq_len=int(payload["seq_len"]),
                bo_objective=candidate_bo_objective,
                importance_metric=str(payload["second_stage_importance_metric"]),
                scoring_method=str(payload["second_stage_method"]),
                flap_metric=str(payload["second_stage_flap_metric"]),
                min_keep_ratio=float(payload["second_stage_bayes_min_keep_ratio"]),
                early_layer_min_keep_ratio=payload.get(
                    "second_stage_bayes_early_min_keep_ratio"
                ),
                early_layer_count=int(payload["second_stage_bayes_early_layer_count"]),
                early_layer_max_prune_factor=float(
                    payload["second_stage_bayes_early_max_prune_factor"]
                ),
                tokenizer=load_tokenizer(model_name),
                latency_batch_size=int(
                    payload.get("second_stage_bayes_latency_batch_size", 1)
                ),
                latency_seq_len=int(
                    payload.get("second_stage_bayes_latency_seq_len", 64)
                ),
                latency_num_runs=int(payload.get("second_stage_bayes_latency_runs", 5)),
                latency_num_warmup_runs=int(
                    payload.get("second_stage_bayes_latency_warmup", 2)
                ),
                latency_input_prompt=str(
                    payload.get(
                        "second_stage_bayes_latency_prompt",
                        "The Leaning Tower of Pisa is known for",
                    )
                ),
                parallel_gpus="",
                logging_level=str(payload.get("logging", "INFO")),
            )
            candidate_bo_ppl = float(bo_best_ppl)
            stage2_allocation_params = {
                "bo_k": float(bo_best_params[0]),
                "bo_c": float(bo_best_params[1]),
            }
            candidate["layerwise_allocation_plot"] = save_layerwise_allocation_plot(
                layer_keep_ratios_for_stage2,
                candidate_idx=idx,
                model_name=model_name,
                graphs_dir="graphs",
            )
            second_stage_result = second_stage_optimize_heads(
                model,
                max_ratio=derived_second_stage_keep_ratio,
                layer_keep_ratios=layer_keep_ratios_for_stage2,
                layer_component_masks=None,
                prune_heads=bool(payload["second_stage_heads"]),
                prune_mlp=bool(payload["second_stage_mlp"]),
                prune_embedding=bool(payload["second_stage_embedding"]),
                prune_norm=bool(payload["second_stage_norm"]),
                dataset_name=payload["prompt_source"],
                max_prompts=int(payload["second_stage_num_prompts"]),
                dataset_max_length=int(payload["second_stage_prompt_max_length"]),
                importance_metric=str(payload["second_stage_importance_metric"]),
                scoring_method=str(payload["second_stage_method"]),
                seed=int(payload["seed"]),
                flap_metric=str(payload["second_stage_flap_metric"]),
            )
            stage2_layer_stats = copy.deepcopy(
                second_stage_result.get("layer_stats", [])
            )

        after_total, after_nonzero = get_parameter_counts(model)
        stage2_pruned_params = float(max(0, before_nonzero_ss - after_nonzero))
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
        candidate["final_active_params"] = float(after_nonzero)

        if candidate_bo_ppl is not None:
            candidate["bo_objective"] = candidate_bo_objective
            candidate["bo_objective_score"] = candidate_bo_ppl
            candidate["bo_calibration_score"] = candidate_bo_ppl
            if candidate_bo_objective == "latency":
                candidate["bo_latency_score"] = candidate_bo_ppl
            else:
                candidate["bo_calibration_ppl"] = candidate_bo_ppl
            if (
                candidate_bo_objective != "latency"
                and str(payload.get("proposed_first_objective")) == "ppl_mix"
            ):
                candidate["final_plot_ppl"] = candidate_bo_ppl
                candidate["final_plot_ppl_label"] = "Post-Stage-2 BO Calibration Score"
        if remaining_budget > 0.0:
            candidate["stage2_keep_ratio"] = float(derived_second_stage_keep_ratio)
            candidate["stage2_layer_keep_ratios"] = copy.deepcopy(
                layer_keep_ratios_for_stage2
            )
            candidate["stage2_layer_prune_ratios"] = (
                _layer_prune_ratios_from_keep_ratios(layer_keep_ratios_for_stage2)
                if layer_keep_ratios_for_stage2 is not None
                else None
            )
            candidate["stage2_importance_metric"] = str(
                payload["second_stage_importance_metric"]
            )
            candidate["stage2_allocation_method"] = "proposed"
            candidate["stage2_allocation_params"] = copy.deepcopy(
                stage2_allocation_params
            )

        candidate["layer_component_pruning_plot"] = save_layer_component_pruning_plot(
            candidate_idx=idx,
            model_name=model_name,
            stage1_attn_mask=attn_mask,
            stage1_mlp_mask=mlp_mask,
            stage2_layer_stats=stage2_layer_stats,
            graphs_dir="graphs",
        )
        if bool(payload.get("evaluate_pareto_candidate_ppl")) and bo_eval_datasets:
            ppl_candidate, ppl_breakdown = _evaluate_perplexity_suite_local(
                model,
                bo_eval_datasets,
                seq_len=int(payload["seq_len"]),
                device=device,
            )
            candidate["calibration_mix_ppl"] = float(ppl_candidate)
            candidate["calibration_mix_ppl_breakdown"] = ppl_breakdown
            candidate["final_plot_ppl"] = float(ppl_candidate)
            candidate["final_plot_ppl_label"] = "Post-Stage-2 Calibration-Mix PPL"

        if (
            candidate.get("stage1_ppl") is not None
            and candidate.get("final_plot_ppl") is None
        ):
            candidate["final_plot_ppl"] = float(candidate["stage1_ppl"])
            candidate["final_plot_ppl_label"] = "Fallback: Stage-1 metric reused"

        if bool(payload.get("measure_final_candidate_latency")):
            try:
                latency_metrics, _, _ = measure_latency(
                    model=model,
                    tokenizer=load_tokenizer(model_name),
                    device=device,
                    seed=int(payload["seed"]),
                )
                candidate["final_latency"] = float(latency_metrics["time_mean(sec)"])
            except Exception as exc:
                candidate["final_latency"] = None
                logging.exception(
                    "Final latency measurement failed for candidate %d: %s", idx, exc
                )

        candidate["_stage2_completed"] = True
        stagewise_record = {
            "candidate_idx": float(idx),
            "total_params": float(total_params),
            "stage1_pruned_params": stage1_pruned_params,
            "stage2_pruned_params": stage2_pruned_params,
        }
        logging.info(
            "Stage-2 candidate-parallel worker finished candidate %d with bo_score=%s.",
            idx,
            f"{candidate_bo_ppl:.4f}" if candidate_bo_ppl is not None else "n/a",
        )
        return {
            "candidate_idx": idx,
            "candidate": candidate,
            "stagewise_record": stagewise_record,
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()


def _evaluate_stage2_bo_candidates_parallel(
    candidates: Sequence[Dict[str, object]],
    args,
    *,
    num_blocks: int,
    bo_eval_datasets: Sequence[Tuple[str, torch.Tensor]],
    prompt_source,
    evaluate_pareto_candidate_ppl: bool,
    measure_stage1_plot_ppl: bool,
    measure_final_candidate_latency: bool,
) -> List[Dict[str, object]]:
    """Map independent Pareto-candidate BO jobs across the requested GPUs."""
    gpu_ids = _parse_downstream_gpu_ids(args.second_stage_bayes_candidate_parallel_gpus)
    pending = [
        (idx, candidate)
        for idx, candidate in enumerate(candidates, start=1)
        if not _candidate_stage2_completed(candidate)
    ]
    if not pending:
        return []
    if not gpu_ids:
        raise ValueError(
            "--second_stage_bayes_candidate_parallel_gpus did not resolve any GPU ids."
        )
    logging.info(
        "Running candidate-parallel stage-2 BO for %d Pareto candidate(s) across GPUs %s.",
        len(pending),
        gpu_ids,
    )

    results = []
    for start in range(0, len(pending), len(gpu_ids)):
        batch = pending[start : start + len(gpu_ids)]
        payloads = []
        for gpu_id, (idx, candidate) in zip(gpu_ids, batch):
            payloads.append(
                {
                    "gpu_id": int(gpu_id),
                    "candidate_idx": int(idx),
                    "candidate": copy.deepcopy(candidate),
                    "model_name": args.model,
                    "cache_dir": args.cache_dir,
                    "num_blocks": int(num_blocks),
                    "sparsity": float(args.sparsity),
                    "second_stage_bayes_trials": int(args.second_stage_bayes_trials),
                    "second_stage_bayes_warmup": int(args.second_stage_bayes_warmup),
                    "second_stage_bayes_objective": args.second_stage_bayes_objective,
                    "second_stage_bayes_latency_batch_size": int(
                        args.second_stage_bayes_latency_batch_size
                    ),
                    "second_stage_bayes_latency_seq_len": int(
                        args.second_stage_bayes_latency_seq_len
                    ),
                    "second_stage_bayes_latency_runs": int(
                        args.second_stage_bayes_latency_runs
                    ),
                    "second_stage_bayes_latency_warmup": int(
                        args.second_stage_bayes_latency_warmup
                    ),
                    "second_stage_bayes_latency_prompt": args.second_stage_bayes_latency_prompt,
                    "seed": int(args.seed),
                    "second_stage_heads": bool(args.second_stage_heads),
                    "second_stage_mlp": bool(args.second_stage_mlp),
                    "second_stage_embedding": bool(args.second_stage_embedding),
                    "second_stage_norm": bool(args.second_stage_norm),
                    "prompt_source": prompt_source,
                    "second_stage_num_prompts": int(args.second_stage_num_prompts),
                    "second_stage_prompt_max_length": int(
                        args.second_stage_prompt_max_length
                    ),
                    "bo_eval_datasets": bo_eval_datasets,
                    "seq_len": int(args.seq_len),
                    "second_stage_importance_metric": args.second_stage_importance_metric,
                    "second_stage_method": args.second_stage_method,
                    "second_stage_flap_metric": args.second_stage_flap_metric,
                    "second_stage_bayes_min_keep_ratio": float(
                        args.second_stage_bayes_min_keep_ratio
                    ),
                    "second_stage_bayes_early_min_keep_ratio": args.second_stage_bayes_early_min_keep_ratio,
                    "second_stage_bayes_early_layer_count": int(
                        args.second_stage_bayes_early_layer_count
                    ),
                    "second_stage_bayes_early_max_prune_factor": float(
                        args.second_stage_bayes_early_max_prune_factor
                    ),
                    "proposed_first_objective": args.proposed_first_objective,
                    "evaluate_pareto_candidate_ppl": bool(
                        evaluate_pareto_candidate_ppl
                    ),
                    "measure_stage1_plot_ppl": bool(measure_stage1_plot_ppl),
                    "measure_final_candidate_latency": bool(
                        measure_final_candidate_latency
                    ),
                    "logging": args.logging,
                }
            )
        with ProcessPoolExecutor(
            max_workers=len(payloads),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(_stage2_bo_candidate_worker, payload)
                for payload in payloads
            ]
            for future in as_completed(futures):
                results.append(future.result())
    return sorted(results, key=lambda item: int(item["candidate_idx"]))
