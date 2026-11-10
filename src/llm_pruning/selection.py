"""Evaluate and select final Pareto candidates."""

from llm_pruning.metrics import _candidate_latency_sort_key, _parse_downstream_tasks
from llm_pruning.evaluation_workers import (
    _evaluate_candidate_downstream_parallel,
    _log_zero_shot_results,
    evaluate_benchmark_average,
)
from llm_pruning.export import _get_candidate_bo_score
from llm_pruning.replay import _replay_candidate_on_fresh_model
from llm_pruning.config import DEFAULT_DOWNSTREAM_TASKS
import logging
from typing import Dict, Optional, Sequence
from llm_pruning.core.utilities import count_params_with_masks, maskModel, unmaskModel


def select_best_pareto_candidate(
    candidates,
    model,
    tokenizer,
    model_name: str,
    alpha: float,
    benchmark_tasks: Optional[Sequence[str]] = None,
):
    """Evaluate Pareto candidates on multiple benchmarks, score them, and return the best one."""
    alpha = max(0.0, min(1.0, alpha))
    accs = []
    params = []
    prev_attn = None
    prev_mlp = None
    benchmark_tasks = list(benchmark_tasks or DEFAULT_DOWNSTREAM_TASKS)

    # Evaluate each candidate
    for idx, cand in enumerate(candidates, start=1):
        attn_mask = cand.get("attn")
        mlp_mask = cand.get("mlp")
        if attn_mask is None or mlp_mask is None:
            raise ValueError("Pareto candidate is missing attn/mlp masks.")
        if prev_attn is not None and prev_mlp is not None:
            unmaskModel(model, prev_attn, prev_mlp)
        maskModel(model, attn_mask, mlp_mask)
        prev_attn, prev_mlp = attn_mask, mlp_mask

        acc, per_task, _ = evaluate_benchmark_average(
            model,
            tokenizer,
            model_name,
            benchmark_tasks,
        )
        if "active_params" in cand and cand["active_params"] is not None:
            active_params = float(cand["active_params"])
        else:
            _, active_params = count_params_with_masks(model, attn_mask, mlp_mask)
        cand["avg_accuracy"] = acc
        cand["task_accuracies"] = per_task
        cand["active_params"] = active_params
        accs.append(acc)
        params.append(active_params)
        logging.info(
            "Pareto candidate %d — avg acc: %.4f | active_params: %s | tasks: %s",
            idx,
            acc,
            active_params,
            {k: f"{v:.4f}" for k, v in per_task.items()},
        )

    # Restore model to unmasked state before scoring
    if prev_attn is not None and prev_mlp is not None:
        unmaskModel(model, prev_attn, prev_mlp)

    acc_min, acc_max = min(accs), max(accs)
    param_min, param_max = min(params), max(params)

    def _norm(val, vmin, vmax):
        if vmax == vmin:
            return 1.0
        return (val - vmin) / (vmax - vmin)

    best_idx = 0
    best_score = float("-inf")
    for idx, cand in enumerate(candidates):
        acc_norm = _norm(cand["avg_accuracy"], acc_min, acc_max)
        param_norm = _norm(cand["active_params"], param_min, param_max)
        score = alpha * acc_norm - (1 - alpha) * param_norm
        cand["score_S"] = score
        logging.info(
            "Pareto candidate %d — acc_norm=%.4f param_norm=%.4f S=%.4f",
            idx + 1,
            acc_norm,
            param_norm,
            score,
        )
        if score > best_score:
            best_score = score
            best_idx = idx

    # Print summary of all candidates with raw Hellaswag and params
    logging.info("=== Pareto candidates summary (avg accuracy, active_params, S) ===")
    for idx, cand in enumerate(candidates, start=1):
        logging.info(
            "Candidate %d: avg_acc=%.4f active_params=%s S=%.4f",
            idx,
            cand.get("avg_accuracy"),
            cand.get("active_params"),
            cand.get("score_S"),
        )

    return candidates[best_idx], candidates


def _log_final_pareto_summary(candidates: Sequence[Dict[str, object]]) -> None:
    logging.info(
        "=== Final Pareto candidates sorted by latency then BO calibration score ==="
    )
    for rank, cand in enumerate(
        sorted(candidates, key=_candidate_latency_sort_key), start=1
    ):
        latency = cand.get("final_latency")
        latency_str = f"{latency:.4f}" if latency is not None else "n/a"
        ppl = _get_candidate_bo_score(cand)
        ppl_str = f"{ppl:.4f}" if ppl is not None else "n/a"
        logging.info(
            "Rank %d | candidate=%s | latency=%s sec | bo_score=%s | active_params=%s | stage1_latency=%s | stage1_fitness=%s",
            rank,
            cand.get("candidate_idx"),
            latency_str,
            ppl_str,
            cand.get("final_active_params", cand.get("active_params")),
            cand.get("latency"),
            cand.get("fitness"),
        )


def _log_all_final_candidates(candidates: Sequence[Dict[str, object]]) -> None:
    logging.info("=== Final Pareto candidates in candidate order ===")
    for cand in candidates:
        latency = cand.get("final_latency")
        latency_str = f"{latency:.4f}" if latency is not None else "n/a"
        ppl = _get_candidate_bo_score(cand)
        ppl_str = f"{ppl:.4f}" if ppl is not None else "n/a"
        logging.info(
            "Candidate %s | latency=%s sec | bo_score=%s | active_params=%s | stage1_latency=%s | stage1_fitness=%s",
            cand.get("candidate_idx"),
            latency_str,
            ppl_str,
            cand.get("final_active_params", cand.get("active_params")),
            cand.get("latency"),
            cand.get("fitness"),
        )


def _evaluate_all_final_candidates_downstream(
    candidates: Sequence[Dict[str, object]],
    tokenizer,
    model_name: str,
    args,
) -> None:
    logging.info("=== Final Pareto candidates downstream evaluation ===")
    benchmark_tasks = _parse_downstream_tasks(args.downstream_tasks)
    logging.info("Downstream tasks: %s", benchmark_tasks)
    for cand in candidates:
        attn_mask = cand.get("attn")
        mlp_mask = cand.get("mlp")
        if attn_mask is None or mlp_mask is None:
            continue

        logging.info(
            "Replaying candidate %s on a fresh base model for downstream evaluation.",
            cand.get("candidate_idx"),
        )
        model = None
        try:
            try:
                parallel_results = _evaluate_candidate_downstream_parallel(
                    cand,
                    model_name,
                    benchmark_tasks,
                    args,
                )
            except Exception as exc:
                logging.exception(
                    "Task-parallel downstream evaluation failed for candidate %s; "
                    "falling back to single-process evaluation: %s",
                    cand.get("candidate_idx"),
                    exc,
                )
                parallel_results = None
            if parallel_results is not None:
                avg_acc, per_task, raw_results = parallel_results
                _log_zero_shot_results(raw_results)
            else:
                model = _replay_candidate_on_fresh_model(
                    cand,
                    args,
                    model_name=model_name,
                )
                avg_acc, per_task, raw_results = evaluate_benchmark_average(
                    model,
                    tokenizer,
                    model_name,
                    benchmark_tasks,
                    batch_size=args.downstream_batch_size,
                    log_zero_shot_results=True,
                )
            cand["downstream_avg_acc"] = avg_acc
            cand["downstream_task_accs"] = per_task
            cand["downstream_eval_results"] = raw_results
            logging.info(
                "Candidate %s | downstream_avg_acc=%.4f | tasks=%s | bo_score=%s | final_latency=%s",
                cand.get("candidate_idx"),
                avg_acc,
                {k: f"{v:.4f}" for k, v in per_task.items()},
                (
                    f"{_get_candidate_bo_score(cand):.4f}"
                    if _get_candidate_bo_score(cand) is not None
                    else "n/a"
                ),
                (
                    f"{cand['final_latency']:.4f}"
                    if cand.get("final_latency") is not None
                    else "n/a"
                ),
            )
        finally:
            if model is not None and attn_mask is not None and mlp_mask is not None:
                unmaskModel(model, attn_mask, mlp_mask)
