"""Parallel perplexity and downstream evaluation workers."""

from llm_pruning.replay import (
    _load_replayed_recovery_model,
    _replay_candidate_on_fresh_model,
)
import argparse
from llm_pruning.bayesian import _parse_downstream_gpu_ids
import copy
import gc
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch
from typing import Dict, List, Optional, Sequence, Tuple
from llm_pruning.core.evaluation import evaluate_perplexity, eval_zero_shot


def _split_tasks_across_gpus(
    tasks: Sequence[str], gpu_ids: Sequence[int]
) -> List[Tuple[int, List[str]]]:
    """Round-robin downstream task shards over the requested GPU ids."""
    if not tasks or not gpu_ids:
        return []
    worker_count = min(len(tasks), len(gpu_ids))
    shards = [(int(gpu_ids[idx]), []) for idx in range(worker_count)]
    for idx, task in enumerate(tasks):
        shards[idx % worker_count][1].append(task)
    return [(gpu_id, shard_tasks) for gpu_id, shard_tasks in shards if shard_tasks]


def _split_named_items_across_gpus(
    items: Sequence[Tuple[str, object]], gpu_ids: Sequence[int]
):
    """Round-robin named evaluation items over the requested GPU ids."""
    if not items or not gpu_ids:
        return []
    worker_count = min(len(items), len(gpu_ids))
    shards = [(int(gpu_ids[idx]), []) for idx in range(worker_count)]
    for idx, item in enumerate(items):
        shards[idx % worker_count][1].append(item)
    return [(gpu_id, shard_items) for gpu_id, shard_items in shards if shard_items]


def _extract_accuracy(task_result: Dict[str, object]) -> float:
    """Extract the highest available accuracy-like metric from an lm-eval task result."""
    metric_values = []
    for key in ("acc_norm", "acc,none", "acc"):
        if key in task_result:
            try:
                metric_values.append(float(task_result[key]))
            except (TypeError, ValueError):
                continue
    if metric_values:
        return max(metric_values)
    for val in task_result.values():
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def _log_zero_shot_results(results: Dict[str, object]) -> None:
    """Log lm-eval results with the same task-by-task detail as the standalone path."""
    logging.info("Zero-shot evaluation results")
    task_results = results.get("results", {})
    if not isinstance(task_results, dict):
        return
    for task_name, task_res in task_results.items():
        if not isinstance(task_res, dict):
            continue
        alias = str(task_res.get("alias", task_name))
        acc = _extract_accuracy(task_res)
        logging.info("%s : %s", alias, acc)


def evaluate_benchmark_average(
    model,
    tokenizer,
    model_name: str,
    tasks: List[str],
    *,
    batch_size=1,
    log_zero_shot_results: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, object]]:
    """Run multiple downstream benchmarks and return average accuracy across tasks."""
    results = eval_zero_shot(
        model_name,
        model,
        task_list=tasks,
        num_fewshot=0,
        use_accelerate=True,
        access_token=os.environ.get("HF_TOKEN"),
        batch_size=batch_size,
    )
    if log_zero_shot_results:
        _log_zero_shot_results(results)
    task_results = results.get("results", {})
    per_task: Dict[str, float] = {}
    scores: List[float] = []
    for task in tasks:
        task_res = task_results.get(task, {})
        if isinstance(task_res, dict) and task_res:
            acc = _extract_accuracy(task_res)
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
            acc = sum(grouped_scores) / len(grouped_scores)
        per_task[task] = acc
        scores.append(acc)
    avg_acc = sum(scores) / len(scores) if scores else 0.0
    return avg_acc, per_task, results


def _merge_lm_eval_results(
    result_payloads: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Merge lm-eval result dictionaries emitted by task-parallel workers."""
    merged: Dict[str, object] = {}
    for payload in result_payloads:
        raw_results = payload.get("raw_results", {})
        if not isinstance(raw_results, dict):
            continue
        for key, value in raw_results.items():
            if isinstance(value, dict):
                merged_section = merged.setdefault(key, {})
                if isinstance(merged_section, dict):
                    merged_section.update(value)
            elif key not in merged:
                merged[key] = value
    return merged


def _downstream_task_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Evaluate a candidate task shard in a fresh process pinned to one CUDA device."""
    gpu_id = int(payload["gpu_id"])
    candidate = payload["candidate"]
    args = argparse.Namespace(**payload["args"])
    model_name = str(payload["model_name"])
    tasks = list(payload["tasks"])
    batch_size = payload.get("batch_size", 1)

    logging.basicConfig(
        level=getattr(logging, str(args.logging).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for downstream parallel evaluation.")
    torch.cuda.set_device(gpu_id)
    device_map = {"": f"cuda:{gpu_id}"}
    logging.info(
        "Downstream worker on cuda:%d evaluating tasks: %s",
        gpu_id,
        tasks,
    )
    model = _replay_candidate_on_fresh_model(
        candidate,
        args,
        model_name=model_name,
        device_map=device_map,
    )
    try:
        avg_acc, per_task, raw_results = evaluate_benchmark_average(
            model,
            None,
            model_name,
            tasks,
            batch_size=batch_size,
            log_zero_shot_results=False,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    return {
        "gpu_id": gpu_id,
        "tasks": tasks,
        "avg_acc": avg_acc,
        "per_task": per_task,
        "raw_results": raw_results,
    }


def _evaluate_candidate_downstream_parallel(
    candidate: Dict[str, object],
    model_name: str,
    tasks: Sequence[str],
    args,
) -> Optional[Tuple[float, Dict[str, float], Dict[str, object]]]:
    """Run a final-candidate downstream eval by splitting tasks across GPUs."""
    gpu_ids = _parse_downstream_gpu_ids(args.downstream_parallel_gpus)
    shards = _split_tasks_across_gpus(tasks, gpu_ids)
    if len(shards) <= 1:
        return None

    logging.info(
        "Running task-parallel downstream evaluation across GPUs %s.",
        [gpu_id for gpu_id, _ in shards],
    )
    args_payload = dict(vars(args))
    futures = []
    executor_kwargs = {
        "max_workers": len(shards),
        "mp_context": mp.get_context("spawn"),
    }
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for gpu_id, shard_tasks in shards:
            futures.append(
                executor.submit(
                    _downstream_task_worker,
                    {
                        "gpu_id": gpu_id,
                        "candidate": copy.deepcopy(candidate),
                        "args": args_payload,
                        "model_name": model_name,
                        "tasks": shard_tasks,
                        "batch_size": args.downstream_batch_size,
                    },
                )
            )
        payloads = [future.result() for future in as_completed(futures)]

    per_task: Dict[str, float] = {}
    for payload in payloads:
        task_scores = payload.get("per_task", {})
        if isinstance(task_scores, dict):
            per_task.update(task_scores)
    ordered_scores = [per_task[task] for task in tasks if task in per_task]
    avg_acc = sum(ordered_scores) / len(ordered_scores) if ordered_scores else 0.0
    raw_results = _merge_lm_eval_results(payloads)
    return avg_acc, per_task, raw_results


def _downstream_model_dir_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Evaluate downstream task shards by loading a saved model directory on one GPU."""
    gpu_id = int(payload["gpu_id"])
    model_dir = str(payload["model_dir"])
    base_model_name = str(payload.get("base_model_name") or model_dir)
    cache_dir = payload.get("cache_dir")
    logging_level = str(payload.get("logging", "INFO"))
    tasks = list(payload["tasks"])
    batch_size = payload.get("batch_size", 1)

    logging.basicConfig(
        level=getattr(logging, logging_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for downstream parallel evaluation.")
    torch.cuda.set_device(gpu_id)
    device_map = {"": f"cuda:{gpu_id}"}
    logging.info(
        "Downstream worker on cuda:%d loading %s and evaluating tasks: %s",
        gpu_id,
        model_dir,
        tasks,
    )
    model = _load_replayed_recovery_model(
        model_dir,
        base_model_name,
        cache_dir,
        device_map=device_map,
    )
    try:
        avg_acc, per_task, raw_results = evaluate_benchmark_average(
            model,
            None,
            model_dir,
            tasks,
            batch_size=batch_size,
            log_zero_shot_results=False,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    return {
        "gpu_id": gpu_id,
        "tasks": tasks,
        "avg_acc": avg_acc,
        "per_task": per_task,
        "raw_results": raw_results,
    }


def _evaluate_model_dir_downstream_parallel(
    model_dir: Path,
    tasks: Sequence[str],
    args,
) -> Optional[Tuple[float, Dict[str, float], Dict[str, object]]]:
    """Run downstream eval for a saved tuned model by splitting tasks across GPUs."""
    gpu_ids = _parse_downstream_gpu_ids(args.downstream_parallel_gpus)
    shards = _split_tasks_across_gpus(tasks, gpu_ids)
    if len(shards) <= 1:
        return None

    logging.info(
        "Running task-parallel downstream evaluation for %s across GPUs %s.",
        model_dir,
        [gpu_id for gpu_id, _ in shards],
    )
    futures = []
    executor_kwargs = {
        "max_workers": len(shards),
        "mp_context": mp.get_context("spawn"),
    }
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for gpu_id, shard_tasks in shards:
            futures.append(
                executor.submit(
                    _downstream_model_dir_worker,
                    {
                        "gpu_id": gpu_id,
                        "model_dir": str(model_dir),
                        "base_model_name": args.model,
                        "cache_dir": args.cache_dir,
                        "logging": args.logging,
                        "tasks": shard_tasks,
                        "batch_size": args.downstream_batch_size,
                    },
                )
            )
        payloads = [future.result() for future in as_completed(futures)]

    per_task: Dict[str, float] = {}
    for payload in payloads:
        task_scores = payload.get("per_task", {})
        if isinstance(task_scores, dict):
            per_task.update(task_scores)
    ordered_scores = [per_task[task] for task in tasks if task in per_task]
    avg_acc = sum(ordered_scores) / len(ordered_scores) if ordered_scores else 0.0
    raw_results = _merge_lm_eval_results(payloads)
    return avg_acc, per_task, raw_results


def _perplexity_dataset_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Evaluate assigned perplexity datasets in a fresh process pinned to one CUDA device."""
    gpu_id = int(payload["gpu_id"])
    candidate = payload["candidate"]
    args = argparse.Namespace(**payload["args"])
    model_name = str(payload["model_name"])
    datasets = list(payload["datasets"])
    batch_size = int(payload.get("batch_size", 1))

    logging.basicConfig(
        level=getattr(logging, str(args.logging).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for parallel perplexity evaluation.")
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    device_map = {"": device}
    logging.info(
        "Perplexity worker on %s evaluating datasets: %s",
        device,
        [name for name, _ in datasets],
    )
    model = _replay_candidate_on_fresh_model(
        candidate,
        args,
        model_name=model_name,
        device_map=device_map,
    )
    try:
        results = {}
        for dataset_name, input_ids in datasets:
            if input_ids is None:
                continue
            results[dataset_name] = evaluate_perplexity(
                model,
                input_ids,
                seq_len=args.seq_len,
                batch_size=batch_size,
                device=device,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    return {
        "gpu_id": gpu_id,
        "results": results,
    }


def _evaluate_candidate_perplexity_parallel(
    candidate: Dict[str, object],
    model_name: str,
    datasets: Sequence[Tuple[str, torch.Tensor]],
    args,
) -> Optional[Dict[str, float]]:
    """Run final-candidate perplexity by splitting datasets across GPUs."""
    valid_datasets = [(name, ids) for name, ids in datasets if ids is not None]
    gpu_ids = _parse_downstream_gpu_ids(args.perplexity_parallel_gpus)
    shards = _split_named_items_across_gpus(valid_datasets, gpu_ids)
    if len(shards) <= 1:
        return None

    logging.info(
        "Running dataset-parallel perplexity evaluation across GPUs %s.",
        [gpu_id for gpu_id, _ in shards],
    )
    args_payload = dict(vars(args))
    futures = []
    executor_kwargs = {
        "max_workers": len(shards),
        "mp_context": mp.get_context("spawn"),
    }
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for gpu_id, shard_datasets in shards:
            futures.append(
                executor.submit(
                    _perplexity_dataset_worker,
                    {
                        "gpu_id": gpu_id,
                        "candidate": copy.deepcopy(candidate),
                        "args": args_payload,
                        "model_name": model_name,
                        "datasets": shard_datasets,
                        "batch_size": args.perplexity_batch_size,
                    },
                )
            )
        payloads = [future.result() for future in as_completed(futures)]

    results: Dict[str, float] = {}
    for payload in payloads:
        worker_results = payload.get("results", {})
        if isinstance(worker_results, dict):
            results.update(
                {str(name): float(value) for name, value in worker_results.items()}
            )
    return results


def _perplexity_model_dir_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Evaluate perplexity datasets by loading a saved model directory on one GPU."""
    gpu_id = int(payload["gpu_id"])
    model_dir = str(payload["model_dir"])
    base_model_name = str(payload.get("base_model_name") or model_dir)
    cache_dir = payload.get("cache_dir")
    logging_level = str(payload.get("logging", "INFO"))
    datasets = list(payload["datasets"])
    seq_len = int(payload["seq_len"])
    batch_size = int(payload.get("batch_size", 1))

    logging.basicConfig(
        level=getattr(logging, logging_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for parallel perplexity evaluation.")
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    logging.info(
        "Perplexity worker on %s loading %s and evaluating datasets: %s",
        device,
        model_dir,
        [name for name, _ in datasets],
    )
    model = _load_replayed_recovery_model(
        model_dir,
        base_model_name,
        cache_dir,
        device_map={"": device},
    )
    try:
        results = {}
        for dataset_name, input_ids in datasets:
            if input_ids is None:
                continue
            results[dataset_name] = evaluate_perplexity(
                model,
                input_ids,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    return {
        "gpu_id": gpu_id,
        "results": results,
    }


def _evaluate_model_dir_perplexity_parallel(
    model_dir: Path,
    datasets: Sequence[Tuple[str, torch.Tensor]],
    args,
) -> Optional[Dict[str, float]]:
    """Run perplexity for a saved tuned model by splitting datasets across GPUs."""
    valid_datasets = [(name, ids) for name, ids in datasets if ids is not None]
    gpu_ids = _parse_downstream_gpu_ids(args.perplexity_parallel_gpus)
    shards = _split_named_items_across_gpus(valid_datasets, gpu_ids)
    if len(shards) <= 1:
        return None

    logging.info(
        "Running dataset-parallel perplexity evaluation for %s across GPUs %s.",
        model_dir,
        [gpu_id for gpu_id, _ in shards],
    )
    futures = []
    executor_kwargs = {
        "max_workers": len(shards),
        "mp_context": mp.get_context("spawn"),
    }
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for gpu_id, shard_datasets in shards:
            futures.append(
                executor.submit(
                    _perplexity_model_dir_worker,
                    {
                        "gpu_id": gpu_id,
                        "model_dir": str(model_dir),
                        "base_model_name": args.model,
                        "cache_dir": args.cache_dir,
                        "logging": args.logging,
                        "datasets": shard_datasets,
                        "seq_len": args.seq_len,
                        "batch_size": args.perplexity_batch_size,
                    },
                )
            )
        payloads = [future.result() for future in as_completed(futures)]

    results: Dict[str, float] = {}
    for payload in payloads:
        worker_results = payload.get("results", {})
        if isinstance(worker_results, dict):
            results.update(
                {str(name): float(value) for name, value in worker_results.items()}
            )
    return results
