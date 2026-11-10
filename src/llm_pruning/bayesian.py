"""Bayesian layer-wise allocation and parallel trial workers."""

from llm_pruning.config import BO_INVALID_SCORE_PENALTY
from llm_pruning.allocation import (
    _build_monotone_prune_ratios,
    _normalize_device_map_location,
)
from llm_pruning.calibration import _evaluate_nll_suite_local
import copy
import gc
import logging
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Dict, List, Optional, Sequence, Tuple, Union
from llm_pruning.refinement.second_stage import second_stage_optimize_heads
from llm_pruning.fine_tune import fine_tune_model
from llm_pruning.core.utilities import (
    loadModel,
    maskModel,
    remove_accelerate_hooks_if_present,
)


def _dtype_num_bytes(dtype: Optional[torch.dtype]) -> Optional[int]:
    """Best-effort byte width for a tensor dtype."""
    if dtype is None:
        return None
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return None


def _estimate_parameter_bytes(model) -> int:
    """Estimate memory occupied by model parameters only."""
    total_bytes = 0
    for param in model.parameters():
        itemsize = _dtype_num_bytes(param.dtype)
        if itemsize is None:
            continue
        total_bytes += param.numel() * itemsize
    return total_bytes


def _format_bytes(num_bytes: Optional[Union[int, float]]) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _available_system_memory_bytes() -> Optional[int]:
    """Return currently available host RAM when it can be determined."""
    host_available = None
    try:
        import psutil  # type: ignore

        host_available = int(psutil.virtual_memory().available)
    except Exception:
        pass

    if host_available is None and hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            if isinstance(page_size, int) and isinstance(available_pages, int):
                host_available = int(page_size * available_pages)
        except Exception:
            pass

    cgroup_available = _available_cgroup_memory_bytes()
    if host_available is None:
        return cgroup_available
    if cgroup_available is None:
        return host_available
    return min(host_available, cgroup_available)


def _read_int_from_text_file(path: Union[str, Path]) -> Optional[int]:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw or raw.lower() == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _available_cgroup_memory_bytes() -> Optional[int]:
    """
    Return remaining memory inside the active Linux cgroup, when available.

    Batch schedulers often enforce per-job memory with cgroups. In that case the
    host may have abundant free RAM while the job itself is close to its cgroup
    limit and gets SIGKILL'ed. Honor the tighter cgroup budget when detectable.
    """
    # cgroup v2
    limit = _read_int_from_text_file("/sys/fs/cgroup/memory.max")
    usage = _read_int_from_text_file("/sys/fs/cgroup/memory.current")
    if limit is not None and usage is not None:
        remaining = limit - usage
        if remaining > 0:
            return remaining

    # cgroup v1
    limit = _read_int_from_text_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    usage = _read_int_from_text_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit is not None and usage is not None:
        # Large sentinel values mean "effectively unlimited" in many v1 setups.
        if limit < (1 << 60):
            remaining = limit - usage
            if remaining > 0:
                return remaining
    return None


def _estimate_stage2_bo_host_ram_requirement_bytes(
    model,
    *,
    can_offload_base_model: bool,
) -> int:
    """
    Estimate peak host RAM pressure for stage-2 BO trial cloning.

    The current BO implementation keeps the base model resident and deep-copies it
    per trial. When the base model is first offloaded to CPU, peak host RAM is at
    least two parameter copies before accounting for module/object overhead.
    """
    parameter_bytes = _estimate_parameter_bytes(model)
    if parameter_bytes <= 0:
        return 0
    copy_count = 2 if can_offload_base_model else 1
    # Add conservative overhead for Python/module metadata and temporary tensors.
    return int(parameter_bytes * copy_count * 1.15)


def _raise_if_stage2_bo_clone_is_unlikely_to_fit(
    model,
    *,
    can_offload_base_model: bool,
) -> None:
    """
    Fail early with a clear error instead of letting the OS kill the process.
    """
    required_bytes = _estimate_stage2_bo_host_ram_requirement_bytes(
        model,
        can_offload_base_model=can_offload_base_model,
    )
    if required_bytes <= 0:
        return

    available_bytes = _available_system_memory_bytes()
    if available_bytes is None:
        return

    logging.info(
        "Stage-2 BO host-RAM preflight: estimated requirement=%s, available=%s.",
        _format_bytes(required_bytes),
        _format_bytes(available_bytes),
    )

    # Keep a safety margin because the process already holds tokenizer state,
    # prompt tensors, BO bookkeeping, and may transiently allocate extra buffers.
    safety_limit = int(available_bytes * 0.85)
    if required_bytes <= safety_limit:
        return

    raise RuntimeError(
        "Stage-2 Bayesian optimization clones the full model for each trial. "
        f"Estimated host RAM needed for trial cloning: {_format_bytes(required_bytes)}; "
        f"currently available host RAM: {_format_bytes(available_bytes)}. "
        "This run is likely to be terminated by the OS before Python can raise a CUDA OOM. "
        "Disable --second_stage_bayes, provide --second_stage_layer_keep_ratios manually, "
        "or use a smaller/quantized model for stage-2 search."
    )


def load_tokenizer(model_name: str):
    """Load a tokenizer for the provided model name."""
    logging.info("Loading tokenizer for %s", model_name)
    lower_name = model_name.lower()
    use_llama_slow = (
        "llama" in lower_name
        and "llama-3" not in lower_name
        and "llama3" not in lower_name
    )
    if use_llama_slow:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        except Exception:
            logging.warning(
                "Slow Llama tokenizer failed; retrying with AutoTokenizer fast implementation.",
                exc_info=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=True
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        if isinstance(tokenizer, bool):
            logging.warning(
                "Tokenizer loading returned bool; retrying with the fast implementation."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=True
            )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return tokenizer


def _parse_downstream_gpu_ids(raw_gpu_ids) -> List[int]:
    """Normalize comma-separated GPU ids used for task-parallel downstream eval."""
    if raw_gpu_ids is None:
        return []
    raw_text = str(raw_gpu_ids).strip()
    if not raw_text:
        return []
    if raw_text.lower() == "auto":
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    gpu_ids = []
    seen = set()
    for value in raw_text.split(","):
        value = value.strip().lower()
        if not value:
            continue
        if value.startswith("cuda:"):
            value = value.split(":", 1)[1]
        gpu_id = int(value)
        if gpu_id in seen:
            continue
        seen.add(gpu_id)
        gpu_ids.append(gpu_id)
    return gpu_ids


def _evaluate_bo_perplexity_score(
    model,
    *,
    bo_eval_datasets: Sequence[Tuple[str, torch.Tensor]],
    seq_len: int,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Aggregate BO evaluation score across the configured prompt datasets.

    Use mean token NLL instead of exponentiated perplexity so BO keeps a stable,
    finite objective when aggressive trials would otherwise overflow exp(NLL).
    """
    score, breakdown = _evaluate_nll_suite_local(
        model,
        datasets=bo_eval_datasets,
        seq_len=seq_len,
        device=device,
    )
    if score is None or not math.isfinite(float(score)):
        return BO_INVALID_SCORE_PENALTY, breakdown
    logging.info(
        "Aggregate BO calibration score (%s, mean token NLL) = %.4f",
        ", ".join(name for name, _ in bo_eval_datasets),
        float(score),
    )
    return float(score), breakdown


def _evaluate_stage2_bo_latency_score(
    model,
    tokenizer,
    *,
    device: str,
    input_prompt: str,
    batch_size: int,
    seq_len: int,
    num_runs: int,
    num_warmup_runs: int,
) -> Tuple[float, Dict[str, float]]:
    """Measure a lightweight CUDA forward-pass latency score for stage-2 BO."""
    device_str = str(device)
    if "cuda" not in device_str:
        raise RuntimeError("Stage-2 BO latency objective requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for stage-2 BO latency scoring.")

    batch_size = max(1, int(batch_size))
    seq_len = max(1, int(seq_len))
    num_runs = max(1, int(num_runs))
    num_warmup_runs = max(0, int(num_warmup_runs))
    original_pad_token = tokenizer.pad_token_id
    original_padding_side = tokenizer.padding_side
    try:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            [input_prompt] * batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )["input_ids"].to(device_str)

        model.eval()
        with torch.cuda.device(device_str):
            for _ in range(num_warmup_runs):
                _ = model(inputs, use_cache=False)
            torch.cuda.synchronize()
            timings: List[float] = []
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(inputs, use_cache=False)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end) / 1000.0)
    finally:
        tokenizer.pad_token_id = original_pad_token
        tokenizer.padding_side = original_padding_side

    latency = float(sum(timings) / len(timings))
    throughput = float(inputs.numel() / max(latency, 1e-9))
    breakdown = {
        "latency_sec": latency,
        "throughput_tokens_per_sec": throughput,
        "batch_size": float(batch_size),
        "seq_len": float(seq_len),
        "runs": float(num_runs),
    }
    logging.info(
        "Aggregate BO latency score = %.6f sec (%.2f tokens/s)",
        latency,
        throughput,
    )
    return latency, breakdown


def _bayes_optimize_layer_keep_ratios(
    base_model,
    *,
    num_layers: int,
    target_prune_ratio: float,
    trials: int,
    warmup: int,
    seed: int,
    max_ratio: float,
    prune_heads: bool,
    prune_mlp: bool,
    prune_embedding: bool,
    prune_norm: bool,
    dataset_name,
    max_prompts: int,
    dataset_max_length: int,
    bo_eval_datasets: Sequence[Tuple[str, torch.Tensor]],
    seq_len: int,
    bo_objective: str,
    importance_metric: str,
    scoring_method: str,
    flap_metric: str,
    min_keep_ratio: float,
    early_layer_min_keep_ratio: Optional[float],
    early_layer_count: int,
    early_layer_max_prune_factor: float,
    tokenizer=None,
    latency_batch_size: int = 1,
    latency_seq_len: int = 64,
    latency_num_runs: int = 5,
    latency_num_warmup_runs: int = 2,
    latency_input_prompt: str = "The Leaning Tower of Pisa is known for",
    parallel_gpus: str = "",
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    stage1_attn_mask: Optional[Sequence[int]] = None,
    stage1_mlp_mask: Optional[Sequence[int]] = None,
    logging_level: str = "INFO",
) -> Tuple[List[float], float, Tuple[float, float]]:
    """Bayesian optimization over (k, c) for monotone per-layer pruning allocation."""
    try:
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf_discrete
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except Exception as exc:
        raise RuntimeError(
            "Bayesian optimization requires botorch and gpytorch. "
            "Install them and retry (see requirements.txt)."
        ) from exc

    if trials <= 0:
        raise ValueError("--second_stage_bayes_trials must be > 0.")
    bo_objective = str(bo_objective or "ppl").strip().lower()
    if bo_objective not in {"ppl", "latency"}:
        raise ValueError(
            "second_stage_bayes_objective must be either 'ppl' or 'latency'."
        )
    if bo_objective == "ppl" and not bo_eval_datasets:
        raise ValueError(
            "Bayesian optimization requires prompt datasets for aggregate perplexity evaluation."
        )
    if bo_objective == "latency" and tokenizer is None:
        raise ValueError(
            "Bayesian optimization latency objective requires a tokenizer."
        )

    rng = np.random.default_rng(seed)
    parallel_gpu_ids = _parse_downstream_gpu_ids(parallel_gpus)
    parallel_enabled = (
        len(parallel_gpu_ids) > 1
        and model_name is not None
        and stage1_attn_mask is not None
        and stage1_mlp_mask is not None
    )
    if parallel_gpus and not parallel_enabled:
        logging.warning(
            "Ignoring --second_stage_bayes_parallel_gpus because fewer than two GPUs were resolved "
            "or replay metadata is unavailable."
        )
    if parallel_enabled:
        logging.info(
            "Stage-2 BO will evaluate trial solutions in parallel across GPUs %s.",
            parallel_gpu_ids,
        )
    # Search space: k in [0.1, 20], c in [0.0, 1.0]
    bounds = torch.tensor([[0.1, 0.0], [20.0, 1.0]], dtype=torch.double)
    original_device = next(base_model.parameters()).device
    device_map = getattr(base_model, "hf_device_map", None)
    if device_map:
        mapped_devices = {
            normalized
            for normalized in (
                _normalize_device_map_location(device) for device in device_map.values()
            )
            if normalized is not None
        }
        compute_devices = {
            device for device in mapped_devices if device not in {"cpu", "disk"}
        }
        has_disk_offload = "disk" in mapped_devices
        can_offload_base_model = not has_disk_offload and len(compute_devices) <= 1
        if compute_devices:
            original_device = torch.device(sorted(compute_devices)[0])
    else:
        can_offload_base_model = original_device.type == "cuda"

    if not parallel_enabled:
        _raise_if_stage2_bo_clone_is_unlikely_to_fit(
            base_model,
            can_offload_base_model=can_offload_base_model,
        )

    if can_offload_base_model:
        if device_map:
            logging.info(
                "Removing Accelerate hooks from the base model before stage-2 BO so it can be moved off GPU for trial cloning."
            )
            base_model = remove_accelerate_hooks_if_present(base_model)
        logging.info(
            "Temporarily offloading the base model to CPU during stage-2 BO to avoid GPU OOM while cloning trial models."
        )
        base_model.to("cpu")
        torch.cuda.empty_cache()
    elif device_map:
        logging.info(
            "Keeping the base model on its dispatched devices during stage-2 BO because the device map spans multiple compute devices or disk offload."
        )

    def _evaluate(k_val: float, c_val: float) -> float:
        score_breakdown = {}
        prune_ratios = _build_monotone_prune_ratios(
            num_layers=num_layers,
            target_prune_ratio=target_prune_ratio,
            k=k_val,
            c=c_val,
            min_keep_ratio=min_keep_ratio,
            early_layer_min_keep_ratio=early_layer_min_keep_ratio,
            early_layer_count=early_layer_count,
            early_layer_max_prune_factor=early_layer_max_prune_factor,
        )
        # Convert prune ratios to keep ratios for stage-2.
        keep_ratios = [min(1.0, max(min_keep_ratio, 1.0 - r)) for r in prune_ratios]
        trial_model = None
        try:
            trial_model = copy.deepcopy(base_model)
            if can_offload_base_model:
                trial_model.to(original_device)
            _ = second_stage_optimize_heads(
                trial_model,
                max_ratio=max_ratio,
                layer_keep_ratios=keep_ratios,
                layer_component_masks=None,
                prune_heads=prune_heads,
                prune_mlp=prune_mlp,
                prune_embedding=prune_embedding,
                prune_norm=prune_norm,
                dataset_name=dataset_name,
                max_prompts=max_prompts,
                dataset_max_length=dataset_max_length,
                importance_metric=importance_metric,
                scoring_method=scoring_method,
                seed=seed,
                flap_metric=flap_metric,
            )
            eval_device = str(next(trial_model.parameters()).device)
            if bo_objective == "latency":
                score, score_breakdown = _evaluate_stage2_bo_latency_score(
                    trial_model,
                    tokenizer,
                    device=eval_device,
                    input_prompt=latency_input_prompt,
                    batch_size=latency_batch_size,
                    seq_len=latency_seq_len,
                    num_runs=latency_num_runs,
                    num_warmup_runs=latency_num_warmup_runs,
                )
            else:
                score, score_breakdown = _evaluate_bo_perplexity_score(
                    trial_model,
                    bo_eval_datasets=bo_eval_datasets,
                    seq_len=seq_len,
                    device=eval_device,
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logging.warning("BO eval hit CUDA OOM; assigning penalty. %s", exc)
                score = BO_INVALID_SCORE_PENALTY
                score_breakdown = (
                    {"latency_sec": float("inf")}
                    if bo_objective == "latency"
                    else {name: float("inf") for name, _ in bo_eval_datasets}
                )
            else:
                raise
        finally:
            if trial_model is not None:
                del trial_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        return float(score), score_breakdown

    def _evaluate_parallel(
        trials_to_eval: Sequence[Tuple[float, float]],
    ) -> List[Dict[str, object]]:
        if not parallel_enabled or len(trials_to_eval) <= 1:
            results = []
            for k_val, c_val in trials_to_eval:
                score, score_breakdown = _evaluate(k_val, c_val)
                results.append(
                    {
                        "k": float(k_val),
                        "c": float(c_val),
                        "score": float(score),
                        "score_breakdown": score_breakdown,
                    }
                )
            return results

        gpu_ids = parallel_gpu_ids[: min(len(parallel_gpu_ids), len(trials_to_eval))]
        logging.info(
            "Evaluating %d stage-2 BO trial solution(s) across GPUs %s.",
            len(gpu_ids),
            gpu_ids,
        )
        args_payloads = []
        for gpu_id, (k_val, c_val) in zip(gpu_ids, trials_to_eval):
            args_payloads.append(
                {
                    "gpu_id": gpu_id,
                    "k": float(k_val),
                    "c": float(c_val),
                    "model_name": model_name,
                    "cache_dir": cache_dir,
                    "attn_mask": list(stage1_attn_mask or []),
                    "mlp_mask": list(stage1_mlp_mask or []),
                    "num_layers": num_layers,
                    "target_prune_ratio": target_prune_ratio,
                    "min_keep_ratio": min_keep_ratio,
                    "early_layer_min_keep_ratio": early_layer_min_keep_ratio,
                    "early_layer_count": early_layer_count,
                    "early_layer_max_prune_factor": early_layer_max_prune_factor,
                    "max_ratio": max_ratio,
                    "prune_heads": prune_heads,
                    "prune_mlp": prune_mlp,
                    "prune_embedding": prune_embedding,
                    "prune_norm": prune_norm,
                    "dataset_name": dataset_name,
                    "max_prompts": max_prompts,
                    "dataset_max_length": dataset_max_length,
                    "importance_metric": importance_metric,
                    "scoring_method": scoring_method,
                    "seed": seed,
                    "flap_metric": flap_metric,
                    "bo_eval_datasets": bo_eval_datasets,
                    "seq_len": seq_len,
                    "bo_objective": bo_objective,
                    "latency_batch_size": int(latency_batch_size),
                    "latency_seq_len": int(latency_seq_len),
                    "latency_num_runs": int(latency_num_runs),
                    "latency_num_warmup_runs": int(latency_num_warmup_runs),
                    "latency_input_prompt": latency_input_prompt,
                    "logging": logging_level,
                }
            )
        executor_kwargs = {
            "max_workers": len(args_payloads),
            "mp_context": mp.get_context("spawn"),
        }
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            futures = [
                executor.submit(_stage2_bo_trial_worker, payload)
                for payload in args_payloads
            ]
            return [future.result() for future in as_completed(futures)]

    X_train = []
    y_train = []
    best_score = float("inf")
    best_params = (1.0, 0.5)
    oom_trial_streak = 0
    max_consecutive_oom_trials = 2

    def _record_result(
        k_val: float, c_val: float, score: float, score_breakdown: Dict[str, float]
    ) -> None:
        nonlocal best_score, best_params, oom_trial_streak
        hit_oom = score >= BO_INVALID_SCORE_PENALTY
        if not math.isfinite(score):
            score = BO_INVALID_SCORE_PENALTY
        X_train.append([k_val, c_val])
        y_train.append(-score)
        if score < best_score:
            best_score = score
            best_params = (k_val, c_val)
        oom_trial_streak = oom_trial_streak + 1 if hit_oom else 0
        logging.info(
            "BO trial %d/%d k=%.4f c=%.4f objective=%s bo_score=%.6f (%s)",
            len(X_train),
            trials,
            k_val,
            c_val,
            bo_objective,
            score,
            ", ".join(f"{name}={value:.4f}" for name, value in score_breakdown.items()),
        )

    def _record(k_val: float, c_val: float) -> None:
        logging.info(
            "Starting stage-2 BO trial %d/%d with k=%.4f c=%.4f.",
            len(X_train) + 1,
            trials,
            k_val,
            c_val,
        )
        score, score_breakdown = _evaluate(k_val, c_val)
        _record_result(k_val, c_val, score, score_breakdown)

    def _record_batch(candidates_to_eval: Sequence[Tuple[float, float]]) -> None:
        remaining = max(0, trials - len(X_train))
        candidates_to_eval = list(candidates_to_eval)[:remaining]
        if not candidates_to_eval:
            return
        if len(candidates_to_eval) == 1:
            _record(candidates_to_eval[0][0], candidates_to_eval[0][1])
            return
        for k_val, c_val in candidates_to_eval:
            logging.info(
                "Queued stage-2 BO trial %d/%d with k=%.4f c=%.4f.",
                len(X_train) + 1,
                trials,
                k_val,
                c_val,
            )
        try:
            results = _evaluate_parallel(candidates_to_eval)
        except Exception as exc:
            logging.exception(
                "Parallel stage-2 BO trial evaluation failed; falling back to sequential evaluation: %s",
                exc,
            )
            results = []
            for k_val, c_val in candidates_to_eval:
                score, score_breakdown = _evaluate(k_val, c_val)
                results.append(
                    {
                        "k": float(k_val),
                        "c": float(c_val),
                        "score": float(score),
                        "score_breakdown": score_breakdown,
                    }
                )
        result_by_params = {
            (round(float(result["k"]), 12), round(float(result["c"]), 12)): result
            for result in results
        }
        for k_val, c_val in candidates_to_eval:
            result = result_by_params.get(
                (round(float(k_val), 12), round(float(c_val), 12))
            )
            if result is None:
                continue
            _record_result(
                float(k_val),
                float(c_val),
                float(result["score"]),
                result.get("score_breakdown", {}),
            )

    try:
        warmup = max(0, min(warmup, trials))
        while len(X_train) < warmup:
            batch_size = min(
                warmup - len(X_train),
                len(parallel_gpu_ids) if parallel_enabled else 1,
            )
            warmup_batch = [
                (
                    float(rng.uniform(bounds[0, 0], bounds[1, 0])),
                    float(rng.uniform(bounds[0, 1], bounds[1, 1])),
                )
                for _ in range(batch_size)
            ]
            _record_batch(warmup_batch)
            if oom_trial_streak >= max_consecutive_oom_trials:
                logging.warning(
                    "Stopping stage-2 BO early after %d consecutive CUDA OOM trials; "
                    "returning the best configuration found so far.",
                    oom_trial_streak,
                )
                break
        while len(X_train) < trials:
            if oom_trial_streak >= max_consecutive_oom_trials:
                logging.warning(
                    "Stopping stage-2 BO early after %d consecutive CUDA OOM trials; "
                    "returning the best configuration found so far.",
                    oom_trial_streak,
                )
                break
            X = torch.tensor(X_train, dtype=torch.double)
            Y = torch.tensor(y_train, dtype=torch.double).unsqueeze(-1)
            if X.shape[0] < 2 or torch.allclose(
                Y.std(), torch.tensor(0.0, dtype=Y.dtype)
            ):
                batch_size = min(
                    trials - len(X_train),
                    len(parallel_gpu_ids) if parallel_enabled else 1,
                )
                random_batch = [
                    (
                        float(rng.uniform(bounds[0, 0], bounds[1, 0])),
                        float(rng.uniform(bounds[0, 1], bounds[1, 1])),
                    )
                    for _ in range(batch_size)
                ]
                _record_batch(random_batch)
                if oom_trial_streak >= max_consecutive_oom_trials:
                    logging.warning(
                        "Stopping stage-2 BO early after %d consecutive CUDA OOM trials; "
                        "returning the best configuration found so far.",
                        oom_trial_streak,
                    )
                    break
                continue
            model = SingleTaskGP(
                X,
                Y,
                input_transform=Normalize(d=2),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with torch.enable_grad():
                fit_gpytorch_mll(mll)
            acq = LogExpectedImprovement(model=model, best_f=Y.max())
            candidate_set = torch.tensor(
                np.column_stack(
                    [
                        rng.uniform(bounds[0, 0], bounds[1, 0], size=256),
                        rng.uniform(bounds[0, 1], bounds[1, 1], size=256),
                    ]
                ),
                dtype=torch.double,
            )
            batch_size = min(
                trials - len(X_train),
                len(parallel_gpu_ids) if parallel_enabled else 1,
            )
            if batch_size <= 1:
                candidate, _ = optimize_acqf_discrete(
                    acq_function=acq,
                    q=1,
                    choices=candidate_set,
                )
                candidate_batch = [
                    (
                        float(candidate[0, 0].item()),
                        float(candidate[0, 1].item()),
                    )
                ]
            else:
                try:
                    with torch.no_grad():
                        acq_values = acq(candidate_set.unsqueeze(1)).reshape(-1)
                    topk = torch.topk(acq_values, k=batch_size).indices.tolist()
                    candidate_batch = [
                        (
                            float(candidate_set[index, 0].item()),
                            float(candidate_set[index, 1].item()),
                        )
                        for index in topk
                    ]
                except Exception:
                    candidate, _ = optimize_acqf_discrete(
                        acq_function=acq,
                        q=1,
                        choices=candidate_set,
                    )
                    candidate_batch = [
                        (
                            float(candidate[0, 0].item()),
                            float(candidate[0, 1].item()),
                        )
                    ]
            _record_batch(candidate_batch)
            if oom_trial_streak >= max_consecutive_oom_trials:
                logging.warning(
                    "Stopping stage-2 BO early after %d consecutive CUDA OOM trials; "
                    "returning the best configuration found so far.",
                    oom_trial_streak,
                )
                break

        best_prune_ratios = _build_monotone_prune_ratios(
            num_layers=num_layers,
            target_prune_ratio=target_prune_ratio,
            k=best_params[0],
            c=best_params[1],
            min_keep_ratio=min_keep_ratio,
            early_layer_min_keep_ratio=early_layer_min_keep_ratio,
            early_layer_count=early_layer_count,
            early_layer_max_prune_factor=early_layer_max_prune_factor,
        )
        best_keep_ratios = [
            min(1.0, max(min_keep_ratio, 1.0 - r)) for r in best_prune_ratios
        ]
        return best_keep_ratios, best_score, best_params
    finally:
        if can_offload_base_model:
            logging.info(
                "Restoring the base model to %s after stage-2 BO.", original_device
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            try:
                base_model.to(original_device)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                logging.warning(
                    "Could not restore the base model to %s after stage-2 BO due to CUDA OOM; "
                    "retrying once after additional cleanup.",
                    original_device,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                try:
                    base_model.to(original_device)
                except RuntimeError as retry_exc:
                    if "out of memory" not in str(retry_exc).lower():
                        raise
                    logging.warning(
                        "Base model restore to %s still OOM after cleanup; leaving it on CPU. "
                        "Stage-2 will continue on CPU unless moved later.",
                        original_device,
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _bayes_optimize_recovery_lora(
    *,
    pruned_model_dir: Path,
    tuned_root_dir: Path,
    arch: str,
    num_train_epochs: float,
    save_steps: int,
    logging_steps: int,
    num_samples: Optional[int],
    target_modules: Optional[List[str]],
    report_to: Optional[str],
    bo_eval_datasets: Sequence[Tuple[str, torch.Tensor]],
    seq_len: int,
    seed: int,
    trials: int,
    warmup: int,
) -> Tuple[object, Path, Dict[str, float]]:
    """Bayesian optimization over recovery LoRA params for perplexity minimization."""
    try:
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf_discrete
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except Exception as exc:
        raise RuntimeError(
            "Bayesian optimization requires botorch and gpytorch. "
            "Install them and retry (see requirements.txt)."
        ) from exc

    if trials <= 0:
        raise ValueError("--recovery_bayes_trials must be > 0.")
    if not bo_eval_datasets:
        raise ValueError(
            "Recovery BO requires prompt datasets for aggregate perplexity evaluation."
        )

    rng = np.random.default_rng(seed)
    tuned_root_dir.mkdir(parents=True, exist_ok=True)

    r_min, r_max = 4, 64
    alpha_min, alpha_max = 8, 256
    dropout_min, dropout_max = 0.0, 0.2
    lr_min, lr_max = 1e-5, 5e-4
    warmup_min, warmup_max = 0.0, 0.1

    def _snap_params(r_val, alpha_val, dropout_val, lr_val, warmup_val):
        r = int(round(r_val / 4.0) * 4)
        r = max(r_min, min(r_max, r))
        alpha = int(round(alpha_val / 8.0) * 8)
        alpha = max(alpha_min, min(alpha_max, alpha))
        dropout = float(min(dropout_max, max(dropout_min, dropout_val)))
        lr = float(min(lr_max, max(lr_min, lr_val)))
        warmup_ratio = float(min(warmup_max, max(warmup_min, warmup_val)))
        return r, alpha, dropout, lr, warmup_ratio

    evaluated: Dict[Tuple[int, int, float, float, float], float] = {}
    X_train = []
    y_train = []
    best_ppl = float("inf")
    best_payload = None

    def _record(candidate_vals: Sequence[float], trial_idx: int) -> None:
        nonlocal best_ppl, best_payload
        r, alpha, dropout, lr, warmup_ratio = _snap_params(*candidate_vals)
        key = (r, alpha, round(dropout, 4), round(lr, 8), round(warmup_ratio, 4))
        if key in evaluated:
            return
        trial_dir = tuned_root_dir / f"trial_{trial_idx:03d}"
        logging.info(
            "Recovery BO trial %d/%d r=%d alpha=%d dropout=%.3f lr=%.6f warmup=%.3f",
            len(X_train) + 1,
            trials,
            r,
            alpha,
            dropout,
            lr,
            warmup_ratio,
        )
        tuned_model = fine_tune_model(
            pruned_model_dir=str(pruned_model_dir),
            tuned_model_dir=str(trial_dir),
            arch=arch,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
            logging_steps=logging_steps,
            num_samples=num_samples,
            target_modules=target_modules,
            lora_rank=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            report_to=report_to,
            model=None,
            tokenizer=None,
        )
        device = str(next(tuned_model.parameters()).device)
        ppl, ppl_breakdown = _evaluate_bo_perplexity_score(
            tuned_model,
            bo_eval_datasets=bo_eval_datasets,
            seq_len=seq_len,
            device=device,
        )
        evaluated[key] = ppl
        X_train.append([r, alpha, dropout, lr, warmup_ratio])
        y_train.append(-ppl)
        if ppl < best_ppl:
            best_ppl = ppl
            best_payload = {
                "model": tuned_model,
                "dir": trial_dir,
                "r": r,
                "alpha": alpha,
                "dropout": dropout,
                "lr": lr,
                "warmup_ratio": warmup_ratio,
                "ppl": ppl,
                "ppl_breakdown": ppl_breakdown,
            }
        else:
            del tuned_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logging.info(
            "Recovery BO score=%.4f (%s)",
            ppl,
            ", ".join(f"{name}={value:.4f}" for name, value in ppl_breakdown.items()),
        )

    warmup = max(0, min(warmup, trials))
    for i in range(warmup):
        candidate = [
            rng.uniform(r_min, r_max),
            rng.uniform(alpha_min, alpha_max),
            rng.uniform(dropout_min, dropout_max),
            10 ** rng.uniform(math.log10(lr_min), math.log10(lr_max)),
            rng.uniform(warmup_min, warmup_max),
        ]
        _record(candidate, i)

    while len(X_train) < trials:
        X = torch.tensor(X_train, dtype=torch.double)
        Y = torch.tensor(y_train, dtype=torch.double).unsqueeze(-1)
        if X.shape[0] < 2 or torch.allclose(Y.std(), torch.tensor(0.0, dtype=Y.dtype)):
            candidate = [
                rng.uniform(r_min, r_max),
                rng.uniform(alpha_min, alpha_max),
                rng.uniform(dropout_min, dropout_max),
                10 ** rng.uniform(math.log10(lr_min), math.log10(lr_max)),
                rng.uniform(warmup_min, warmup_max),
            ]
            _record(candidate, len(X_train))
            continue
        model = SingleTaskGP(
            X,
            Y,
            input_transform=Normalize(d=5),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with torch.enable_grad():
            fit_gpytorch_mll(mll)
        acq = LogExpectedImprovement(model=model, best_f=Y.max())
        candidate_set = torch.tensor(
            np.column_stack(
                [
                    rng.uniform(r_min, r_max, size=128),
                    rng.uniform(alpha_min, alpha_max, size=128),
                    rng.uniform(dropout_min, dropout_max, size=128),
                    10 ** rng.uniform(math.log10(lr_min), math.log10(lr_max), size=128),
                    rng.uniform(warmup_min, warmup_max, size=128),
                ]
            ),
            dtype=torch.double,
        )
        candidate, _ = optimize_acqf_discrete(
            acq_function=acq,
            q=1,
            choices=candidate_set,
        )
        _record(candidate.squeeze(0).tolist(), len(X_train))

    if best_payload is None:
        raise RuntimeError("Recovery BO failed to produce a candidate.")
    return (
        best_payload["model"],
        best_payload["dir"],
        {
            "r": best_payload["r"],
            "alpha": best_payload["alpha"],
            "dropout": best_payload["dropout"],
            "learning_rate": best_payload["lr"],
            "warmup_ratio": best_payload["warmup_ratio"],
            "ppl": best_payload["ppl"],
            "ppl_breakdown": best_payload["ppl_breakdown"],
        },
    )


def _stage2_bo_trial_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Evaluate one stage-2 BO trial in a fresh process pinned to one CUDA device."""
    gpu_id = int(payload["gpu_id"])
    k_val = float(payload["k"])
    c_val = float(payload["c"])
    model_name = str(payload["model_name"])
    cache_dir = payload.get("cache_dir")
    attn_mask = payload["attn_mask"]
    mlp_mask = payload["mlp_mask"]
    bo_eval_datasets = payload["bo_eval_datasets"]
    bo_objective = str(payload.get("bo_objective", "ppl") or "ppl").lower()
    device = f"cuda:{gpu_id}"

    logging.basicConfig(
        level=getattr(
            logging, str(payload.get("logging", "INFO")).upper(), logging.INFO
        ),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for parallel stage-2 BO evaluation.")
    torch.cuda.set_device(gpu_id)
    logging.info(
        "Stage-2 BO worker on %s evaluating k=%.4f c=%.4f.",
        device,
        k_val,
        c_val,
    )
    prune_ratios = _build_monotone_prune_ratios(
        num_layers=int(payload["num_layers"]),
        target_prune_ratio=float(payload["target_prune_ratio"]),
        k=k_val,
        c=c_val,
        min_keep_ratio=float(payload["min_keep_ratio"]),
        early_layer_min_keep_ratio=payload.get("early_layer_min_keep_ratio"),
        early_layer_count=int(payload["early_layer_count"]),
        early_layer_max_prune_factor=float(payload["early_layer_max_prune_factor"]),
    )
    keep_ratios = [
        min(1.0, max(float(payload["min_keep_ratio"]), 1.0 - ratio))
        for ratio in prune_ratios
    ]
    model = None
    try:
        model = loadModel(model_name, cache_dir, device_map={"": device})
        maskModel(model, attn_mask, mlp_mask)
        _ = second_stage_optimize_heads(
            model,
            max_ratio=float(payload["max_ratio"]),
            layer_keep_ratios=keep_ratios,
            layer_component_masks=None,
            prune_heads=bool(payload["prune_heads"]),
            prune_mlp=bool(payload["prune_mlp"]),
            prune_embedding=bool(payload["prune_embedding"]),
            prune_norm=bool(payload["prune_norm"]),
            dataset_name=payload["dataset_name"],
            max_prompts=int(payload["max_prompts"]),
            dataset_max_length=int(payload["dataset_max_length"]),
            importance_metric=str(payload["importance_metric"]),
            scoring_method=str(payload["scoring_method"]),
            seed=int(payload["seed"]),
            flap_metric=str(payload["flap_metric"]),
        )
        if bo_objective == "latency":
            score, score_breakdown = _evaluate_stage2_bo_latency_score(
                model,
                load_tokenizer(model_name),
                device=device,
                input_prompt=str(
                    payload.get(
                        "latency_input_prompt", "The Leaning Tower of Pisa is known for"
                    )
                ),
                batch_size=int(payload.get("latency_batch_size", 1)),
                seq_len=int(payload.get("latency_seq_len", 64)),
                num_runs=int(payload.get("latency_num_runs", 5)),
                num_warmup_runs=int(payload.get("latency_num_warmup_runs", 2)),
            )
        else:
            score, score_breakdown = _evaluate_bo_perplexity_score(
                model,
                bo_eval_datasets=bo_eval_datasets,
                seq_len=int(payload["seq_len"]),
                device=device,
            )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        logging.warning("Parallel BO eval hit CUDA OOM; assigning penalty. %s", exc)
        score = BO_INVALID_SCORE_PENALTY
        score_breakdown = (
            {"latency_sec": float("inf")}
            if bo_objective == "latency"
            else {name: float("inf") for name, _ in bo_eval_datasets}
        )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    return {
        "gpu_id": gpu_id,
        "k": k_val,
        "c": c_val,
        "score": float(score),
        "score_breakdown": score_breakdown,
    }
