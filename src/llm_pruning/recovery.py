"""Launch optional distributed LoRA recovery."""

from llm_pruning.bayesian import _parse_downstream_gpu_ids
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _run_recovery_finetune_ddp(
    *,
    pruned_model_dir: Path,
    tuned_model_dir: Path,
    candidate: Dict[str, object],
    args,
    target_modules: Optional[Sequence[str]],
    gpu_ids: Sequence[int],
) -> None:
    """Run recovery LoRA fine-tuning with torchrun/DDP on the requested GPUs."""
    if len(gpu_ids) < 2:
        raise ValueError("DDP recovery fine-tuning requires at least two GPU ids.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    replay_candidate_path = tuned_model_dir.parent / "replay_candidate.json"
    replay_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(replay_candidate_path, "w", encoding="utf-8") as handle:
        json.dump(
            candidate,
            handle,
            default=lambda value: (
                value.tolist() if hasattr(value, "tolist") else str(value)
            ),
        )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(len(gpu_ids)),
        "--module",
        "llm_pruning.fine_tune",
        "--pruned_model_dir",
        str(pruned_model_dir),
        "--tuned_model_dir",
        str(tuned_model_dir),
        "--base_model_name",
        args.model,
        "--replay_candidate_json",
        str(replay_candidate_path),
        "--arch",
        args.recovery_arch,
        "--num_train_epochs",
        str(args.recovery_epochs),
        "--save_steps",
        str(args.recovery_save_steps),
        "--logging_steps",
        str(args.recovery_logging_steps),
        "--batch_size",
        "1",
        "--warmup_ratio",
        str(args.recovery_warmup_ratio),
    ]
    if args.cache_dir is not None:
        command.extend(["--cache_dir", str(args.cache_dir)])
    if args.recovery_num_samples is not None:
        command.extend(["--num_samples", str(args.recovery_num_samples)])
    if target_modules:
        command.extend(["--target_modules", ",".join(target_modules)])
    if args.recovery_report_to is not None:
        command.extend(["--report_to", str(args.recovery_report_to)])

    logging.info(
        "Launching recovery fine-tuning with torchrun on GPUs %s.",
        env["CUDA_VISIBLE_DEVICES"],
    )
    logging.info("Recovery fine-tuning command: %s", " ".join(command))
    subprocess.run(command, check=True, env=env)


def _resolve_recovery_finetune_gpu_ids(args) -> List[int]:
    """Pick GPU ids for recovery fine-tuning from explicit args or existing GPU config."""
    candidates = [
        getattr(args, "recovery_finetune_gpus", ""),
        getattr(args, "downstream_parallel_gpus", ""),
        getattr(args, "second_stage_bayes_candidate_parallel_gpus", ""),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "auto",
    ]
    first_resolved: List[int] = []
    for candidate in candidates:
        resolved = _parse_downstream_gpu_ids(candidate)
        if resolved and not first_resolved:
            first_resolved = resolved
        if len(resolved) > 1:
            return resolved
    return first_resolved
