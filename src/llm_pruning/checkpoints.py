"""Checkpoint signatures, persistence, and resume handling."""

from llm_pruning.reporting import _slugify_filename
import copy
import json
import logging
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Sequence

_LEGACY_STAGE2_SIGNATURE_DEFAULTS = {
    "second_stage_allocation_method": "auto",
    "second_stage_allocation_beta": 0.5,
    "second_stage_allocation_alpha": None,
}

_STAGE1_CHECKPOINT_KEYS = (
    "model",
    "method",
    "sparsity",
    "sparsity_alpha",
    "stage2_only",
    "proposed_first_objective",
    "proposed_second_objective",
    "proposed_pop_size",
    "proposed_n_gen",
    "stage1_importance_penalty",
    "stage1_importance_penalty_weight",
    "seed",
)

_STAGE2_CHECKPOINT_KEYS = (
    "second_stage_heads",
    "second_stage_mlp",
    "second_stage_embedding",
    "second_stage_norm",
    "second_stage_method",
    "second_stage_allocation_method",
    "second_stage_allocation_beta",
    "second_stage_allocation_alpha",
    "second_stage_bayes",
    "second_stage_bayes_objective",
    "second_stage_prompt_set",
    "proposed_pareto_selection",
)


def _build_stage1_checkpoint_signature(args) -> Dict[str, object]:
    return {
        "model": args.model,
        "method": args.method,
        "sparsity": float(args.sparsity),
        "sparsity_alpha": float(args.sparsity_alpha),
        "stage2_only": bool(args.stage2_only),
        "proposed_first_objective": args.proposed_first_objective,
        "proposed_second_objective": args.proposed_second_objective,
        "proposed_pop_size": int(args.proposed_pop_size),
        "proposed_n_gen": int(args.proposed_n_gen),
        "stage1_importance_penalty": bool(args.stage1_importance_penalty),
        "stage1_importance_penalty_weight": float(
            args.stage1_importance_penalty_weight
        ),
        "seed": int(args.seed),
    }


def _build_stage2_checkpoint_signature(args) -> Dict[str, object]:
    return {
        "second_stage_heads": bool(args.second_stage_heads),
        "second_stage_mlp": bool(args.second_stage_mlp),
        "second_stage_embedding": bool(args.second_stage_embedding),
        "second_stage_norm": bool(args.second_stage_norm),
        "second_stage_method": args.second_stage_method,
        "second_stage_allocation_method": args.second_stage_allocation_method,
        "second_stage_allocation_beta": float(args.second_stage_allocation_beta),
        "second_stage_allocation_alpha": (
            float(args.second_stage_allocation_alpha)
            if args.second_stage_allocation_alpha is not None
            else None
        ),
        "second_stage_bayes": bool(args.second_stage_bayes),
        "second_stage_bayes_objective": args.second_stage_bayes_objective,
        "second_stage_prompt_set": args.second_stage_prompt_set,
        "proposed_pareto_selection": args.proposed_pareto_selection,
    }


def _build_pipeline_checkpoint_signature(args) -> Dict[str, object]:
    """Capture the run configuration that must match for safe resume."""
    return {
        "stage1": _build_stage1_checkpoint_signature(args),
        "stage2": _build_stage2_checkpoint_signature(args),
    }


def _extract_legacy_stage_signature(
    signature: Dict[str, object],
    *,
    keys: Sequence[str],
) -> Dict[str, object]:
    normalized = copy.deepcopy(signature)
    for key, value in _LEGACY_STAGE2_SIGNATURE_DEFAULTS.items():
        normalized.setdefault(key, value)
    return {key: normalized.get(key) for key in keys}


def _signature_diff_keys(
    saved_signature: Dict[str, object], expected_signature: Dict[str, object]
) -> List[str]:
    return sorted(
        key
        for key in set(saved_signature.keys()).union(expected_signature.keys())
        if saved_signature.get(key) != expected_signature.get(key)
    )


def _build_pipeline_checkpoint_path(args) -> Path:
    """Return the stable JSON checkpoint path used for resume/eval-only flows."""
    if not bool(getattr(args, "pipeline_checkpoint", True)):
        return Path(
            args.pipeline_checkpoint_path or "pipeline_checkpoint_disabled.json"
        )
    if getattr(args, "pipeline_checkpoint_path", None):
        path = Path(args.pipeline_checkpoint_path)
    else:
        path = Path("checkpoints") / _build_run_artifact_filename(
            args,
            suffix="pipeline_state",
            extension=".json",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_run_artifact_filename(args, *, suffix: str, extension: str) -> str:
    """Build a readable filename that is unique per model/method/sparsity."""
    model_slug = _slugify_filename(args.model)
    method_slug = _slugify_filename(args.method)
    sparsity_slug = f"{float(args.sparsity):.4f}".replace(".", "p")
    return f"{model_slug}_{method_slug}_sparsity_{sparsity_slug}_{suffix}{extension}"


def _make_json_safe(value):
    """Convert numpy/Path containers into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_pipeline_checkpoint(
    path: Path, signature: Dict[str, object]
) -> Optional[Dict[str, object]]:
    """Load a matching checkpoint if one exists for the current run configuration."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logging.warning("Failed to load pipeline checkpoint %s: %s", path, exc)
        return None

    if payload.get("format_version") != 1:
        logging.warning("Ignoring incompatible pipeline checkpoint at %s.", path)
        return None
    expected_stage1_signature = signature.get("stage1") or {}
    expected_stage2_signature = signature.get("stage2") or {}
    if not isinstance(expected_stage1_signature, dict) or not isinstance(
        expected_stage2_signature, dict
    ):
        raise ValueError(
            "Pipeline checkpoint signature must contain 'stage1' and 'stage2' dicts."
        )

    saved_stage1_signature = payload.get("stage1_signature")
    saved_stage2_signature = payload.get("stage2_signature")

    if not isinstance(saved_stage1_signature, dict) or not isinstance(
        saved_stage2_signature, dict
    ):
        saved_signature = payload.get("run_signature") or {}
        if not isinstance(saved_signature, dict):
            logging.warning("Ignoring malformed pipeline checkpoint at %s.", path)
            return None
        saved_stage1_signature = _extract_legacy_stage_signature(
            saved_signature,
            keys=_STAGE1_CHECKPOINT_KEYS,
        )
        saved_stage2_signature = _extract_legacy_stage_signature(
            saved_signature,
            keys=_STAGE2_CHECKPOINT_KEYS,
        )

    if not isinstance(saved_stage1_signature, dict) or not isinstance(
        saved_stage2_signature, dict
    ):
        logging.warning("Ignoring malformed pipeline checkpoint at %s.", path)
        return None

    if saved_stage1_signature != expected_stage1_signature:
        differing_keys = _signature_diff_keys(
            saved_stage1_signature, expected_stage1_signature
        )
        logging.warning(
            "Ignoring pipeline checkpoint at %s because its stage-1 signature does not match the current arguments. "
            "Differing keys: %s",
            path,
            ", ".join(differing_keys) if differing_keys else "unknown",
        )
        return None

    stage2_signature_matches = saved_stage2_signature == expected_stage2_signature
    if not stage2_signature_matches:
        differing_keys = _signature_diff_keys(
            saved_stage2_signature, expected_stage2_signature
        )
        logging.info(
            "Stage-2 configuration differs from the checkpoint at %s. Reusing stage-1 artifacts and clearing saved "
            "stage-2 progress. Differing keys: %s",
            path,
            ", ".join(differing_keys) if differing_keys else "unknown",
        )

    payload["stage1_signature"] = copy.deepcopy(saved_stage1_signature)
    payload["stage2_signature"] = copy.deepcopy(saved_stage2_signature)
    payload["run_signature"] = {
        "stage1": copy.deepcopy(saved_stage1_signature),
        "stage2": copy.deepcopy(saved_stage2_signature),
    }
    payload["_stage2_signature_matches"] = bool(stage2_signature_matches)
    return payload


def _new_pipeline_checkpoint(
    path: Path, signature: Dict[str, object]
) -> Dict[str, object]:
    now = datetime.now().isoformat()
    stage1_signature = copy.deepcopy(signature.get("stage1") or {})
    stage2_signature = copy.deepcopy(signature.get("stage2") or {})
    return {
        "format_version": 1,
        "created_at": now,
        "updated_at": now,
        "checkpoint_path": str(path.resolve()),
        "run_signature": {
            "stage1": stage1_signature,
            "stage2": stage2_signature,
        },
        "stage1_signature": stage1_signature,
        "stage2_signature": stage2_signature,
        "paths": {
            "proposed_save_pareto": None,
            "proposed_pareto_plot": None,
        },
        "stage1": {
            "completed": False,
            "importance_scores": None,
        },
        "stage2": {
            "completed": False,
            "completed_candidate_indices": [],
            "stagewise_plot_records": [],
        },
        "selection": {
            "completed": False,
            "selected_candidate_idx": None,
            "best_idx": None,
            "last_idx": None,
        },
        "mask_candidates": [],
    }


def _save_pipeline_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    payload = copy.deepcopy(payload)
    payload["updated_at"] = datetime.now().isoformat()
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_make_json_safe(payload), handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _candidate_stage2_completed(candidate: Dict[str, object]) -> bool:
    return bool(candidate.get("_stage2_completed"))


def _completed_stage2_candidate_indices(
    candidates: Sequence[Dict[str, object]],
) -> List[int]:
    return [
        idx
        for idx, candidate in enumerate(candidates, start=1)
        if _candidate_stage2_completed(candidate)
    ]


def _persist_pipeline_checkpoint(
    checkpoint_state: Dict[str, object],
    checkpoint_path: Path,
    *,
    args,
    mask_candidates: Sequence[Dict[str, object]],
    stage1_completed: Optional[bool] = None,
    stage1_importance_scores: Optional[Dict[str, object]] = None,
    stage2_completed: Optional[bool] = None,
    stagewise_plot_records: Optional[Sequence[Dict[str, object]]] = None,
    selected_candidate_idx: Optional[int] = None,
    best_idx: Optional[int] = None,
    last_idx: Optional[int] = None,
) -> None:
    if not bool(getattr(args, "pipeline_checkpoint", True)):
        return
    stage1_signature = _build_stage1_checkpoint_signature(args)
    stage2_signature = _build_stage2_checkpoint_signature(args)
    checkpoint_state["run_signature"] = {
        "stage1": copy.deepcopy(stage1_signature),
        "stage2": copy.deepcopy(stage2_signature),
    }
    checkpoint_state["stage1_signature"] = copy.deepcopy(stage1_signature)
    checkpoint_state["stage2_signature"] = copy.deepcopy(stage2_signature)
    checkpoint_state["paths"]["proposed_save_pareto"] = args.proposed_save_pareto
    checkpoint_state["paths"]["proposed_pareto_plot"] = args.proposed_pareto_plot
    checkpoint_state["mask_candidates"] = copy.deepcopy(list(mask_candidates))
    if stage1_completed is not None:
        checkpoint_state["stage1"]["completed"] = bool(stage1_completed)
    if stage1_importance_scores is not None:
        checkpoint_state["stage1"]["importance_scores"] = copy.deepcopy(
            stage1_importance_scores
        )
    completed_indices = _completed_stage2_candidate_indices(mask_candidates)
    checkpoint_state["stage2"]["completed_candidate_indices"] = completed_indices
    if stage2_completed is None:
        stage2_completed = bool(mask_candidates) and len(completed_indices) == len(
            mask_candidates
        )
    checkpoint_state["stage2"]["completed"] = bool(stage2_completed)
    if stagewise_plot_records is not None:
        checkpoint_state["stage2"]["stagewise_plot_records"] = copy.deepcopy(
            list(stagewise_plot_records)
        )
    if selected_candidate_idx is not None:
        checkpoint_state["selection"]["selected_candidate_idx"] = int(
            selected_candidate_idx
        )
        checkpoint_state["selection"]["completed"] = True
    if best_idx is not None:
        checkpoint_state["selection"]["best_idx"] = int(best_idx)
    if last_idx is not None:
        checkpoint_state["selection"]["last_idx"] = int(last_idx)
    _save_pipeline_checkpoint(checkpoint_path, checkpoint_state)
