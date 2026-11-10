"""Export selected candidates and their replay metadata."""

from llm_pruning.prompt_sources import _get_second_stage_prompt_source
from llm_pruning.reporting import _slugify_filename
from llm_pruning.bayesian import load_tokenizer
from llm_pruning.checkpoints import _make_json_safe
import json
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM
from typing import Dict, Optional, Sequence
from llm_pruning.refinement.second_stage import (
    apply_saved_second_stage_pruning,
    second_stage_optimize_heads,
)
from llm_pruning.core.utilities import maskModel


def _load_concrete_model_for_export(model_name: str, cache_dir: Optional[str] = None):
    """Load a non-dispatched model with concrete tensors suitable for serialization."""
    logging.info("Loading a concrete export model for %s.", model_name)
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,
        cache_dir=cache_dir,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )


def _build_selected_stage2_model_dir(
    args,
    *,
    model_name: str,
    candidate_idx: int,
) -> Path:
    root = Path(args.selected_stage2_model_dir)
    model_slug = _slugify_filename(Path(str(model_name)).name)
    return root / f"{model_slug}_candidate_{candidate_idx}_stage2_selected"


def _save_selected_stage2_model(
    candidate: Dict[str, object],
    args,
    *,
    model_name: Optional[str] = None,
    candidate_idx: Optional[int] = None,
) -> Path:
    """Persist the selected stage-2-pruned model and tokenizer for external benchmarking."""
    model_name = model_name or args.model
    candidate_idx = int(candidate_idx or candidate.get("candidate_idx") or 1)
    output_dir = _build_selected_stage2_model_dir(
        args,
        model_name=model_name,
        candidate_idx=candidate_idx,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    export_model = _load_concrete_model_for_export(model_name, args.cache_dir)
    export_tokenizer = load_tokenizer(model_name)
    attn_mask = candidate.get("attn")
    mlp_mask = candidate.get("mlp")
    if attn_mask is None or mlp_mask is None:
        raise ValueError("Selected candidate is missing attn/mlp masks.")
    maskModel(export_model, attn_mask, mlp_mask)

    saved_stage2_layer_stats = candidate.get("stage2_layer_stats") or []
    if saved_stage2_layer_stats:
        _ = apply_saved_second_stage_pruning(export_model, saved_stage2_layer_stats)
    elif candidate.get("stage2_keep_ratio") is not None:
        _ = second_stage_optimize_heads(
            export_model,
            max_ratio=float(candidate["stage2_keep_ratio"]),
            layer_keep_ratios=candidate.get("stage2_layer_keep_ratios"),
            layer_component_masks=None,
            prune_heads=args.second_stage_heads,
            prune_mlp=args.second_stage_mlp,
            prune_embedding=args.second_stage_embedding,
            prune_norm=args.second_stage_norm,
            dataset_name=_get_second_stage_prompt_source(args),
            max_prompts=args.second_stage_num_prompts,
            dataset_max_length=args.second_stage_prompt_max_length,
            importance_metric=(
                candidate.get("stage2_importance_metric")
                or args.second_stage_importance_metric
            ),
            scoring_method=args.second_stage_method,
            seed=args.seed,
            flap_metric=args.second_stage_flap_metric,
        )

    export_model.save_pretrained(output_dir)
    export_tokenizer.save_pretrained(output_dir)
    metadata = {
        "model": model_name,
        "candidate_idx": candidate_idx,
        "stage2_keep_ratio": candidate.get("stage2_keep_ratio"),
        "stage2_allocation_method": candidate.get("stage2_allocation_method"),
        "stage2_allocation_params": candidate.get("stage2_allocation_params"),
        "stage2_layer_keep_ratios": candidate.get("stage2_layer_keep_ratios"),
        "stage2_layer_stats": candidate.get("stage2_layer_stats"),
        "stage2_importance_metric": candidate.get("stage2_importance_metric"),
        "bo_objective": candidate.get("bo_objective"),
        "bo_objective_score": candidate.get("bo_objective_score"),
        "bo_calibration_score": candidate.get("bo_calibration_score"),
        "bo_calibration_ppl": candidate.get("bo_calibration_ppl"),
        "bo_latency_score": candidate.get("bo_latency_score"),
        "final_latency": candidate.get("final_latency"),
        "final_active_params": candidate.get("final_active_params"),
    }
    with (output_dir / "candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_make_json_safe(metadata), handle, indent=2, sort_keys=True)
    logging.info(
        "Saved selected stage-2 model for candidate %d to %s",
        candidate_idx,
        output_dir.resolve(),
    )
    return output_dir


def _resolve_selected_stage2_export_candidate_idx(
    mask_candidates: Sequence[Dict[str, object]],
    checkpoint_state: Dict[str, object],
    *,
    best_idx: Optional[int] = None,
    last_idx: Optional[int] = None,
) -> Optional[int]:
    """Return the unique candidate index to export when one is well-defined."""
    selected_candidate_idx = checkpoint_state.get("selection", {}).get(
        "selected_candidate_idx"
    )
    if selected_candidate_idx is not None:
        selected_candidate_idx = int(selected_candidate_idx)
        if 1 <= selected_candidate_idx <= len(mask_candidates):
            return selected_candidate_idx

    if best_idx is not None and 1 <= int(best_idx) <= len(mask_candidates):
        return int(best_idx)

    best_ppl_idx = None
    best_ppl_value = float("inf")
    for idx, candidate in enumerate(mask_candidates, start=1):
        ppl = _get_candidate_bo_score(candidate)
        if ppl is None:
            continue
        try:
            ppl_value = float(ppl)
        except (TypeError, ValueError):
            continue
        if ppl_value < best_ppl_value:
            best_ppl_value = ppl_value
            best_ppl_idx = idx
    if best_ppl_idx is not None:
        logging.info(
            "No explicit selected candidate found for export; using candidate %d with the best saved BO calibration score (%.4f).",
            best_ppl_idx,
            best_ppl_value,
        )
        return best_ppl_idx

    if last_idx is not None and 1 <= int(last_idx) <= len(mask_candidates):
        logging.info(
            "No explicit selected candidate found for export; using the current final stage-2 candidate %d.",
            int(last_idx),
        )
        return int(last_idx)

    if len(mask_candidates) == 1:
        return 1

    return None


def _get_candidate_bo_score(candidate: Dict[str, object]) -> Optional[float]:
    """Return the stable BO objective score, falling back to legacy checkpoint fields."""
    value = candidate.get("bo_objective_score")
    if value is None:
        value = candidate.get("bo_calibration_score")
    if value is None:
        value = candidate.get("bo_calibration_ppl")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
