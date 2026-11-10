"""Reconstruct candidate models from coarse masks and saved width decisions."""

from llm_pruning.prompt_sources import _get_second_stage_prompt_source
import json
import logging
from pathlib import Path
from peft import PeftModel
from typing import Optional, Union
from llm_pruning.refinement.second_stage import (
    apply_saved_second_stage_pruning,
    second_stage_optimize_heads,
)
from llm_pruning.core.utilities import loadModel, maskModel


def _replay_candidate_on_fresh_model(
    candidate,
    args,
    *,
    model_name: Optional[str] = None,
    device_map="auto",
):
    """Load a clean base model, apply the saved stage-1 mask, then replay stage-2 pruning."""
    model_name = model_name or args.model
    attn_mask = candidate.get("attn")
    mlp_mask = candidate.get("mlp")
    if attn_mask is None or mlp_mask is None:
        raise ValueError("Saved candidate is missing attn/mlp masks.")

    model = loadModel(model_name, args.cache_dir, device_map=device_map)
    maskModel(model, attn_mask, mlp_mask)

    saved_stage2_layer_stats = candidate.get("stage2_layer_stats") or []
    best_stage2_keep_ratio = candidate.get("stage2_keep_ratio")
    best_stage2_layer_keep_ratios = candidate.get("stage2_layer_keep_ratios")
    best_stage2_importance_metric = (
        candidate.get("stage2_importance_metric") or args.second_stage_importance_metric
    )
    if saved_stage2_layer_stats:
        logging.info(
            "Replaying saved stage-2 pruning decisions from %d layer records.",
            len(saved_stage2_layer_stats),
        )
        _ = apply_saved_second_stage_pruning(model, saved_stage2_layer_stats)
    elif best_stage2_keep_ratio is not None:
        _ = second_stage_optimize_heads(
            model,
            max_ratio=best_stage2_keep_ratio,
            layer_keep_ratios=best_stage2_layer_keep_ratios,
            layer_component_masks=None,
            prune_heads=args.second_stage_heads,
            prune_mlp=args.second_stage_mlp,
            prune_embedding=args.second_stage_embedding,
            prune_norm=args.second_stage_norm,
            dataset_name=_get_second_stage_prompt_source(args),
            max_prompts=args.second_stage_num_prompts,
            dataset_max_length=args.second_stage_prompt_max_length,
            importance_metric=best_stage2_importance_metric,
            scoring_method=args.second_stage_method,
            seed=args.seed,
            flap_metric=args.second_stage_flap_metric,
        )
    return model


def _load_replayed_recovery_model(
    model_dir: Union[str, Path],
    base_model_name: str,
    cache_dir: Optional[str],
    *,
    device_map="auto",
):
    """Load base model, replay pruning, and apply a saved recovery LoRA adapter."""
    model_dir = Path(model_dir)
    replay_candidate_path = model_dir.parent / "replay_candidate.json"
    adapter_config_path = model_dir / "adapter_config.json"
    if not replay_candidate_path.exists() or not adapter_config_path.exists():
        return loadModel(str(model_dir), cache_dir, device_map=device_map)

    with open(replay_candidate_path, "r", encoding="utf-8") as handle:
        candidate = json.load(handle)

    model = loadModel(base_model_name, cache_dir, device_map=device_map)
    attn_mask = candidate.get("attn")
    mlp_mask = candidate.get("mlp")
    if attn_mask is None or mlp_mask is None:
        raise ValueError(
            f"Replay candidate is missing attn/mlp masks: {replay_candidate_path}"
        )
    maskModel(model, attn_mask, mlp_mask)
    saved_stage2_layer_stats = candidate.get("stage2_layer_stats") or []
    if saved_stage2_layer_stats:
        apply_saved_second_stage_pruning(model, saved_stage2_layer_stats)
    model = PeftModel.from_pretrained(model, str(model_dir))
    return model.merge_and_unload()
