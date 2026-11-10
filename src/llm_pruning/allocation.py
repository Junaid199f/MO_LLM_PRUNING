"""Layer-wise sparsity allocation and comparison strategies."""

from llm_pruning.budget import _allocate_prune_budget, _build_monotone_prune_ratios

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Sequence, Tuple
from llm_pruning.refinement import second_stage as stage2_impl


def _normalize_device_map_location(device) -> Optional[str]:
    """Normalize Accelerate device-map entries into torch.device-compatible strings."""
    if device is None:
        return None
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, int):
        return f"cuda:{device}"
    device_str = str(device).strip()
    if not device_str:
        return None
    if device_str.isdigit():
        return f"cuda:{device_str}"
    return device_str


def _resolve_model_compute_device(model) -> torch.device:
    """Best-effort primary compute device for models loaded with accelerate/device maps."""
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        mapped_devices = [
            _normalize_device_map_location(device) for device in device_map.values()
        ]
        compute_devices = sorted(
            {
                device
                for device in mapped_devices
                if device is not None and device not in {"cpu", "disk"}
            }
        )
        if compute_devices:
            return torch.device(compute_devices[0])
    return next(model.parameters()).device


def _build_stage2_layer_prune_caps(
    num_layers: int,
    target_prune_ratio: float,
    *,
    min_keep_ratio: float = 0.01,
    early_layer_min_keep_ratio: Optional[float] = None,
    early_layer_count: int = 2,
    early_layer_max_prune_factor: float = 0.75,
) -> List[float]:
    """Return the per-layer maximum prune ratios allowed by stage-2 safeguards."""
    if num_layers <= 0:
        return []
    target_prune_ratio = min(1.0, max(0.0, float(target_prune_ratio)))
    min_keep_ratio = min(0.999999, max(1e-6, float(min_keep_ratio)))
    early_layer_count = max(0, min(num_layers, int(early_layer_count)))
    early_layer_max_prune_factor = min(
        1.0, max(0.0, float(early_layer_max_prune_factor))
    )
    layer_caps = [1.0 - min_keep_ratio] * num_layers
    for layer_idx in range(early_layer_count):
        early_cap = min(
            layer_caps[layer_idx], target_prune_ratio * early_layer_max_prune_factor
        )
        if early_layer_min_keep_ratio is not None:
            constrained_early_keep = min(
                0.999999,
                max(min_keep_ratio, float(early_layer_min_keep_ratio)),
            )
            early_cap = min(early_cap, 1.0 - constrained_early_keep)
        layer_caps[layer_idx] = max(0.0, early_cap)
    return layer_caps


def _build_stage2_keep_ratios_from_prune_weights(
    weights: Sequence[float],
    *,
    num_layers: int,
    target_prune_ratio: float,
    min_keep_ratio: float = 0.01,
    early_layer_min_keep_ratio: Optional[float] = None,
    early_layer_count: int = 2,
    early_layer_max_prune_factor: float = 0.75,
) -> List[float]:
    """Allocate the exact stage-2 prune budget by weights and convert it to keep ratios."""
    if num_layers <= 0:
        return []
    if len(weights) != num_layers:
        raise ValueError(
            f"Expected {num_layers} stage-2 allocation weights, got {len(weights)}."
        )
    target_prune_ratio = min(1.0, max(0.0, float(target_prune_ratio)))
    min_keep_ratio = min(0.999999, max(1e-6, float(min_keep_ratio)))
    layer_caps = _build_stage2_layer_prune_caps(
        num_layers,
        target_prune_ratio,
        min_keep_ratio=min_keep_ratio,
        early_layer_min_keep_ratio=early_layer_min_keep_ratio,
        early_layer_count=early_layer_count,
        early_layer_max_prune_factor=early_layer_max_prune_factor,
    )
    prune_ratios = _allocate_prune_budget(
        weights,
        num_layers * target_prune_ratio,
        max_prune_by_layer=layer_caps,
    )
    return [
        min(1.0, max(min_keep_ratio, 1.0 - float(prune_ratio)))
        for prune_ratio in prune_ratios
    ]


def _build_layer_keep_ratios_from_beta(
    base_ratio: float,
    beta: float,
    num_layers: int,
) -> List[float]:
    """Redistribute keep ratios around a global keep ratio while preserving the average."""
    if num_layers <= 0:
        return []
    beta = min(1.0, max(0.0, float(beta)))
    min_keep = 1e-6
    base_keep = min(1.0, max(min_keep, float(base_ratio)))
    if num_layers == 1:
        return [round(base_keep, 6)]

    max_delta = min(base_keep - min_keep, 1.0 - base_keep)
    if max_delta <= 0.0:
        return [round(base_keep, 6)] * num_layers

    values = []
    for layer_idx in range(num_layers):
        normalized_pos = 1.0 - (2.0 * layer_idx / (num_layers - 1))
        ratio = base_keep + beta * max_delta * normalized_pos
        ratio = min(1.0, max(min_keep, float(ratio)))
        values.append(round(ratio, 6))
    return values


def _iter_stage2_layers(model) -> Sequence[object]:
    return getattr(getattr(model, "model", None), "layers", []) or []


def _aggregate_layer_signal(
    head_scores: Dict[int, object],
    mlp_scores: Dict[int, object],
    *,
    num_layers: int,
    default: float = 1.0,
) -> List[float]:
    """Aggregate per-head/per-neuron scores into one layer-level importance value."""
    aggregated: List[float] = []
    for layer_idx in range(num_layers):
        values: List[float] = []
        head_val = head_scores.get(layer_idx)
        if head_val is not None:
            arr = np.asarray(head_val, dtype=np.float64)
            if arr.size:
                values.append(float(np.mean(np.abs(arr))))
        mlp_val = mlp_scores.get(layer_idx)
        if mlp_val is not None:
            arr = np.asarray(mlp_val, dtype=np.float64)
            if arr.size:
                values.append(float(np.mean(np.abs(arr))))
        aggregated.append(float(np.mean(values)) if values else float(default))
    return aggregated


def _iter_layer_prunable_linear_modules(layer) -> List[object]:
    """Return the linear modules that define a transformer block's sparse shape."""
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    modules = [
        getattr(attn, "q_proj", None),
        getattr(attn, "k_proj", None),
        getattr(attn, "v_proj", None),
        getattr(attn, "o_proj", None),
        getattr(mlp, "gate_proj", None),
        getattr(mlp, "up_proj", None),
        getattr(mlp, "down_proj", None),
        getattr(mlp, "gate_up_proj", None),
        getattr(mlp, "fc1", None),
        getattr(mlp, "fc2", None),
    ]
    return [
        module
        for module in modules
        if module is not None and getattr(module, "weight", None) is not None
    ]


def _layer_shape_stats(layer) -> Tuple[float, float, float]:
    """Return ER/ERK aggregate density statistics for a transformer block."""
    param_count = 0.0
    er_density_sum = 0.0
    erk_density_sum = 0.0
    for module in _iter_layer_prunable_linear_modules(layer):
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        out_dim, in_dim = weight.shape
        prod = float(max(1, int(in_dim) * int(out_dim)))
        param_count += prod
        er_density_sum += prod * (float(in_dim + out_dim) / prod)
        erk_density_sum += prod * (float(in_dim + out_dim) / prod)
    if param_count <= 0.0:
        return 1.0, 1.0, 1.0
    return param_count, er_density_sum / param_count, erk_density_sum / param_count


def _compute_er_style_keep_scores(model, *, variant: str) -> List[float]:
    """Compute ER/ERK layer density scores from prunable linear shapes."""
    scores: List[float] = []
    layers = list(_iter_stage2_layers(model))
    for layer in layers:
        _param_count, er_density, erk_density = _layer_shape_stats(layer)
        if variant == "er_plus":
            scores.append(float(erk_density))
        else:
            scores.append(float(er_density))
    return scores or [1.0] * len(layers)


@torch.no_grad()
def _compute_owl_lod_scores(
    model,
    *,
    dataset_name,
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    max_length: int,
    outlier_multiplier: float = 7.0,
) -> List[float]:
    """Compute OWL Layerwise Outlier Distribution from Wanda-style salience."""
    prompts = stage2_impl._load_prompts(  # type: ignore[attr-defined]
        dataset_name,
        dataset_split,
        max_prompts,
        random_subset,
    )
    layers = list(_iter_stage2_layers(model))
    if not prompts or not layers:
        return [1.0] * len(layers)

    tokenizer = stage2_impl._resolve_tokenizer(model)  # type: ignore[attr-defined]
    device = _resolve_model_compute_device(model)
    module_inputs: Dict[object, Dict[str, object]] = {}
    module_to_layer: Dict[object, int] = {}
    handles = []

    for layer_idx, layer in enumerate(layers):
        for module in _iter_layer_prunable_linear_modules(layer):
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            module_inputs[module] = {
                "sumsq": torch.zeros(
                    int(weight.shape[1]), device=weight.device, dtype=torch.float32
                ),
                "count": 0,
            }
            module_to_layer[module] = layer_idx

            def _hook(_module, inputs, _output, tracked=module):
                if not inputs:
                    return
                activation = inputs[0]
                if activation is None:
                    return
                activation = activation.detach().to(
                    device=getattr(tracked, "weight").device, dtype=torch.float32
                )
                if activation.ndim == 1:
                    activation = activation.unsqueeze(0)
                activation = activation.reshape(-1, activation.shape[-1])
                if activation.shape[-1] != module_inputs[tracked]["sumsq"].numel():
                    return
                module_inputs[tracked]["sumsq"] += activation.pow(2).sum(dim=0)
                module_inputs[tracked]["count"] += int(activation.shape[0])

            handles.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        for batch in stage2_impl._iter_tokenized_prompts(  # type: ignore[attr-defined]
            tokenizer,
            prompts,
            device=device,
            max_length=max(8, int(max_length)),
        ):
            _ = model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    per_layer: List[List[float]] = [[] for _ in layers]
    multiplier = max(1.0, float(outlier_multiplier))
    for module, stats in module_inputs.items():
        weight = getattr(module, "weight", None)
        count = max(1, int(stats["count"]))
        if weight is None or weight.numel() == 0:
            continue
        input_scale = torch.sqrt(stats["sumsq"] / count).to(
            device=weight.device, dtype=torch.float32
        )
        salience = weight.detach().float().abs() * input_scale.reshape(1, -1)
        mean_salience = float(salience.mean().item())
        if mean_salience <= 0.0 or not math.isfinite(mean_salience):
            continue
        lod = float((salience > (multiplier * mean_salience)).float().mean().item())
        per_layer[module_to_layer[module]].append(lod)

    return [float(np.mean(values)) if values else 1.0 for values in per_layer]


def _layer_sensitivity_from_component_scores(
    head_scores: Dict[int, object],
    mlp_scores: Dict[int, object],
    *,
    num_layers: int,
    target_prune_ratio: float,
    default: float = 1.0,
) -> List[float]:
    """Estimate layer sensitivity from salience of units selected by local pruning."""
    sensitivities: List[float] = []
    prune_ratio = min(1.0, max(0.0, float(target_prune_ratio)))
    for layer_idx in range(num_layers):
        selected_values: List[float] = []
        all_values: List[float] = []
        for score_map in (head_scores, mlp_scores):
            raw_values = score_map.get(layer_idx)
            if raw_values is None:
                continue
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            values = np.abs(values)
            all_values.extend(values.tolist())
            prune_count = int(round(values.size * prune_ratio))
            prune_count = max(0, min(prune_count, values.size - 1))
            if prune_count > 0:
                selected_values.extend(
                    np.partition(values, prune_count - 1)[:prune_count].tolist()
                )
        if selected_values:
            sensitivities.append(float(np.mean(selected_values)))
        elif all_values:
            sensitivities.append(float(np.mean(all_values)))
        else:
            sensitivities.append(float(default))
    return sensitivities


def _normalize_positive_scores(
    values: Sequence[float], *, eps: float = 1e-8
) -> List[float]:
    normalized = [max(eps, float(value)) for value in values]
    max_value = max(normalized, default=1.0)
    if max_value <= eps:
        return [1.0] * len(normalized)
    return [value / max_value for value in normalized]


def _build_stage2_keep_ratios_for_method(
    model,
    *,
    allocation_method: str,
    target_keep_ratio: float,
    target_prune_ratio: float,
    dataset_name,
    dataset_split: str,
    max_prompts: int,
    dataset_max_length: int,
    importance_metric: str,
    scoring_method: str,
    min_keep_ratio: float,
    early_layer_min_keep_ratio: Optional[float],
    early_layer_count: int,
    early_layer_max_prune_factor: float,
    beta_value: float,
) -> Tuple[List[float], Dict[str, object]]:
    """Build stage-2 per-layer keep ratios for non-BO allocation baselines."""
    num_layers = len(_iter_stage2_layers(model))
    if num_layers <= 0:
        return [], {}

    if allocation_method == "uniform":
        return _build_stage2_keep_ratios_from_prune_weights(
            [1.0] * num_layers,
            num_layers=num_layers,
            target_prune_ratio=target_prune_ratio,
            min_keep_ratio=min_keep_ratio,
            early_layer_min_keep_ratio=early_layer_min_keep_ratio,
            early_layer_count=early_layer_count,
            early_layer_max_prune_factor=early_layer_max_prune_factor,
        ), {"target_keep_ratio": float(target_keep_ratio)}

    if allocation_method == "beta":
        beta_keep_ratios = _build_layer_keep_ratios_from_beta(
            target_keep_ratio,
            beta_value,
            num_layers,
        )
        beta_prune_weights = [
            max(0.0, 1.0 - float(ratio)) for ratio in beta_keep_ratios
        ]
        return _build_stage2_keep_ratios_from_prune_weights(
            beta_prune_weights,
            num_layers=num_layers,
            target_prune_ratio=target_prune_ratio,
            min_keep_ratio=min_keep_ratio,
            early_layer_min_keep_ratio=early_layer_min_keep_ratio,
            early_layer_count=early_layer_count,
            early_layer_max_prune_factor=early_layer_max_prune_factor,
        ), {
            "beta": float(beta_value),
            "raw_beta_keep_ratios": [float(ratio) for ratio in beta_keep_ratios],
        }

    if allocation_method in {"er", "er_plus"}:
        keep_scores = _compute_er_style_keep_scores(model, variant=allocation_method)
    elif allocation_method == "owl":
        lod_scores = _compute_owl_lod_scores(
            model,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_prompts=max_prompts,
            random_subset=False,
            max_length=dataset_max_length,
        )
        prune_weights = [
            max(1e-8, 1.0 - min(1.0, max(0.0, float(lod)))) for lod in lod_scores
        ]
        keep_ratios = _build_stage2_keep_ratios_from_prune_weights(
            prune_weights,
            num_layers=num_layers,
            target_prune_ratio=target_prune_ratio,
            min_keep_ratio=min_keep_ratio,
            early_layer_min_keep_ratio=early_layer_min_keep_ratio,
            early_layer_count=early_layer_count,
            early_layer_max_prune_factor=early_layer_max_prune_factor,
        )
        return keep_ratios, {"owl_lod_scores": [float(score) for score in lod_scores]}
    elif allocation_method in {"lsa", "dsa", "dlp"}:
        tokenizer = stage2_impl._resolve_tokenizer(model)  # type: ignore[attr-defined]
        arch = stage2_impl._infer_architecture(model)  # type: ignore[attr-defined]
        head_scores, mlp_scores = stage2_impl._collect_stage2_scores(  # type: ignore[attr-defined]
            model,
            tokenizer,
            arch,
            scoring_method=scoring_method,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_prompts=max_prompts,
            random_subset=False,
            prompt_max_length=dataset_max_length,
            importance_metric=importance_metric,
            collect_heads=True,
            collect_mlp=True,
        )
        importance_scores = _aggregate_layer_signal(
            head_scores,
            mlp_scores,
            num_layers=num_layers,
        )
        if allocation_method == "lsa":
            keep_scores = _layer_sensitivity_from_component_scores(
                head_scores,
                mlp_scores,
                num_layers=num_layers,
                target_prune_ratio=target_prune_ratio,
            )
        elif allocation_method == "dsa":
            keep_scores = importance_scores
        else:
            owl_scores = _compute_owl_lod_scores(
                model,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                max_prompts=max_prompts,
                random_subset=False,
                max_length=dataset_max_length,
            )
            owl_norm = _normalize_positive_scores(owl_scores)
            imp_norm = _normalize_positive_scores(importance_scores)
            sensitivity_scores = _layer_sensitivity_from_component_scores(
                head_scores,
                mlp_scores,
                num_layers=num_layers,
                target_prune_ratio=target_prune_ratio,
            )
            sensitivity_norm = _normalize_positive_scores(sensitivity_scores)
            keep_scores = [
                float((owl_norm[idx] + imp_norm[idx] + sensitivity_norm[idx]) / 3.0)
                for idx in range(num_layers)
            ]
    else:
        raise ValueError(
            f"Unsupported stage-2 allocation method '{allocation_method}'."
        )

    keep_scores = _normalize_positive_scores(keep_scores)
    prune_weights = [1.0 / score for score in keep_scores]
    keep_ratios = _build_stage2_keep_ratios_from_prune_weights(
        prune_weights,
        num_layers=num_layers,
        target_prune_ratio=target_prune_ratio,
        min_keep_ratio=min_keep_ratio,
        early_layer_min_keep_ratio=early_layer_min_keep_ratio,
        early_layer_count=early_layer_count,
        early_layer_max_prune_factor=early_layer_max_prune_factor,
    )
    return keep_ratios, {"layer_keep_scores": [float(score) for score in keep_scores]}
