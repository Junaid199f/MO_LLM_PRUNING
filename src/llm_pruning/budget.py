"""Pure sparsity-budget arithmetic and monotone layer allocations."""

import logging
import math
from typing import List, Optional, Sequence


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _allocate_prune_budget(
    weights: Sequence[float],
    target_total: float,
    *,
    max_prune_by_layer: Sequence[float],
) -> List[float]:
    """Allocate an exact global prune budget across layers using monotone weights."""
    num_layers = len(weights)
    if num_layers == 0:
        return []
    if len(max_prune_by_layer) != num_layers:
        raise ValueError(
            "max_prune_by_layer must have one entry per layer; "
            f"got {len(max_prune_by_layer)} entries for {num_layers} layers."
        )

    layer_caps = [
        min(1.0, max(0.0, float(max_prune))) for max_prune in max_prune_by_layer
    ]
    target_total = min(sum(layer_caps), max(0.0, float(target_total)))
    if target_total <= 0.0:
        return [0.0] * num_layers
    if max(layer_caps, default=0.0) <= 0.0:
        return [0.0] * num_layers

    ratios = [0.0] * num_layers
    active_indices = list(range(num_layers))
    remaining_budget = target_total
    tolerance = 1e-12

    while active_indices and remaining_budget > tolerance:
        active_weights = [max(0.0, float(weights[idx])) for idx in active_indices]
        active_weight_sum = sum(active_weights)
        if active_weight_sum <= tolerance:
            equal_share = remaining_budget / len(active_indices)
            proposed = [equal_share] * len(active_indices)
        else:
            proposed = [
                remaining_budget * (weight / active_weight_sum)
                for weight in active_weights
            ]

        saturated = []
        for pos, layer_idx in enumerate(active_indices):
            layer_cap = layer_caps[layer_idx]
            if proposed[pos] >= layer_cap - tolerance:
                ratios[layer_idx] = layer_cap
                saturated.append(layer_idx)

        if not saturated:
            for pos, layer_idx in enumerate(active_indices):
                ratios[layer_idx] = proposed[pos]
            remaining_budget = 0.0
            break

        remaining_budget -= sum(layer_caps[layer_idx] for layer_idx in saturated)
        active_indices = [idx for idx in active_indices if idx not in saturated]

    if remaining_budget > tolerance and active_indices:
        last_idx = active_indices[-1]
        ratios[last_idx] = min(
            layer_caps[last_idx], ratios[last_idx] + remaining_budget
        )

    return [
        min(layer_caps[idx], max(0.0, float(ratio))) for idx, ratio in enumerate(ratios)
    ]


def _build_monotone_prune_ratios(
    num_layers: int,
    target_prune_ratio: float,
    k: float,
    c: float,
    *,
    min_keep_ratio: float = 0.01,
    early_layer_min_keep_ratio: Optional[float] = None,
    early_layer_count: int = 2,
    early_layer_max_prune_factor: float = 0.75,
) -> List[float]:
    """Monotone prune ratios: lower in early layers, higher in deeper layers."""
    if num_layers <= 0:
        return []
    k = max(0.01, float(k))
    c = min(1.0, max(0.0, float(c)))
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

    weights = []
    for layer_idx in range(num_layers):
        pos = layer_idx / max(1, num_layers - 1)
        weights.append(_sigmoid(k * (pos - c)))
    target_total = num_layers * target_prune_ratio
    max_total = sum(layer_caps)
    if target_total > max_total + 1e-9:
        logging.warning(
            "Requested BO prune budget %.6f exceeds safeguarded capacity %.6f; "
            "using the maximum feasible allocation instead.",
            target_total,
            max_total,
        )
    prune_ratios = _allocate_prune_budget(
        weights,
        target_total,
        max_prune_by_layer=layer_caps,
    )
    achieved_mean = (sum(prune_ratios) / num_layers) if num_layers else 0.0
    if abs(achieved_mean - target_prune_ratio) > 1e-6:
        logging.warning(
            "Layer prune allocation drifted from target: requested %.6f, achieved %.6f.",
            target_prune_ratio,
            achieved_mean,
        )
    return prune_ratios
