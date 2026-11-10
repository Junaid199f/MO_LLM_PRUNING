from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

from .utilities import remove_accelerate_hooks_if_present


class IdentityTransformerBlock(nn.Module):
    """Replaces an entire transformer block with an identity mapping."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return (hidden_states,)


class PrunedSelfAttention(nn.Module):
    """Lightweight attention stub that returns zeros and keeps the signature."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        zeros = torch.zeros_like(hidden_states)
        return zeros, None, None


class PrunedMLP(nn.Module):
    """Lightweight MLP stub that returns zeros."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return torch.zeros_like(hidden_states)


def _replace_layernorm(layer, attribute_names: Iterable[str]):
    """Replace layernorm modules with identity layers when available."""
    for name in attribute_names:
        if hasattr(layer, name):
            setattr(layer, name, nn.Identity())


def apply_structural_pruning(
    model,
    attn_mask: List[int],
    mlp_mask: List[int],
    *,
    drop_layernorm: bool = True,
) -> None:
    """
    Convert binary EvoPress masks into structural pruning by replacing blocks and
    submodules with lightweight identities.

    Args:
      model: Transformer model whose ``model.layers`` correspond to the masks.
      attn_mask: Binary mask (1 = prune attention submodule).
      mlp_mask: Binary mask (1 = prune MLP submodule).
      drop_layernorm: When True, layer norm modules adjacent to pruned submodules
        are replaced with identities to avoid unnecessary computation.
    """
    if len(attn_mask) != len(model.model.layers) or len(mlp_mask) != len(
        model.model.layers
    ):
        raise ValueError("Mask lengths must match the number of transformer layers.")

    remove_accelerate_hooks_if_present(model)

    for idx, layer in enumerate(model.model.layers):
        prune_attn = bool(attn_mask[idx])
        prune_mlp = bool(mlp_mask[idx])

        if prune_attn and prune_mlp:
            model.model.layers[idx] = IdentityTransformerBlock()
            continue

        if prune_attn:
            layer.self_attn = PrunedSelfAttention()
            if drop_layernorm:
                _replace_layernorm(
                    layer,
                    (
                        "input_layernorm",
                        "self_attn_layer_norm",
                        "ln1",
                        "attention_norm",
                    ),
                )

        if prune_mlp:
            layer.mlp = PrunedMLP()
            if drop_layernorm:
                _replace_layernorm(
                    layer,
                    (
                        "post_attention_layernorm",
                        "postatt_layernorm",
                        "ln_2",
                        "ffn_norm",
                    ),
                )
