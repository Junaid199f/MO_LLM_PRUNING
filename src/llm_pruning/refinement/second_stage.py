from __future__ import annotations

from collections import defaultdict
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from llm_pruning.core.utilities import remove_accelerate_hooks_if_present
from .stage2_structured_backends import apply_progressive_structured_pruning

try:
    from llm_pruning.importance.second_stage_criterion import (
        measure_second_stage_heads_importance,
        measure_second_stage_mlps_importance,
    )
except ImportError as exc:  # pragma: no cover
    measure_second_stage_heads_importance = None
    measure_second_stage_mlps_importance = None
    _SECOND_STAGE_IMPORT_ERROR = exc
else:
    _SECOND_STAGE_IMPORT_ERROR = None

_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}
_LLAMA_FAMILY = {"llama", "mistral", "qwen2", "phi3", "phi4"}
_COMPONENT_HEADS = 1
_COMPONENT_MLP = 2
_COMPONENT_NORM = 4
_COMPONENT_EMBEDDING = 8
_IMPORTANCE_METRICS = {"l1", "l2", "mean", "abs_mean"}
_SECOND_STAGE_METHODS = {
    "importance",
    "wanda-sp",
    "taylor",
    "gradient",
    "flap",
    "wactigrad",
}
_SECOND_STAGE_METHOD_ALIASES = {
    "importance": "importance",
    "importance-based": "importance",
    "importance_based": "importance",
    "wanda": "wanda-sp",
    "wanda-sp": "wanda-sp",
    "wanda_sp": "wanda-sp",
    "wandasp": "wanda-sp",
    "gradient": "gradient",
    "grad": "gradient",
    "gradient-based": "gradient",
    "gradient_based": "gradient",
    "taylor": "taylor",
    "flap": "flap",
    "wactigrad": "wactigrad",
    "wacti-grad": "wactigrad",
}


def _validate_importance_metric(metric: str) -> str:
    metric = (metric or "").lower()
    if metric not in _IMPORTANCE_METRICS:
        raise ValueError(
            f"Unsupported importance metric '{metric}'. Choose from {_IMPORTANCE_METRICS}."
        )
    return metric


def _validate_second_stage_method(method: str) -> str:
    normalized = (method or "importance").strip().lower().replace("_", "-")
    resolved = _SECOND_STAGE_METHOD_ALIASES.get(normalized)
    if resolved not in _SECOND_STAGE_METHODS:
        raise ValueError(
            f"Unsupported second-stage method '{method}'. Choose from {sorted(_SECOND_STAGE_METHODS)}."
        )
    return resolved


def _ensure_second_stage_available() -> None:
    if (
        measure_second_stage_heads_importance is None
        or measure_second_stage_mlps_importance is None
    ):
        raise ImportError(
            "Second-stage utilities could not be imported from proposed_second_stage."
        ) from _SECOND_STAGE_IMPORT_ERROR


def _infer_architecture(model) -> str:
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type == "llama":
        return "llama"
    if model_type == "mistral":
        return "mistral"
    if model_type == "qwen2":
        return "qwen2"
    if model_type == "phi3":
        return "phi3"
    if model_type == "phi4":
        return "phi4"
    raise ValueError(
        f"Unsupported architecture '{model_type}' for second-stage pruning."
    )


def _resolve_tokenizer(model) -> AutoTokenizer:
    identifier = getattr(model, "name_or_path", None) or getattr(
        model.config, "_name_or_path", None
    )
    if not identifier:
        raise ValueError(
            "Unable to infer the pretrained identifier for tokenizer loading. "
            "Ensure the model was loaded via from_pretrained."
        )

    tokenizer = _TOKENIZER_CACHE.get(identifier)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            identifier, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        _TOKENIZER_CACHE[identifier] = tokenizer
    return tokenizer


def _build_prompt(example: Dict[str, object]) -> Optional[str]:
    """Best-effort prompt extraction for a variety of dataset formats."""
    text = example.get("text") or example.get("content")
    if isinstance(text, str) and text.strip():
        return text

    instruction = example.get("instruction")
    input_text = example.get("input")
    if isinstance(instruction, str) and instruction.strip():
        combined = instruction.strip()
        if isinstance(input_text, str) and input_text.strip():
            combined += "\n" + input_text.strip()
        return combined

    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _load_prompt_json_file(prompt_file: Union[str, Path]) -> List[str]:
    path = Path(prompt_file)
    if not path.is_file() or path.suffix.lower() != ".json":
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Prompt file must contain a list of prompt objects: {path}")

    prompts: List[str] = []
    for idx, entry in enumerate(payload):
        if isinstance(entry, str):
            prompt = entry.strip()
        elif isinstance(entry, dict):
            prompt = _build_prompt(entry)
        else:
            raise ValueError(
                f"Unsupported prompt entry type in {path} at index {idx}: {type(entry)}"
            )
        if prompt:
            prompts.append(prompt)
    return prompts


def _normalize_dataset_specs(
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
) -> List[Dict[str, object]]:
    if isinstance(dataset_name, str):
        return [{"path": dataset_name, "split": dataset_split}]

    specs: List[Dict[str, object]] = []
    for item in dataset_name:
        if isinstance(item, str):
            specs.append({"path": item, "split": dataset_split})
            continue
        if isinstance(item, dict):
            spec = dict(item)
            if "path" not in spec:
                if "dataset" in spec:
                    spec["path"] = spec["dataset"]
                elif "name" in spec and "config" not in spec:
                    spec["path"] = spec["name"]
                    spec.pop("name", None)
            if "config" not in spec and "name" in spec:
                spec["config"] = spec.pop("name")
            spec.setdefault("split", dataset_split)
            specs.append(spec)
            continue
        raise TypeError(f"Unsupported dataset spec type: {type(item)}")
    return specs


def _load_prompts(
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
) -> List[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("Could not import datasets to build prompts: %s", exc)
        return []

    specs = _normalize_dataset_specs(dataset_name, dataset_split)
    is_multi = not isinstance(dataset_name, str) or len(specs) > 1
    per_dataset_cap = (
        max_prompts
        if not is_multi
        else max(1, int(math.ceil(max_prompts / len(specs))))
    )

    prompts: List[str] = []
    for spec in specs:
        dataset_path = spec.get("path")
        if not dataset_path:
            continue
        prompt_file = Path(str(dataset_path))
        if prompt_file.is_file() and prompt_file.suffix.lower() == ".json":
            file_prompts = _load_prompt_json_file(prompt_file)
            spec_max = spec.get("max_prompts")
            dataset_max = int(spec_max) if spec_max is not None else per_dataset_cap
            if random_subset and len(file_prompts) > dataset_max:
                rng = np.random.default_rng()
                selection = rng.choice(
                    len(file_prompts), size=dataset_max, replace=False
                )
                file_prompts = [file_prompts[idx] for idx in selection]
            prompts.extend(file_prompts[:dataset_max])
            continue

        config = spec.get("config")
        split = spec.get("split", dataset_split)
        spec_max = spec.get("max_prompts")
        dataset_max = int(spec_max) if spec_max is not None else per_dataset_cap
        dataset_random = bool(spec.get("random_subset", random_subset))

        try:
            if config:
                dataset = load_dataset(
                    dataset_path,
                    config,
                    split=split,
                )
            else:
                dataset = load_dataset(
                    dataset_path,
                    split=split,
                )
        except Exception as exc:  # pragma: no cover - runtime data fetch
            logging.warning(
                "Failed to load dataset %s/%s: %s", dataset_path, split, exc
            )
            if is_multi:
                continue
            return []

        if dataset_random and len(dataset) > dataset_max:
            dataset = dataset.shuffle()
        dataset = dataset.select(range(min(dataset_max, len(dataset))))

        collected = 0
        for example in dataset:
            prompt = _build_prompt(example)
            if prompt:
                prompts.append(prompt)
                collected += 1
            if collected >= dataset_max:
                break

    if len(prompts) > max_prompts:
        if random_subset:
            rng = np.random.default_rng()
            selection = rng.choice(len(prompts), size=max_prompts, replace=False)
            prompts = [prompts[idx] for idx in selection]
        else:
            prompts = prompts[:max_prompts]
    return prompts


def _iter_tokenized_prompts(
    tokenizer,
    prompts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
):
    for text in prompts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if not encoded or "input_ids" not in encoded:
            continue
        yield {key: value.to(device) for key, value in encoded.items()}


def _get_attention_score_module(layer):
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    return getattr(attn, "o_proj", None)


def _get_mlp_score_module(layer):
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    for name in ("down_proj", "fc2"):
        module = getattr(mlp, name, None)
        if module is not None and getattr(module, "weight", None) is not None:
            return module
    return None


def _group_attention_column_scores(layer, column_scores: torch.Tensor) -> np.ndarray:
    attn = getattr(layer, "self_attn", None)
    num_heads = getattr(attn, "num_heads", 0) if attn is not None else 0
    if num_heads <= 0:
        return np.array([])
    total_features = int(column_scores.numel())
    if total_features == 0 or total_features % num_heads != 0:
        return np.array([])
    head_dim = total_features // num_heads
    grouped = column_scores.view(num_heads, head_dim).sum(dim=1)
    return grouped.cpu().numpy()


def _group_attention_feature_l2(layer, feature_values: torch.Tensor) -> np.ndarray:
    attn = getattr(layer, "self_attn", None)
    num_heads = getattr(attn, "num_heads", 0) if attn is not None else 0
    if num_heads <= 0:
        return np.array([])
    total_features = int(feature_values.numel())
    if total_features == 0 or total_features % num_heads != 0:
        return np.array([])
    head_dim = total_features // num_heads
    grouped = feature_values.view(num_heads, head_dim)
    scores = torch.sqrt(torch.clamp(grouped.pow(2).sum(dim=1), min=0.0))
    return scores.cpu().numpy()


def _weight_column_scores(weight: torch.Tensor, importance_metric: str) -> torch.Tensor:
    metric = _validate_importance_metric(importance_metric)
    values = weight.detach().float()
    if metric == "l1":
        return values.abs().sum(dim=0)
    if metric == "l2":
        return torch.sqrt(values.pow(2).mean(dim=0))
    if metric == "mean":
        return values.mean(dim=0)
    return values.abs().mean(dim=0)


def _fallback_weight_only_scores(model, importance_metric: str):
    head_scores: Dict[int, np.ndarray] = {}
    mlp_scores: Dict[int, np.ndarray] = {}
    for layer_idx, layer in enumerate(
        getattr(getattr(model, "model", None), "layers", []) or []
    ):
        attn_module = _get_attention_score_module(layer)
        if attn_module is not None and getattr(attn_module, "weight", None) is not None:
            column_scores = _weight_column_scores(attn_module.weight, importance_metric)
            grouped = _group_attention_column_scores(layer, column_scores)
            if grouped.size:
                head_scores[layer_idx] = grouped

        mlp_module = _get_mlp_score_module(layer)
        if mlp_module is not None and getattr(mlp_module, "weight", None) is not None:
            mlp_scores[layer_idx] = (
                _weight_column_scores(mlp_module.weight, importance_metric)
                .cpu()
                .numpy()
            )
    return head_scores, mlp_scores


def _collect_activation_based_scores(
    model,
    tokenizer,
    *,
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    max_length: int,
    scoring_method: str,
    importance_metric: str,
    collect_heads: bool,
    collect_mlp: bool,
):
    prompts = _load_prompts(dataset_name, dataset_split, max_prompts, random_subset)
    if not prompts:
        logging.warning(
            "No prompts available for second-stage %s scoring; falling back to weight-only scores.",
            scoring_method,
        )
        return _fallback_weight_only_scores(model, importance_metric)

    layers = getattr(getattr(model, "model", None), "layers", []) or []
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    handles = []
    module_state: Dict[int, Dict[str, object]] = {}

    def register_module(layer_idx: int, kind: str, module, layer) -> None:
        if module is None or getattr(module, "weight", None) is None:
            return
        feature_dim = int(module.weight.shape[1])
        state: Dict[str, object] = {
            "kind": kind,
            "layer_idx": layer_idx,
            "module": module,
            "layer": layer,
            "sum_abs": torch.zeros(feature_dim, dtype=torch.float32, device=device),
            "count": 0,
        }
        if scoring_method == "flap":
            state["sum"] = torch.zeros(feature_dim, dtype=torch.float32, device=device)
            state["sum_sq"] = torch.zeros(
                feature_dim, dtype=torch.float32, device=device
            )
        module_state[id(module)] = state

        def _hook(_module, inputs, _output):
            if not inputs:
                return
            data = inputs[0]
            if data is None or data.ndim < 2:
                return
            data = data.detach().float()
            reduce_dims = tuple(range(data.ndim - 1))
            denom = int(np.prod(data.shape[:-1]))
            if denom <= 0:
                return
            state["sum_abs"] += data.abs().sum(dim=reduce_dims).to(device=device)
            if scoring_method == "flap":
                state["sum"] += data.sum(dim=reduce_dims).to(device=device)
                state["sum_sq"] += data.pow(2).sum(dim=reduce_dims).to(device=device)
            state["count"] = int(state["count"]) + denom

        handles.append(module.register_forward_hook(_hook))

    for layer_idx, layer in enumerate(layers):
        if collect_heads:
            register_module(
                layer_idx, "head", _get_attention_score_module(layer), layer
            )
        if collect_mlp:
            register_module(layer_idx, "mlp", _get_mlp_score_module(layer), layer)

    with torch.no_grad():
        for encoded in _iter_tokenized_prompts(
            tokenizer, prompts, device=device, max_length=max_length
        ):
            try:
                _ = model(**encoded)
            except Exception as exc:  # pragma: no cover - runtime best-effort
                logging.warning(
                    "Skipping prompt during %s scoring due to error: %s",
                    scoring_method,
                    exc,
                )

    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    head_scores: Dict[int, np.ndarray] = {}
    mlp_scores: Dict[int, np.ndarray] = {}
    for state in module_state.values():
        count = int(state["count"])
        if count <= 0:
            continue
        module = state["module"]
        layer = state["layer"]
        feature_scores = state["sum_abs"] / max(1, count)
        if scoring_method == "flap":
            mean = state["sum"] / max(1, count)
            second_moment = state["sum_sq"] / max(1, count)
            variance = torch.clamp(second_moment - mean.pow(2), min=0.0)
            feature_scores = variance.sqrt()
        weighted_scores = feature_scores * _weight_column_scores(
            module.weight, importance_metric
        ).to(device=feature_scores.device)
        layer_idx = int(state["layer_idx"])
        if state["kind"] == "head":
            grouped = _group_attention_column_scores(layer, weighted_scores)
            if grouped.size:
                head_scores[layer_idx] = grouped
        else:
            mlp_scores[layer_idx] = weighted_scores.cpu().numpy()

    return head_scores, mlp_scores


def _weight_gradient_feature_l2(
    weight: Optional[torch.Tensor],
    grad: Optional[torch.Tensor],
    *,
    reduce_dim: int,
) -> Optional[torch.Tensor]:
    if weight is None or grad is None:
        return None
    wg = weight.detach().float() * grad.detach().float()
    return torch.sqrt(torch.clamp(wg.pow(2).mean(dim=reduce_dim), min=0.0))


def _group_feature_scores_l2(
    feature_scores: Optional[torch.Tensor], num_groups: int
) -> Optional[torch.Tensor]:
    if feature_scores is None or feature_scores.ndim != 1 or num_groups <= 0:
        return None
    if feature_scores.numel() == 0 or feature_scores.numel() % num_groups != 0:
        return None
    group_width = feature_scores.numel() // num_groups
    grouped = feature_scores.view(num_groups, group_width)
    return torch.sqrt(torch.clamp(grouped.pow(2).sum(dim=1), min=0.0))


def _attention_wactigrad_scores(layer) -> Optional[np.ndarray]:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    num_heads = getattr(attn, "num_heads", 0)
    if num_heads <= 0:
        return None

    total = None
    q_proj = getattr(attn, "q_proj", None)
    if q_proj is not None:
        scores = _group_feature_scores_l2(
            _weight_gradient_feature_l2(
                getattr(q_proj, "weight", None),
                getattr(getattr(q_proj, "weight", None), "grad", None),
                reduce_dim=1,
            ),
            num_heads,
        )
        if scores is not None:
            total = scores if total is None else total + scores

    for proj_name in ("k_proj", "v_proj"):
        proj = getattr(attn, proj_name, None)
        if proj is None:
            continue
        kv_heads = getattr(attn, "num_key_value_heads", None) or num_heads
        scores = _group_feature_scores_l2(
            _weight_gradient_feature_l2(
                getattr(proj, "weight", None),
                getattr(getattr(proj, "weight", None), "grad", None),
                reduce_dim=1,
            ),
            kv_heads,
        )
        if scores is None:
            continue
        expanded = _expand_kv_group_scores(scores, num_heads)
        total = expanded if total is None else total + expanded

    o_proj = getattr(attn, "o_proj", None)
    if o_proj is not None:
        scores = _group_feature_scores_l2(
            _weight_gradient_feature_l2(
                getattr(o_proj, "weight", None),
                getattr(getattr(o_proj, "weight", None), "grad", None),
                reduce_dim=0,
            ),
            num_heads,
        )
        if scores is not None:
            total = scores if total is None else total + scores

    if total is None:
        return None
    return total.cpu().numpy()


def _mlp_wactigrad_scores(layer) -> Optional[np.ndarray]:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None

    total = None
    gate = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    gate_up_proj = getattr(mlp, "gate_up_proj", None)
    fc1 = getattr(mlp, "fc1", None)
    fc2 = getattr(mlp, "fc2", None)

    if gate is not None and up_proj is not None and down_proj is not None:
        for proj in (gate, up_proj):
            scores = _weight_gradient_feature_l2(
                getattr(proj, "weight", None),
                getattr(getattr(proj, "weight", None), "grad", None),
                reduce_dim=1,
            )
            if scores is not None:
                total = scores if total is None else total + scores
        scores = _weight_gradient_feature_l2(
            getattr(down_proj, "weight", None),
            getattr(getattr(down_proj, "weight", None), "grad", None),
            reduce_dim=0,
        )
        if scores is not None:
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    if gate_up_proj is not None and down_proj is not None:
        scores = _weight_gradient_feature_l2(
            getattr(gate_up_proj, "weight", None),
            getattr(getattr(gate_up_proj, "weight", None), "grad", None),
            reduce_dim=1,
        )
        if scores is not None and scores.numel() % 2 == 0:
            half = scores.numel() // 2
            merged = scores[:half] + scores[half:]
            total = merged if total is None else total + merged
        scores = _weight_gradient_feature_l2(
            getattr(down_proj, "weight", None),
            getattr(getattr(down_proj, "weight", None), "grad", None),
            reduce_dim=0,
        )
        if scores is not None:
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    if fc1 is not None and fc2 is not None:
        scores = _weight_gradient_feature_l2(
            getattr(fc1, "weight", None),
            getattr(getattr(fc1, "weight", None), "grad", None),
            reduce_dim=1,
        )
        if scores is not None:
            total = scores if total is None else total + scores
        scores = _weight_gradient_feature_l2(
            getattr(fc2, "weight", None),
            getattr(getattr(fc2, "weight", None), "grad", None),
            reduce_dim=0,
        )
        if scores is not None:
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    return None


def _collect_wactigrad_scores(
    model,
    tokenizer,
    *,
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    max_length: int,
    collect_heads: bool,
    collect_mlp: bool,
):
    prompts = _load_prompts(dataset_name, dataset_split, max_prompts, random_subset)
    if not prompts:
        logging.warning(
            "No prompts available for WActiGrad scoring; falling back to weight-only scores."
        )
        return _fallback_weight_only_scores(model, "l2")

    layers = getattr(getattr(model, "model", None), "layers", []) or []
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    handles = []
    module_state: Dict[int, Dict[str, object]] = {}

    def register_module(layer_idx: int, kind: str, module, layer) -> None:
        if module is None or getattr(module, "weight", None) is None:
            return
        feature_dim = int(module.weight.shape[1])
        module_state[id(module)] = {
            "kind": kind,
            "layer_idx": layer_idx,
            "layer": layer,
            "sum": torch.zeros(feature_dim, dtype=torch.float32, device=device),
            "count": 0,
        }

        def _hook(_module, inputs, _output):
            if not inputs:
                return
            data = inputs[0]
            if data is None or data.ndim < 2:
                return
            data = data.detach().float()
            reduce_dims = tuple(range(data.ndim - 1))
            denom = int(np.prod(data.shape[:-1]))
            if denom <= 0:
                return
            state = module_state[id(_module)]
            state["sum"] += data.sum(dim=reduce_dims).to(device=device)
            state["count"] = int(state["count"]) + denom

        handles.append(module.register_forward_hook(_hook))

    for layer_idx, layer in enumerate(layers):
        if collect_heads:
            register_module(
                layer_idx, "head", _get_attention_score_module(layer), layer
            )
        if collect_mlp:
            register_module(layer_idx, "mlp", _get_mlp_score_module(layer), layer)

    head_wg_scores: Dict[int, np.ndarray] = {}
    head_wg_counts: Dict[int, int] = defaultdict(int)
    mlp_wg_scores: Dict[int, np.ndarray] = {}
    mlp_wg_counts: Dict[int, int] = defaultdict(int)

    with torch.enable_grad():
        for encoded in _iter_tokenized_prompts(
            tokenizer, prompts, device=device, max_length=max_length
        ):
            model.zero_grad(set_to_none=True)
            try:
                outputs = model(**encoded, labels=encoded["input_ids"])
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    continue
                loss.backward()
            except Exception as exc:  # pragma: no cover - runtime best-effort
                logging.warning(
                    "Skipping prompt during WActiGrad scoring due to error: %s", exc
                )
                continue

            for layer_idx, layer in enumerate(layers):
                if collect_heads:
                    scores = _attention_wactigrad_scores(layer)
                    if scores is not None:
                        existing = head_wg_scores.get(layer_idx)
                        head_wg_scores[layer_idx] = (
                            scores if existing is None else existing + scores
                        )
                        head_wg_counts[layer_idx] += 1

                if collect_mlp:
                    scores = _mlp_wactigrad_scores(layer)
                    if scores is not None:
                        existing = mlp_wg_scores.get(layer_idx)
                        mlp_wg_scores[layer_idx] = (
                            scores if existing is None else existing + scores
                        )
                        mlp_wg_counts[layer_idx] += 1

    model.zero_grad(set_to_none=True)
    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    head_scores: Dict[int, np.ndarray] = {}
    mlp_scores: Dict[int, np.ndarray] = {}
    for state in module_state.values():
        count = int(state["count"])
        if count <= 0:
            continue
        layer_idx = int(state["layer_idx"])
        mean_activation = state["sum"] / max(1, count)
        if state["kind"] == "head":
            activation_scores = _group_attention_feature_l2(
                state["layer"], mean_activation
            )
            gradient_scores = None
            if head_wg_counts.get(layer_idx, 0) > 0:
                gradient_scores = head_wg_scores[layer_idx] / max(
                    1, head_wg_counts[layer_idx]
                )
            if gradient_scores is not None and activation_scores.size:
                head_scores[layer_idx] = activation_scores + gradient_scores
            elif gradient_scores is not None:
                head_scores[layer_idx] = gradient_scores
            elif activation_scores.size:
                head_scores[layer_idx] = activation_scores
        else:
            activation_scores = mean_activation.abs().cpu().numpy()
            gradient_scores = None
            if mlp_wg_counts.get(layer_idx, 0) > 0:
                gradient_scores = mlp_wg_scores[layer_idx] / max(
                    1, mlp_wg_counts[layer_idx]
                )
            if gradient_scores is not None and activation_scores.size:
                mlp_scores[layer_idx] = activation_scores + gradient_scores
            elif gradient_scores is not None:
                mlp_scores[layer_idx] = gradient_scores
            elif activation_scores.size:
                mlp_scores[layer_idx] = activation_scores

    return head_scores, mlp_scores


def _expand_kv_group_scores(
    group_scores: torch.Tensor, target_heads: int
) -> torch.Tensor:
    if group_scores.numel() == target_heads:
        return group_scores
    if group_scores.numel() <= 0:
        return torch.zeros(
            target_heads, device=group_scores.device, dtype=group_scores.dtype
        )
    repeats = max(1, target_heads // group_scores.numel())
    expanded = group_scores.repeat_interleave(repeats)
    if expanded.numel() < target_heads:
        pad = expanded.new_full((target_heads - expanded.numel(),), float(expanded[-1]))
        expanded = torch.cat([expanded, pad], dim=0)
    return expanded[:target_heads]


def _head_group_scores_from_rows(
    weight_salience: torch.Tensor, num_heads: int
) -> Optional[torch.Tensor]:
    if (
        num_heads <= 0
        or weight_salience.ndim != 2
        or weight_salience.shape[0] % num_heads != 0
    ):
        return None
    head_dim = weight_salience.shape[0] // num_heads
    return weight_salience.view(num_heads, head_dim, weight_salience.shape[1]).sum(
        dim=(1, 2)
    )


def _head_group_scores_from_cols(
    weight_salience: torch.Tensor, num_heads: int
) -> Optional[torch.Tensor]:
    if (
        num_heads <= 0
        or weight_salience.ndim != 2
        or weight_salience.shape[1] % num_heads != 0
    ):
        return None
    head_dim = weight_salience.shape[1] // num_heads
    return weight_salience.view(weight_salience.shape[0], num_heads, head_dim).sum(
        dim=(0, 2)
    )


def _attention_gradient_scores(layer) -> Optional[np.ndarray]:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    num_heads = getattr(attn, "num_heads", 0)
    if num_heads <= 0:
        return None

    total = None
    q_proj = getattr(attn, "q_proj", None)
    if (
        q_proj is not None
        and getattr(q_proj, "weight", None) is not None
        and q_proj.weight.grad is not None
    ):
        scores = _head_group_scores_from_rows(
            q_proj.weight.grad.detach().float().abs(), num_heads
        )
        if scores is not None:
            total = scores if total is None else total + scores

    for proj_name in ("k_proj", "v_proj"):
        proj = getattr(attn, proj_name, None)
        if (
            proj is None
            or getattr(proj, "weight", None) is None
            or proj.weight.grad is None
        ):
            continue
        kv_heads = getattr(attn, "num_key_value_heads", None) or num_heads
        scores = _head_group_scores_from_rows(
            proj.weight.grad.detach().float().abs(), kv_heads
        )
        if scores is None:
            continue
        expanded = _expand_kv_group_scores(scores, num_heads)
        total = expanded if total is None else total + expanded

    o_proj = getattr(attn, "o_proj", None)
    if (
        o_proj is not None
        and getattr(o_proj, "weight", None) is not None
        and o_proj.weight.grad is not None
    ):
        scores = _head_group_scores_from_cols(
            o_proj.weight.grad.detach().float().abs(), num_heads
        )
        if scores is not None:
            total = scores if total is None else total + scores

    if total is None:
        return None
    return total.cpu().numpy()


def _attention_taylor_scores(layer) -> Optional[np.ndarray]:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    num_heads = getattr(attn, "num_heads", 0)
    if num_heads <= 0:
        return None

    total = None
    q_proj = getattr(attn, "q_proj", None)
    if (
        q_proj is not None
        and getattr(q_proj, "weight", None) is not None
        and q_proj.weight.grad is not None
    ):
        salience = (
            q_proj.weight.detach().float() * q_proj.weight.grad.detach().float()
        ).abs()
        scores = _head_group_scores_from_rows(salience, num_heads)
        if scores is not None:
            total = scores if total is None else total + scores

    for proj_name in ("k_proj", "v_proj"):
        proj = getattr(attn, proj_name, None)
        if (
            proj is None
            or getattr(proj, "weight", None) is None
            or proj.weight.grad is None
        ):
            continue
        salience = (
            proj.weight.detach().float() * proj.weight.grad.detach().float()
        ).abs()
        kv_heads = getattr(attn, "num_key_value_heads", None) or num_heads
        scores = _head_group_scores_from_rows(salience, kv_heads)
        if scores is None:
            continue
        expanded = _expand_kv_group_scores(scores, num_heads)
        total = expanded if total is None else total + expanded

    o_proj = getattr(attn, "o_proj", None)
    if (
        o_proj is not None
        and getattr(o_proj, "weight", None) is not None
        and o_proj.weight.grad is not None
    ):
        salience = (
            o_proj.weight.detach().float() * o_proj.weight.grad.detach().float()
        ).abs()
        scores = _head_group_scores_from_cols(salience, num_heads)
        if scores is not None:
            total = scores if total is None else total + scores

    if total is None:
        return None
    return total.cpu().numpy()


def _mlp_gradient_scores(layer) -> Optional[np.ndarray]:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None

    total = None
    gate = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    gate_up_proj = getattr(mlp, "gate_up_proj", None)
    fc1 = getattr(mlp, "fc1", None)
    fc2 = getattr(mlp, "fc2", None)

    if gate is not None and up_proj is not None and down_proj is not None:
        for proj in (gate, up_proj):
            if (
                getattr(proj, "weight", None) is not None
                and proj.weight.grad is not None
            ):
                scores = proj.weight.grad.detach().float().abs().sum(dim=1)
                total = scores if total is None else total + scores
        if (
            getattr(down_proj, "weight", None) is not None
            and down_proj.weight.grad is not None
        ):
            scores = down_proj.weight.grad.detach().float().abs().sum(dim=0)
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    if gate_up_proj is not None and down_proj is not None:
        if (
            getattr(gate_up_proj, "weight", None) is not None
            and gate_up_proj.weight.grad is not None
        ):
            salience = gate_up_proj.weight.grad.detach().float().abs()
            if salience.shape[0] % 2 == 0:
                half = salience.shape[0] // 2
                scores = salience[:half].sum(dim=1) + salience[half:].sum(dim=1)
                total = scores if total is None else total + scores
        if (
            getattr(down_proj, "weight", None) is not None
            and down_proj.weight.grad is not None
        ):
            scores = down_proj.weight.grad.detach().float().abs().sum(dim=0)
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    if fc1 is not None and fc2 is not None:
        if getattr(fc1, "weight", None) is not None and fc1.weight.grad is not None:
            scores = fc1.weight.grad.detach().float().abs().sum(dim=1)
            total = scores if total is None else total + scores
        if getattr(fc2, "weight", None) is not None and fc2.weight.grad is not None:
            scores = fc2.weight.grad.detach().float().abs().sum(dim=0)
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    return None


def _mlp_taylor_scores(layer) -> Optional[np.ndarray]:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None

    total = None
    gate = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    gate_up_proj = getattr(mlp, "gate_up_proj", None)
    fc1 = getattr(mlp, "fc1", None)
    fc2 = getattr(mlp, "fc2", None)

    if gate is not None and up_proj is not None and down_proj is not None:
        for proj in (gate, up_proj):
            if (
                getattr(proj, "weight", None) is not None
                and proj.weight.grad is not None
            ):
                salience = (
                    proj.weight.detach().float() * proj.weight.grad.detach().float()
                ).abs()
                scores = salience.sum(dim=1)
                total = scores if total is None else total + scores
        if (
            getattr(down_proj, "weight", None) is not None
            and down_proj.weight.grad is not None
        ):
            salience = (
                down_proj.weight.detach().float()
                * down_proj.weight.grad.detach().float()
            ).abs()
            scores = salience.sum(dim=0)
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    if gate_up_proj is not None and down_proj is not None:
        if (
            getattr(gate_up_proj, "weight", None) is not None
            and gate_up_proj.weight.grad is not None
        ):
            salience = (
                gate_up_proj.weight.detach().float()
                * gate_up_proj.weight.grad.detach().float()
            ).abs()
            if salience.shape[0] % 2 == 0:
                half = salience.shape[0] // 2
                scores = salience[:half].sum(dim=1) + salience[half:].sum(dim=1)
                total = scores if total is None else total + scores
        if (
            getattr(down_proj, "weight", None) is not None
            and down_proj.weight.grad is not None
        ):
            salience = (
                down_proj.weight.detach().float()
                * down_proj.weight.grad.detach().float()
            ).abs()
            scores = salience.sum(dim=0)
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    if fc1 is not None and fc2 is not None:
        if getattr(fc1, "weight", None) is not None and fc1.weight.grad is not None:
            salience = (
                fc1.weight.detach().float() * fc1.weight.grad.detach().float()
            ).abs()
            scores = salience.sum(dim=1)
            total = scores if total is None else total + scores
        if getattr(fc2, "weight", None) is not None and fc2.weight.grad is not None:
            salience = (
                fc2.weight.detach().float() * fc2.weight.grad.detach().float()
            ).abs()
            scores = salience.sum(dim=0)
            total = scores if total is None else total + scores
        return None if total is None else total.cpu().numpy()

    return None


def _collect_gradient_scores(
    model,
    tokenizer,
    *,
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    max_length: int,
    collect_heads: bool,
    collect_mlp: bool,
):
    prompts = _load_prompts(dataset_name, dataset_split, max_prompts, random_subset)
    if not prompts:
        logging.warning(
            "No prompts available for gradient scoring; falling back to weight-only scores."
        )
        return _fallback_weight_only_scores(model, "l1")

    layers = getattr(getattr(model, "model", None), "layers", []) or []
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    head_scores = defaultdict(lambda: None)
    mlp_scores = defaultdict(lambda: None)
    processed = 0

    with torch.enable_grad():
        for encoded in _iter_tokenized_prompts(
            tokenizer, prompts, device=device, max_length=max_length
        ):
            model.zero_grad(set_to_none=True)
            try:
                outputs = model(**encoded, labels=encoded["input_ids"])
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    continue
                loss.backward()
            except Exception as exc:  # pragma: no cover - runtime best-effort
                logging.warning(
                    "Skipping prompt during gradient scoring due to error: %s", exc
                )
                continue

            for layer_idx, layer in enumerate(layers):
                if collect_heads:
                    scores = _attention_gradient_scores(layer)
                    if scores is not None:
                        existing = head_scores[layer_idx]
                        head_scores[layer_idx] = (
                            scores if existing is None else existing + scores
                        )

                if collect_mlp:
                    scores = _mlp_gradient_scores(layer)
                    if scores is not None:
                        existing = mlp_scores[layer_idx]
                        mlp_scores[layer_idx] = (
                            scores if existing is None else existing + scores
                        )

            processed += 1

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    processed = max(1, processed)
    return (
        {
            layer_idx: values / processed
            for layer_idx, values in head_scores.items()
            if values is not None
        },
        {
            layer_idx: values / processed
            for layer_idx, values in mlp_scores.items()
            if values is not None
        },
    )


def _collect_taylor_scores(
    model,
    tokenizer,
    *,
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    max_length: int,
    collect_heads: bool,
    collect_mlp: bool,
):
    prompts = _load_prompts(dataset_name, dataset_split, max_prompts, random_subset)
    if not prompts:
        logging.warning(
            "No prompts available for Taylor scoring; falling back to weight-only scores."
        )
        return _fallback_weight_only_scores(model, "l1")

    layers = getattr(getattr(model, "model", None), "layers", []) or []
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    head_scores = defaultdict(lambda: None)
    mlp_scores = defaultdict(lambda: None)
    processed = 0

    with torch.enable_grad():
        for encoded in _iter_tokenized_prompts(
            tokenizer, prompts, device=device, max_length=max_length
        ):
            model.zero_grad(set_to_none=True)
            try:
                outputs = model(**encoded, labels=encoded["input_ids"])
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    continue
                loss.backward()
            except Exception as exc:  # pragma: no cover - runtime best-effort
                logging.warning(
                    "Skipping prompt during Taylor scoring due to error: %s", exc
                )
                continue

            for layer_idx, layer in enumerate(layers):
                if collect_heads:
                    scores = _attention_taylor_scores(layer)
                    if scores is not None:
                        existing = head_scores[layer_idx]
                        head_scores[layer_idx] = (
                            scores if existing is None else existing + scores
                        )

                if collect_mlp:
                    scores = _mlp_taylor_scores(layer)
                    if scores is not None:
                        existing = mlp_scores[layer_idx]
                        mlp_scores[layer_idx] = (
                            scores if existing is None else existing + scores
                        )

            processed += 1

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    processed = max(1, processed)
    return (
        {
            layer_idx: values / processed
            for layer_idx, values in head_scores.items()
            if values is not None
        },
        {
            layer_idx: values / processed
            for layer_idx, values in mlp_scores.items()
            if values is not None
        },
    )


def _collect_stage2_scores(
    model,
    tokenizer,
    arch: str,
    *,
    scoring_method: str,
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    prompt_max_length: Optional[int],
    importance_metric: str,
    collect_heads: bool,
    collect_mlp: bool,
):
    scoring_method = _validate_second_stage_method(scoring_method)
    max_length = max(8, int(prompt_max_length or 256))

    if scoring_method == "importance":
        _ensure_second_stage_available()
        head_scores = {}
        mlp_scores = {}
        if collect_heads:
            head_scores = measure_second_stage_heads_importance(  # type: ignore[misc]
                model,
                tokenizer,
                arch,
                dataset_name=dataset_name,
                split=dataset_split,
                max_prompts=max_prompts,
                random_subset=random_subset,
            )
        if collect_mlp:
            mlp_scores = measure_second_stage_mlps_importance(  # type: ignore[misc]
                model,
                tokenizer,
                arch,
                dataset_name=dataset_name,
                split=dataset_split,
                max_prompts=max_prompts,
                random_subset=random_subset,
            )
        return head_scores, mlp_scores

    if scoring_method in {"wanda-sp", "flap"}:
        return _collect_activation_based_scores(
            model,
            tokenizer,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_prompts=max_prompts,
            random_subset=random_subset,
            max_length=max_length,
            scoring_method=scoring_method,
            importance_metric=importance_metric,
            collect_heads=collect_heads,
            collect_mlp=collect_mlp,
        )

    if scoring_method == "wactigrad":
        return _collect_wactigrad_scores(
            model,
            tokenizer,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_prompts=max_prompts,
            random_subset=random_subset,
            max_length=max_length,
            collect_heads=collect_heads,
            collect_mlp=collect_mlp,
        )

    if scoring_method == "gradient":
        return _collect_gradient_scores(
            model,
            tokenizer,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_prompts=max_prompts,
            random_subset=random_subset,
            max_length=max_length,
            collect_heads=collect_heads,
            collect_mlp=collect_mlp,
        )

    return _collect_taylor_scores(
        model,
        tokenizer,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        max_prompts=max_prompts,
        random_subset=random_subset,
        max_length=max_length,
        collect_heads=collect_heads,
        collect_mlp=collect_mlp,
    )


def _select_indices(
    importance: Union[np.ndarray, Sequence[float]],
    num_to_select: int,
    criterion: str,
) -> List[int]:
    if num_to_select <= 0:
        return []

    scores = np.asarray(importance)
    if scores.size == 0:
        return []

    num_to_select = min(num_to_select, scores.size)
    if criterion == "standard":
        order = np.argsort(scores)
        return order[:num_to_select].tolist()
    if criterion == "reversed":
        order = np.argsort(scores)[::-1]
        return order[:num_to_select].tolist()
    if criterion == "random":
        rng = np.random.default_rng()
        selection = rng.choice(scores.size, size=num_to_select, replace=False)
        return sorted(selection.tolist())
    raise ValueError(f"Invalid pruning criterion '{criterion}'.")


def _select_attention_heads_to_prune(
    attn_layer, head_scores: np.ndarray, layer_pruning_ratio: float, criterion: str
) -> List[int]:
    """Select a valid set of attention heads, respecting GQA KV-group constraints."""
    scores = np.asarray(head_scores)
    num_heads = int(scores.size)
    if num_heads <= 0:
        return []

    kv_heads = int(getattr(attn_layer, "num_key_value_heads", num_heads) or num_heads)
    group_size = getattr(attn_layer, "num_key_value_groups", None)
    is_gqa = kv_heads > 0 and kv_heads < num_heads
    if not is_gqa:
        num_prune_heads = int(round(layer_pruning_ratio * num_heads))
        num_prune_heads = max(0, min(num_prune_heads, num_heads - 1))
        return _select_indices(scores, num_prune_heads, criterion)

    if not group_size or group_size <= 0:
        if num_heads % kv_heads != 0:
            logging.warning(
                "Skipping GQA head selection: num_heads (%d) is not divisible by num_key_value_heads (%d).",
                num_heads,
                kv_heads,
            )
            return []
        group_size = num_heads // kv_heads
    if num_heads != kv_heads * int(group_size):
        logging.warning(
            "Skipping GQA head selection: num_heads (%d) != num_key_value_heads (%d) * group_size (%d).",
            num_heads,
            kv_heads,
            int(group_size),
        )
        return []

    grouped_scores = scores.reshape(kv_heads, int(group_size)).mean(axis=1)
    num_prune_groups = int(round(layer_pruning_ratio * kv_heads))
    num_prune_groups = max(0, min(num_prune_groups, kv_heads - 1))
    groups_to_prune = _select_indices(grouped_scores, num_prune_groups, criterion)
    return [
        int(group * int(group_size) + offset)
        for group in groups_to_prune
        for offset in range(int(group_size))
    ]


def _derive_kv_mapping(attn_layer, kept_heads: List[int]) -> List[int]:
    """Map kept query heads to their shared key/value head indices."""
    num_heads = getattr(attn_layer, "num_heads", 0)
    kv_heads = getattr(attn_layer, "num_key_value_heads", num_heads)
    if not kv_heads or kv_heads <= 0 or kv_heads == num_heads:
        return kept_heads.copy()

    groups = getattr(attn_layer, "num_key_value_groups", None)
    if not groups or groups <= 0:
        groups = max(1, num_heads // kv_heads)

    mapped = {max(0, min(kv_heads - 1, head // groups)) for head in kept_heads}
    return sorted(mapped) or [0]


def _prune_attention_heads(attn_layer, heads_to_prune: List[int]) -> None:
    if not heads_to_prune:
        return

    num_heads = getattr(attn_layer, "num_heads", 0)
    head_dim = getattr(attn_layer, "head_dim", 0)
    if num_heads == 0 or head_dim == 0:
        return

    kv_heads = getattr(attn_layer, "num_key_value_heads", num_heads)
    group_size = getattr(attn_layer, "num_key_value_groups", None)
    is_gqa = kv_heads and kv_heads < num_heads  # Grouped Query Attention

    if is_gqa:
        if not group_size or group_size <= 0:
            if num_heads % kv_heads != 0:
                logging.warning(
                    "Skipping attention pruning: num_heads (%d) not divisible by num_key_value_heads (%d).",
                    num_heads,
                    kv_heads,
                )
                return
            group_size = num_heads // kv_heads
        if num_heads != kv_heads * group_size:
            logging.warning(
                "Skipping attention pruning: num_heads (%d) != num_key_value_heads (%d) * num_key_value_groups (%d).",
                num_heads,
                kv_heads,
                group_size,
            )
            return
        pruned_groups = set(head // group_size for head in heads_to_prune)
        heads_to_keep = [
            group * group_size + offset
            for group in range(kv_heads)
            if group not in pruned_groups
            for offset in range(group_size)
        ]
        if not heads_to_keep:
            logging.warning(
                "Skipping attention pruning that would remove every KV group."
            )
            return
        kv_heads_to_keep = sorted(set(range(kv_heads)) - pruned_groups)
    else:
        heads_to_keep = sorted(set(range(num_heads)) - set(heads_to_prune))
        if not heads_to_keep:
            logging.warning("Skipping attention pruning that would remove every head.")
            return
        kv_heads_to_keep = _derive_kv_mapping(attn_layer, heads_to_keep)

    q_keep_indices: List[int] = []
    for head in heads_to_keep:
        q_keep_indices.extend(range(head * head_dim, (head + 1) * head_dim))

    kv_keep_indices: List[int] = []
    for kv_head in kv_heads_to_keep:
        kv_keep_indices.extend(range(kv_head * head_dim, (kv_head + 1) * head_dim))
    if not kv_keep_indices:
        kv_keep_indices = list(range(head_dim))

    with torch.no_grad():
        qkv_proj = getattr(attn_layer, "qkv_proj", None)
        if qkv_proj is not None and qkv_proj.weight is not None:
            # Assume standard concatenation [Q | K | V].
            q_chunk = num_heads * head_dim
            kv_chunk = kv_heads * head_dim
            q_indices = q_keep_indices
            k_indices = [idx + q_chunk for idx in kv_keep_indices]
            v_indices = [idx + q_chunk + kv_chunk for idx in kv_keep_indices]
            keep_indices = q_indices + k_indices + v_indices
            qkv_proj.weight = torch.nn.Parameter(
                qkv_proj.weight.data[keep_indices, :].clone()
            )
            if qkv_proj.bias is not None:
                qkv_proj.bias = torch.nn.Parameter(
                    qkv_proj.bias.data[keep_indices].clone()
                )
            qkv_proj.out_features = len(keep_indices)
        else:
            q_proj = getattr(attn_layer, "q_proj", None)
            if q_proj is not None and q_proj.weight is not None:
                q_proj.weight = torch.nn.Parameter(
                    q_proj.weight.data[q_keep_indices, :].clone()
                )
                if q_proj.bias is not None:
                    q_proj.bias = torch.nn.Parameter(
                        q_proj.bias.data[q_keep_indices].clone()
                    )
                q_proj.out_features = len(q_keep_indices)

            for proj_name in ("k_proj", "v_proj"):
                proj = getattr(attn_layer, proj_name, None)
                if proj is None or proj.weight is None:
                    continue
                proj.weight = torch.nn.Parameter(
                    proj.weight.data[kv_keep_indices, :].clone()
                )
                if proj.bias is not None:
                    proj.bias = torch.nn.Parameter(
                        proj.bias.data[kv_keep_indices].clone()
                    )
                proj.out_features = len(kv_keep_indices)

        o_proj = getattr(attn_layer, "o_proj", None)
        if o_proj is not None and o_proj.weight is not None:
            o_proj.weight = torch.nn.Parameter(
                o_proj.weight.data[:, q_keep_indices].clone()
            )
            o_proj.in_features = len(q_keep_indices)

        new_num_heads = len(heads_to_keep)
        new_kv_heads = len(kv_heads_to_keep)

        # For GQA models, ensure the constraint: new_num_heads % new_kv_heads == 0
        if is_gqa and new_num_heads > 0 and new_kv_heads > 0:
            if new_num_heads % new_kv_heads != 0:
                # Adjust KV heads to maintain divisibility
                for candidate_kv in range(min(new_kv_heads, new_num_heads), 0, -1):
                    if new_num_heads % candidate_kv == 0:
                        new_kv_heads = candidate_kv
                        logging.info(
                            "Adjusted num_key_value_heads from %d to %d to maintain GQA constraint "
                            "and ensure new_num_heads (%d) %% new_kv_heads == 0.",
                            len(kv_heads_to_keep),
                            new_kv_heads,
                            new_num_heads,
                        )
                        break

        attn_layer.num_heads = new_num_heads
        attn_layer.hidden_size = new_num_heads * head_dim
        if hasattr(attn_layer, "num_key_value_heads"):
            attn_layer.num_key_value_heads = max(1, new_kv_heads)
        if hasattr(attn_layer, "num_key_value_groups"):
            attn_layer.num_key_value_groups = max(
                1, new_num_heads // max(1, new_kv_heads)
            )


def _get_hidden_size(model) -> Optional[int]:
    for attr in ("hidden_size", "d_model"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    dim = getattr(embed_tokens, "embedding_dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    return None


def _embedding_importance(model, *, importance_metric: str) -> np.ndarray:
    """Estimate per-dimension importance scores for the embedding width."""
    importance_metric = _validate_importance_metric(importance_metric)
    model_body = getattr(model, "model", None)
    if model_body is None:
        return np.array([])

    first_layer = None
    try:
        layers = getattr(model_body, "layers", None)
        if layers:
            first_layer = layers[0]
    except Exception:  # pragma: no cover - defensive
        first_layer = None

    for norm_name in ("input_layernorm", "post_attention_layernorm"):
        norm = getattr(first_layer, norm_name, None) if first_layer else None
        weight = getattr(norm, "weight", None)
        if weight is not None:
            values = weight.detach().float()
            if importance_metric == "mean":
                values = values
            elif importance_metric == "l2":
                values = values.abs()
            else:
                values = values.abs()
            values = values.cpu().numpy()
            if values.size:
                return values

    embed_tokens = getattr(model_body, "embed_tokens", None)
    if embed_tokens is not None and getattr(embed_tokens, "weight", None) is not None:
        weights = embed_tokens.weight.detach().float()
        if importance_metric == "l1":
            scores = weights.abs().sum(dim=0)
        elif importance_metric == "l2":
            scores = torch.sqrt(weights.pow(2).mean(dim=0))
        elif importance_metric == "mean":
            scores = weights.mean(dim=0)
        else:
            scores = weights.abs().mean(dim=0)
        return scores.cpu().numpy()

    hidden_size = _get_hidden_size(model)
    if hidden_size:
        return np.ones(hidden_size, dtype=np.float32)
    return np.array([])


def _dataset_embedding_importance(
    model,
    tokenizer,
    *,
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    max_length: int = 256,
    importance_metric: str,
) -> np.ndarray:
    """Compute embedding-dimension importance from dataset activations."""
    importance_metric = _validate_importance_metric(importance_metric)
    prompts = _load_prompts(dataset_name, dataset_split, max_prompts, random_subset)
    if not prompts:
        logging.warning(
            "Falling back to weight-based embedding importance (no prompts)."
        )
        return _embedding_importance(model, importance_metric=importance_metric)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embedding = model.get_input_embeddings()
    if embedding is None:
        return _embedding_importance(model, importance_metric=importance_metric)

    accum: Optional[torch.Tensor] = None
    total: int = 0
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for text in prompts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            if not encoded or "input_ids" not in encoded:
                continue
            inputs = {k: v.to(device) for k, v in encoded.items()}
            embeds = embedding(inputs["input_ids"]).to(dtype=dtype)
            if embeds.ndim < 2:
                continue
            if accum is None:
                accum = torch.zeros(embeds.size(-1), device=device, dtype=torch.float32)
            reduce_dims = tuple(
                range(embeds.dim() - 1)
            )  # sum over batch/sequence if present
            if importance_metric == "l2":
                accum += embeds.pow(2).sum(dim=reduce_dims).float()
            elif importance_metric == "mean":
                accum += embeds.sum(dim=reduce_dims).float()
            else:
                accum += embeds.abs().sum(dim=reduce_dims).float()
            total += int(np.prod(embeds.shape[:-1]))

    if was_training:
        model.train()

    if accum is None or total == 0:
        logging.warning(
            "Falling back to weight-based embedding importance (no activations)."
        )
        return _embedding_importance(model, importance_metric=importance_metric)
    if importance_metric == "l2":
        scores = torch.sqrt(accum / max(1, total))
    else:
        scores = accum / max(1, total)
    return scores.cpu().numpy()


def _dataset_norm_importance(
    model,
    tokenizer,
    *,
    dataset_name: Union[str, Sequence[Union[str, Dict[str, object]]]],
    dataset_split: str,
    max_prompts: int,
    random_subset: bool,
    max_length: int = 256,
    importance_metric: str,
) -> Dict[int, np.ndarray]:
    """Collect per-channel norm activations keyed by module id."""
    importance_metric = _validate_importance_metric(importance_metric)
    prompts = _load_prompts(dataset_name, dataset_split, max_prompts, random_subset)
    if not prompts:
        logging.warning(
            "No prompts available for norm importance; falling back to weight norms."
        )
        return {}

    layers = getattr(getattr(model, "model", None), "layers", []) or []
    norm_modules: List[torch.nn.Module] = []
    for layer in layers:
        for norm_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn_layer_norm",
            "final_layer_norm",
        ):
            norm = getattr(layer, norm_name, None)
            if norm is not None and getattr(norm, "weight", None) is not None:
                norm_modules.append(norm)
    top_norm = getattr(getattr(model, "model", None), "norm", None)
    if top_norm is not None and getattr(top_norm, "weight", None) is not None:
        norm_modules.append(top_norm)

    if not norm_modules:
        return {}

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    sums = [
        torch.zeros_like(mod.weight, dtype=torch.float32, device=device)
        for mod in norm_modules
    ]
    counts = [0] * len(norm_modules)
    handles = []

    def make_hook(idx: int):
        def _hook(module, inputs, output):
            tensor = inputs[0] if inputs else output
            if tensor is None:
                return
            with torch.no_grad():
                data = tensor.detach().float()
                if data.dim() == 3:
                    reduce_dims = (0, 1)
                    denom = data.shape[0] * data.shape[1]
                elif data.dim() == 2:
                    reduce_dims = (0,)
                    denom = data.shape[0]
                else:
                    return
                if denom == 0:
                    return
                if importance_metric == "l2":
                    values = data.pow(2).sum(dim=reduce_dims)
                elif importance_metric == "mean":
                    values = data.sum(dim=reduce_dims)
                else:
                    values = data.abs().sum(dim=reduce_dims)
                if values.shape != sums[idx].shape:
                    return
                sums[idx] += values.to(device=sums[idx].device)
                counts[idx] += int(denom)

        return _hook

    for i, mod in enumerate(norm_modules):
        handles.append(mod.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for text in prompts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            if not encoded or "input_ids" not in encoded:
                continue
            inputs = {k: v.to(device) for k, v in encoded.items()}
            try:
                _ = model(**inputs)
            except Exception as exc:  # pragma: no cover - runtime best-effort
                logging.warning(
                    "Skipping prompt during norm collection due to error: %s", exc
                )
                continue

    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    importances: Dict[int, np.ndarray] = {}
    for mod, acc, denom in zip(norm_modules, sums, counts):
        if denom > 0:
            if importance_metric == "l2":
                scores = torch.sqrt(acc / max(1, denom))
            else:
                scores = acc / max(1, denom)
            importances[id(mod)] = scores.cpu().numpy()
    return importances


def _clone_norm(norm_module, keep_indices: List[int]):
    if norm_module is None or getattr(norm_module, "weight", None) is None:
        return norm_module

    weight = norm_module.weight.data[keep_indices].clone()
    bias = None
    existing_bias = getattr(norm_module, "bias", None)
    if existing_bias is not None:
        bias = existing_bias.data[keep_indices].clone()

    eps = getattr(norm_module, "eps", getattr(norm_module, "variance_epsilon", 1e-5))
    NormClass = type(norm_module)
    new_norm = NormClass(len(keep_indices), eps=eps)
    new_norm = new_norm.to(
        device=norm_module.weight.device, dtype=norm_module.weight.dtype
    )
    new_norm.weight = torch.nn.Parameter(weight)
    if bias is not None:
        new_norm.bias = torch.nn.Parameter(bias)
    return new_norm


def _prune_norm_channels(
    norm_module,
    pruning_ratio: float,
    importance: Optional[np.ndarray] = None,
    importance_metric: str = "l1",
) -> int:
    """Zero out the least-important RMSNorm/LN weights based on pruning ratio."""
    importance_metric = _validate_importance_metric(importance_metric)
    if norm_module is None or getattr(norm_module, "weight", None) is None:
        return 0

    weight = norm_module.weight.data
    total = weight.numel()
    if total == 0 or pruning_ratio <= 0:
        return 0

    num_prune = int(round(total * pruning_ratio))
    num_prune = max(0, min(num_prune, total - 1))
    if num_prune == 0:
        return 0

    if importance is not None and importance.shape[0] == total:
        scores = torch.as_tensor(importance, device=weight.device, dtype=weight.dtype)
    else:
        if importance_metric == "mean":
            scores = weight
        elif importance_metric == "l2":
            scores = weight.pow(2).sqrt()
        else:
            scores = weight.abs()
    prune_indices = torch.topk(scores, k=num_prune, largest=False).indices
    with torch.no_grad():
        norm_module.weight[prune_indices] = 0
        if getattr(norm_module, "bias", None) is not None:
            norm_module.bias[prune_indices] = 0
    return num_prune


def _clone_linear(
    module, *, keep_in: Optional[List[int]], keep_out: Optional[List[int]]
):
    if module is None or getattr(module, "weight", None) is None:
        return module

    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None
    if keep_out is not None:
        weight = weight[keep_out, :]
        if bias is not None:
            bias = bias[keep_out]
    if keep_in is not None:
        weight = weight[:, keep_in]

    new_linear = torch.nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_linear.weight = torch.nn.Parameter(weight.clone())
    if bias is not None:
        new_linear.bias = torch.nn.Parameter(bias.clone())
    return new_linear


@torch.no_grad()
def _prune_embedding_dimensions(model, dims_to_prune: List[int]) -> Optional[int]:
    if not dims_to_prune:
        return _get_hidden_size(model)

    hidden_size = _get_hidden_size(model)
    if hidden_size is None:
        logging.warning(
            "Unable to infer hidden size; skipping embedding-dimension pruning."
        )
        return None

    keep_indices = sorted(set(range(hidden_size)) - set(dims_to_prune))
    if not keep_indices:
        logging.warning("Skipping embedding pruning that would remove every dimension.")
        return hidden_size

    model_body = getattr(model, "model", None)
    if model_body is None:
        logging.warning("Model has no accessible body for embedding pruning; skipping.")
        return hidden_size

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    target_hidden = len(keep_indices)

    input_embeddings = model.get_input_embeddings()
    if (
        input_embeddings is not None
        and getattr(input_embeddings, "weight", None) is not None
    ):
        new_embedding = torch.nn.Embedding(
            input_embeddings.num_embeddings,
            target_hidden,
            padding_idx=input_embeddings.padding_idx,
            max_norm=input_embeddings.max_norm,
            norm_type=input_embeddings.norm_type,
            scale_grad_by_freq=input_embeddings.scale_grad_by_freq,
            sparse=input_embeddings.sparse,
            device=device,
            dtype=dtype,
        )
        new_embedding.weight = torch.nn.Parameter(
            input_embeddings.weight.data[:, keep_indices].clone()
        )
        model.set_input_embeddings(new_embedding)

    if (
        getattr(model_body, "embed_tokens", None) is not None
        and model_body.embed_tokens is not model.get_input_embeddings()
    ):
        embed_tokens = model_body.embed_tokens
        embed_tokens.weight = torch.nn.Parameter(
            embed_tokens.weight.data[:, keep_indices].clone()
        )
        embed_tokens.embedding_dim = target_hidden

    layers = getattr(model_body, "layers", []) or []
    for layer in layers:
        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm = _clone_norm(layer.input_layernorm, keep_indices)
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm = _clone_norm(
                layer.post_attention_layernorm, keep_indices
            )

        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "qkv_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and getattr(proj, "weight", None) is not None:
                    setattr(
                        attn,
                        proj_name,
                        _clone_linear(proj, keep_in=keep_indices, keep_out=None),
                    )

            o_proj = getattr(attn, "o_proj", None)
            if o_proj is not None and getattr(o_proj, "weight", None) is not None:
                setattr(
                    attn,
                    "o_proj",
                    _clone_linear(o_proj, keep_in=None, keep_out=keep_indices),
                )

            for attr in ("hidden_size", "embed_dim"):
                if hasattr(attn, attr):
                    setattr(attn, attr, target_hidden)

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "fc1"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None and getattr(proj, "weight", None) is not None:
                    setattr(
                        mlp,
                        proj_name,
                        _clone_linear(proj, keep_in=keep_indices, keep_out=None),
                    )

            for proj_name in ("down_proj", "fc2"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None and getattr(proj, "weight", None) is not None:
                    setattr(
                        mlp,
                        proj_name,
                        _clone_linear(proj, keep_in=None, keep_out=keep_indices),
                    )

            if hasattr(mlp, "hidden_size"):
                mlp.hidden_size = target_hidden

    if hasattr(model_body, "norm"):
        model_body.norm = _clone_norm(model_body.norm, keep_indices)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        model.lm_head = _clone_linear(lm_head, keep_in=keep_indices, keep_out=None)

    if hasattr(model.config, "hidden_size"):
        model.config.hidden_size = target_hidden
    if hasattr(model_body, "hidden_size"):
        model_body.hidden_size = target_hidden

    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning("Failed to re-tie embeddings after pruning: %s", exc)

    return target_hidden


def _prune_mlp_units(mlp_layer, units_to_prune: List[int]) -> None:
    if not units_to_prune:
        return

    gate = getattr(mlp_layer, "gate_proj", None)
    up_proj = getattr(mlp_layer, "up_proj", None)
    down_proj = getattr(mlp_layer, "down_proj", None)
    fused_gate_up = getattr(mlp_layer, "gate_up_proj", None)
    fc1 = getattr(mlp_layer, "fc1", None)
    fc2 = getattr(mlp_layer, "fc2", None)

    # Pattern 1: separate gate/up + down (Llama/Mistral/Qwen2)
    if gate is not None and up_proj is not None and down_proj is not None:
        intermediate_size = gate.out_features
        keep_indices = sorted(set(range(intermediate_size)) - set(units_to_prune))
        if not keep_indices:
            logging.warning("Skipping MLP pruning that would remove all neurons.")
            return

        with torch.no_grad():
            for proj in (gate, up_proj):
                proj.weight = torch.nn.Parameter(
                    proj.weight.data[keep_indices, :].clone()
                )
                if proj.bias is not None:
                    proj.bias = torch.nn.Parameter(proj.bias.data[keep_indices].clone())
                proj.out_features = len(keep_indices)

            down_proj.weight = torch.nn.Parameter(
                down_proj.weight.data[:, keep_indices].clone()
            )
            if down_proj.bias is not None:
                down_proj.bias = torch.nn.Parameter(down_proj.bias.data.clone())
            down_proj.in_features = len(keep_indices)
        return

    # Pattern 2: fused gate+up projection + down (Phi-3)
    if fused_gate_up is not None and down_proj is not None:
        total_out = fused_gate_up.out_features
        if total_out % 2 != 0:
            logging.warning(
                "Unexpected fused gate_up_proj size %d (expected even). Skipping.",
                total_out,
            )
            return
        intermediate_size = total_out // 2
        keep = sorted(set(range(intermediate_size)) - set(units_to_prune))
        if not keep:
            logging.warning("Skipping MLP pruning that would remove all neurons.")
            return
        fused_keep = keep + [idx + intermediate_size for idx in keep]
        with torch.no_grad():
            fused_gate_up.weight = torch.nn.Parameter(
                fused_gate_up.weight.data[fused_keep, :].clone()
            )
            if fused_gate_up.bias is not None:
                fused_gate_up.bias = torch.nn.Parameter(
                    fused_gate_up.bias.data[fused_keep].clone()
                )
            fused_gate_up.out_features = len(fused_keep)

            down_proj.weight = torch.nn.Parameter(
                down_proj.weight.data[:, keep].clone()
            )
            if down_proj.bias is not None:
                down_proj.bias = torch.nn.Parameter(down_proj.bias.data.clone())
            down_proj.in_features = len(keep)
        return

    # Pattern 3: generic feed-forward (fc1/fc2)
    if fc1 is not None and fc2 is not None:
        intermediate_size = fc1.out_features
        keep_indices = sorted(set(range(intermediate_size)) - set(units_to_prune))
        if not keep_indices:
            logging.warning("Skipping MLP pruning that would remove all neurons.")
            return
        with torch.no_grad():
            fc1.weight = torch.nn.Parameter(fc1.weight.data[keep_indices, :].clone())
            if fc1.bias is not None:
                fc1.bias = torch.nn.Parameter(fc1.bias.data[keep_indices].clone())
            fc1.out_features = len(keep_indices)

            fc2.weight = torch.nn.Parameter(fc2.weight.data[:, keep_indices].clone())
            if fc2.bias is not None:
                fc2.bias = torch.nn.Parameter(fc2.bias.data.clone())
            fc2.in_features = len(keep_indices)
        return

    logging.warning("MLP layer is missing expected projections; skipping.")


@torch.no_grad()
def apply_saved_second_stage_pruning(
    model,
    layer_stats: Optional[Sequence[Dict[str, object]]],
) -> Dict[str, Optional[int]]:
    """
    Reapply previously selected stage-2 head/MLP pruning decisions.

    This is intended for candidate replay paths where the exact per-layer pruned
    indices were already computed and stored during the original search.
    """
    first_layer = model.model.layers[0].self_attn
    original_heads = getattr(first_layer, "num_heads", 0)
    head_dim = getattr(first_layer, "head_dim", None)
    original_hidden_size = _get_hidden_size(model)
    if original_heads == 0 or head_dim is None:
        raise RuntimeError("Model appears to have no attention heads to prune.")

    if not layer_stats:
        return _summaries(
            original_heads, original_heads, head_dim, original_hidden_size
        )

    num_layers = len(getattr(getattr(model, "model", None), "layers", []) or [])
    applied_layer_stats: List[Dict[str, object]] = []
    total_saved_layers = len(layer_stats)
    for raw_layer_stat in layer_stats:
        if not isinstance(raw_layer_stat, dict):
            continue
        try:
            layer_idx = int(raw_layer_stat.get("layer_idx", -1))
        except (TypeError, ValueError):
            continue
        if layer_idx < 0 or layer_idx >= num_layers:
            logging.warning(
                "Skipping saved stage-2 layer stats for invalid layer index %s.",
                raw_layer_stat.get("layer_idx"),
            )
            continue

        layer = model.model.layers[layer_idx]
        heads_to_prune = sorted(
            {
                int(index)
                for index in raw_layer_stat.get("pruned_head_indices", []) or []
            }
        )
        mlp_to_prune = sorted(
            {int(index) for index in raw_layer_stat.get("pruned_mlp_indices", []) or []}
        )

        if heads_to_prune:
            _prune_attention_heads(layer.self_attn, heads_to_prune)
        if mlp_to_prune:
            _prune_mlp_units(layer.mlp, mlp_to_prune)
        applied_layer_stats.append(dict(raw_layer_stat))
        if (
            len(applied_layer_stats) == 1
            or len(applied_layer_stats) == total_saved_layers
            or len(applied_layer_stats) % 5 == 0
        ):
            logging.info(
                "Replayed saved stage-2 pruning for %d/%d layers (current layer=%d).",
                len(applied_layer_stats),
                total_saved_layers,
                layer_idx,
            )

    new_first_layer = model.model.layers[0].self_attn
    new_num_heads = getattr(new_first_layer, "num_heads", original_heads)
    model.config.num_attention_heads = new_num_heads
    if hasattr(model.config, "num_key_value_heads"):
        new_kv_heads = getattr(new_first_layer, "num_key_value_heads", None)
        if new_kv_heads is not None:
            model.config.num_key_value_heads = max(1, int(new_kv_heads))

    summary = _summaries(original_heads, new_num_heads, head_dim, original_hidden_size)
    summary["layer_stats"] = applied_layer_stats
    return summary


def _summaries(
    num_heads_before: int,
    num_heads_after: int,
    head_dim: Optional[int],
    hidden_size: Optional[int],
) -> Dict[str, Optional[int]]:
    return {
        "num_heads": num_heads_after,
        "key_dim": head_dim,
        "value_dim": head_dim,
        "hidden_size": hidden_size,
        "kl_score": None,
        "layer_stats": [],
    }


def _layer_scores(
    importance: Union[Dict[int, np.ndarray], Sequence[np.ndarray]], idx: int
):
    if isinstance(importance, dict):
        return importance.get(idx)
    if isinstance(importance, Sequence) and idx < len(importance):
        return importance[idx]
    return None


def _resolve_layer_component_masks(
    num_layers: int,
    *,
    prune_heads: bool,
    prune_mlp: bool,
    prune_embedding: bool,
    prune_norm: bool,
    layer_component_masks: Optional[Sequence[int]],
) -> Dict[str, object]:
    """Resolve per-layer component switches with global prune flags as upper bounds."""
    allow_heads = bool(prune_heads)
    allow_mlp = bool(prune_mlp)
    allow_embedding = bool(prune_embedding)
    allow_norm = bool(prune_norm)

    heads_by_layer = [allow_heads] * num_layers
    mlp_by_layer = [allow_mlp] * num_layers
    norm_by_layer = [allow_norm] * num_layers
    embedding_enabled = allow_embedding

    if layer_component_masks is not None:
        if not isinstance(layer_component_masks, (list, tuple)):
            raise ValueError(
                "layer_component_masks must be a list/tuple when provided."
            )
        if len(layer_component_masks) == 0:
            raise ValueError("layer_component_masks cannot be empty when provided.")

        heads_by_layer = []
        mlp_by_layer = []
        norm_by_layer = []
        embedding_enabled = False
        default_mask = int(layer_component_masks[-1])
        for layer_idx in range(num_layers):
            raw_mask = (
                layer_component_masks[layer_idx]
                if layer_idx < len(layer_component_masks)
                else default_mask
            )
            mask = int(raw_mask)
            if mask < 0:
                raise ValueError(
                    f"layer_component_masks[{layer_idx}] must be >= 0; got {mask}."
                )
            layer_heads = allow_heads and bool(mask & _COMPONENT_HEADS)
            layer_mlp = allow_mlp and bool(mask & _COMPONENT_MLP)
            layer_norm = allow_norm and bool(mask & _COMPONENT_NORM)
            layer_embedding = allow_embedding and bool(mask & _COMPONENT_EMBEDDING)
            heads_by_layer.append(layer_heads)
            mlp_by_layer.append(layer_mlp)
            norm_by_layer.append(layer_norm)
            embedding_enabled = embedding_enabled or layer_embedding

    return {
        "heads_by_layer": heads_by_layer,
        "mlp_by_layer": mlp_by_layer,
        "norm_by_layer": norm_by_layer,
        "embedding_enabled": embedding_enabled,
        "heads_enabled": any(heads_by_layer),
        "mlp_enabled": any(mlp_by_layer),
        "norm_enabled": any(norm_by_layer),
    }


@torch.no_grad()
def second_stage_optimize_heads(
    model,
    *,
    max_ratio: float = 1.0,
    layer_keep_ratios: Optional[
        Union[Dict[int, float], Sequence[Optional[float]]]
    ] = None,
    layer_component_masks: Optional[Sequence[int]] = None,
    apply_to_model: bool = False,
    prune_heads: bool = True,
    prune_mlp: bool = True,
    prune_embedding: bool = True,
    prune_norm: bool = False,
    dataset_name: Union[
        str, Sequence[Union[str, Dict[str, object]]]
    ] = "yahma/alpaca-cleaned",
    dataset_split: str = "train",
    max_prompts: int = 32,
    random_subset: bool = False,
    criterion: str = "standard",
    prompt_max_length: Optional[int] = None,
    use_dataset_for_embedding: bool = False,
    use_dataset_for_norm: bool = False,
    dataset_max_length: int = 256,
    importance_metric: str = "l1",
    scoring_method: str = "importance",
    seed: int = 0,
    flap_metric: str = "WIFV",
) -> Dict[str, Optional[int]]:
    """
    Second-stage structural pruning of attention heads, MLP neurons, and embedding dimensions.
    Per-layer keep ratios can override the global max_ratio when provided.
    """

    if not (0 < max_ratio <= 1.0):
        raise ValueError("max_ratio must be within (0, 1].")
    if layer_keep_ratios is not None and not isinstance(
        layer_keep_ratios, (dict, list, tuple)
    ):
        raise ValueError("layer_keep_ratios must be a dict or list when provided.")

    scoring_method = _validate_second_stage_method(scoring_method)
    if scoring_method in {"wanda-sp", "flap", "taylor", "gradient", "wactigrad"}:
        logging.info(
            "Stage-2 method %s performs joint structured pruning over attention heads and MLP neurons; "
            "ignoring prune_heads=%s, prune_mlp=%s, prune_embedding=%s, prune_norm=%s, and layer_component_masks.",
            scoring_method,
            prune_heads,
            prune_mlp,
            prune_embedding,
            prune_norm,
        )
        prune_heads = True
        prune_mlp = True
        prune_embedding = False
        prune_norm = False
        layer_component_masks = None
    config_msg = (
        "Second-stage pruning flags -> heads: %s, mlp: %s, embedding: %s, norm: %s, method: %s"
        % (prune_heads, prune_mlp, prune_embedding, prune_norm, scoring_method)
    )
    logging.info(config_msg)
    print(config_msg)

    importance_metric = _validate_importance_metric(importance_metric)
    first_layer = model.model.layers[0].self_attn
    num_layers = len(getattr(getattr(model, "model", None), "layers", []) or [])
    original_heads = getattr(first_layer, "num_heads", 0)
    head_dim = getattr(first_layer, "head_dim", None)
    original_hidden_size = _get_hidden_size(model)
    if original_heads == 0 or head_dim is None:
        raise RuntimeError("Model appears to have no attention heads to prune.")

    target_heads = (
        original_heads
        if not prune_heads
        else max(1, int(round(original_heads * max_ratio)))
    )
    arch = _infer_architecture(model)
    tokenizer = _resolve_tokenizer(model)
    max_prompts = max(1, max_prompts)

    component_usage = _resolve_layer_component_masks(
        num_layers,
        prune_heads=prune_heads,
        prune_mlp=prune_mlp,
        prune_embedding=prune_embedding,
        prune_norm=prune_norm,
        layer_component_masks=layer_component_masks,
    )
    heads_by_layer = component_usage["heads_by_layer"]
    mlp_by_layer = component_usage["mlp_by_layer"]
    norm_by_layer = component_usage["norm_by_layer"]
    embedding_enabled = bool(component_usage["embedding_enabled"])
    heads_enabled = bool(component_usage["heads_enabled"])
    mlp_enabled = bool(component_usage["mlp_enabled"])
    norm_enabled = bool(component_usage["norm_enabled"])

    if not any((heads_enabled, mlp_enabled, embedding_enabled, norm_enabled)):
        logging.info(
            "Second stage disabled: no active components after applying masks."
        )
        return _summaries(
            original_heads, original_heads, head_dim, original_hidden_size
        )

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        # Keep Accelerate hooks for sharded models so inputs are dispatched correctly.
        unique_devices = {
            str(device)
            for device in device_map.values()
            if device not in (None, "disk")
        }
        if len(unique_devices) <= 1:
            remove_accelerate_hooks_if_present(model)
    else:
        remove_accelerate_hooks_if_present(model)

    second_stage_head_importances = {}
    second_stage_mlp_importances = {}
    layer_stats: List[Dict[str, int]] = []
    if scoring_method in {"wanda-sp", "flap"} and (heads_enabled or mlp_enabled):
        logging.info(
            "Applying progressive second-stage pruning (%d prompts, dataset=%s/%s, criterion=%s, method=%s, flap_metric=%s).",
            max_prompts,
            dataset_name,
            dataset_split,
            criterion,
            scoring_method,
            flap_metric,
        )
        layer_stats = apply_progressive_structured_pruning(
            model,
            tokenizer,
            scoring_method=scoring_method,
            dataset_name=dataset_name,
            max_prompts=max_prompts,
            prompt_max_length=prompt_max_length or dataset_max_length,
            seed=seed,
            criterion=criterion,
            layer_keep_ratios=layer_keep_ratios,
            max_ratio=max_ratio,
            heads_by_layer=heads_by_layer,
            mlp_by_layer=mlp_by_layer,
            flap_metric=flap_metric,
            select_indices_fn=_select_indices,
            prune_attention_fn=_prune_attention_heads,
            prune_mlp_fn=_prune_mlp_units,
        )
    elif heads_enabled or mlp_enabled:
        logging.info(
            "Collecting second-stage scores (%d prompts, dataset=%s/%s, criterion=%s, method=%s).",
            max_prompts,
            dataset_name,
            dataset_split,
            criterion,
            scoring_method,
        )
        second_stage_head_importances, second_stage_mlp_importances = (
            _collect_stage2_scores(
                model,
                tokenizer,
                arch,
                scoring_method=scoring_method,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                max_prompts=max_prompts,
                random_subset=random_subset,
                prompt_max_length=prompt_max_length or dataset_max_length,
                importance_metric=importance_metric,
                collect_heads=heads_enabled,
                collect_mlp=mlp_enabled,
            )
        )

    if layer_keep_ratios:
        if isinstance(layer_keep_ratios, dict):
            ratio_values = [v for v in layer_keep_ratios.values() if v is not None]
        else:
            ratio_values = [v for v in layer_keep_ratios if v is not None]
        if ratio_values:
            first_ratio = ratio_values[0]
            if any(abs(v - first_ratio) > 1e-6 for v in ratio_values[1:]):
                logging.warning(
                    "Per-layer keep ratios provided; model.config.num_attention_heads is set from layer 0."
                )

    global_pruning_ratio = max(0.0, min(1.0, 1.0 - max_ratio))
    if global_pruning_ratio == 0.0 and not layer_keep_ratios:
        return _summaries(
            original_heads, original_heads, head_dim, original_hidden_size
        )

    pruning_ratio = global_pruning_ratio
    if scoring_method in {"wanda-sp", "flap"}:
        logging.info("Stage-2 %s pruning applied progressively.", scoring_method)
    else:
        logging.info(
            "Applying second-stage pruning with prune ratio %.4f.", pruning_ratio
        )
    for layer_idx, layer in enumerate(model.model.layers):
        if scoring_method in {"wanda-sp", "flap"}:
            continue

        head_scores = _layer_scores(second_stage_head_importances, layer_idx)
        mlp_scores = _layer_scores(second_stage_mlp_importances, layer_idx)
        if head_scores is None or mlp_scores is None:
            logging.warning(
                "Skipping layer %d due to missing second-stage scores.", layer_idx
            )
            continue

        head_scores = np.asarray(head_scores)
        mlp_scores = np.asarray(mlp_scores)
        num_heads = head_scores.size
        intermediate_size = mlp_scores.size
        if num_heads == 0 or intermediate_size == 0:
            logging.warning(
                "Layer %d has no measurable heads or neurons; skipping.", layer_idx
            )
            continue

        layer_keep_ratio = max_ratio
        if layer_keep_ratios is not None:
            if isinstance(layer_keep_ratios, dict):
                layer_keep_ratio = layer_keep_ratios.get(layer_idx, max_ratio)
            elif layer_idx < len(layer_keep_ratios):
                layer_keep_ratio = layer_keep_ratios[layer_idx]
            if layer_keep_ratio is None:
                layer_keep_ratio = max_ratio
        if not (0 < layer_keep_ratio <= 1.0):
            raise ValueError(
                f"Layer {layer_idx} keep ratio must be within (0, 1]; got {layer_keep_ratio}."
            )
        layer_pruning_ratio = max(0.0, min(1.0, 1.0 - layer_keep_ratio))
        logging.info(
            "Layer %d pruning ratio: %.2f%% (keep ratio: %.2f%%).",
            layer_idx,
            layer_pruning_ratio * 100.0,
            layer_keep_ratio * 100.0,
        )
        print(
            "Layer %d pruning ratio: %.2f%% (keep ratio: %.2f%%)."
            % (layer_idx, layer_pruning_ratio * 100.0, layer_keep_ratio * 100.0)
        )

        layer_prune_heads = (
            heads_by_layer[layer_idx]
            if layer_idx < len(heads_by_layer)
            else bool(prune_heads)
        )
        layer_prune_mlp = (
            mlp_by_layer[layer_idx]
            if layer_idx < len(mlp_by_layer)
            else bool(prune_mlp)
        )

        num_prune_heads = 0
        heads_to_prune: List[int] = []
        if layer_prune_heads:
            heads_to_prune = _select_attention_heads_to_prune(
                layer.self_attn,
                head_scores,
                layer_pruning_ratio,
                criterion,
            )
            num_prune_heads = len(heads_to_prune)

        num_prune_mlp_units = 0
        mlp_units_to_prune: List[int] = []
        if layer_prune_mlp and intermediate_size > 0:
            # Enforce the same per-layer pruning ratio for MLP units as requested.
            num_prune_mlp_units = int(round(intermediate_size * layer_pruning_ratio))
            num_prune_mlp_units = max(
                0, min(num_prune_mlp_units, intermediate_size - 1)
            )
            mlp_units_to_prune = _select_indices(
                mlp_scores, num_prune_mlp_units, criterion
            )

        if heads_to_prune:
            logging.debug(
                "Layer %d: pruning %d/%d heads.",
                layer_idx,
                len(heads_to_prune),
                num_heads,
            )
        if mlp_units_to_prune:
            logging.debug(
                "Layer %d: pruning %d/%d MLP neurons.",
                layer_idx,
                len(mlp_units_to_prune),
                intermediate_size,
            )

        _prune_attention_heads(layer.self_attn, heads_to_prune)
        _prune_mlp_units(layer.mlp, mlp_units_to_prune)
        layer_stats.append(
            {
                "layer_idx": int(layer_idx),
                "heads_pruned": int(len(heads_to_prune) > 0),
                "mlp_pruned": int(len(mlp_units_to_prune) > 0),
                "total_heads": int(num_heads),
                "total_mlp_units": int(intermediate_size),
                "num_pruned_heads": int(len(heads_to_prune)),
                "num_pruned_mlp_units": int(len(mlp_units_to_prune)),
                "pruned_head_indices": [int(index) for index in heads_to_prune],
                "pruned_mlp_indices": [int(index) for index in mlp_units_to_prune],
            }
        )

    new_first_layer = model.model.layers[0].self_attn
    new_num_heads = getattr(new_first_layer, "num_heads", target_heads)
    if heads_enabled:
        model.config.num_attention_heads = new_num_heads
        if hasattr(model.config, "num_key_value_heads"):
            new_kv_heads = getattr(new_first_layer, "num_key_value_heads", None)
            if new_kv_heads is None:
                new_kv_heads = min(
                    getattr(model.config, "num_key_value_heads"), new_num_heads
                )
            model.config.num_key_value_heads = max(1, new_kv_heads)
            if hasattr(model.config, "num_key_value_groups"):
                groups = getattr(new_first_layer, "num_key_value_groups", None)
                if groups is None:
                    groups = max(
                        1, new_num_heads // max(1, model.config.num_key_value_heads)
                    )
                model.config.num_key_value_groups = groups
    else:
        new_num_heads = original_heads
    new_mlp = getattr(model.model.layers[0], "mlp", None)
    new_intermediate = None
    if new_mlp is not None:
        if hasattr(new_mlp, "gate_proj"):
            new_intermediate = getattr(new_mlp.gate_proj, "out_features", None)
        elif hasattr(new_mlp, "gate_up_proj"):
            out_f = getattr(new_mlp.gate_up_proj, "out_features", None)
            if out_f is not None:
                new_intermediate = out_f // 2
        elif hasattr(new_mlp, "fc1"):
            new_intermediate = getattr(new_mlp.fc1, "out_features", None)
    if mlp_enabled and new_intermediate is not None:
        model.config.intermediate_size = new_intermediate

    new_hidden_size = None
    if embedding_enabled:
        if use_dataset_for_embedding:
            logging.info(
                "Collecting dataset-driven embedding importance (%d prompts, dataset=%s/%s).",
                max_prompts,
                dataset_name,
                dataset_split,
            )
            embedding_importances = _dataset_embedding_importance(
                model,
                tokenizer,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                max_prompts=max_prompts,
                random_subset=random_subset,
                max_length=dataset_max_length,
                importance_metric=importance_metric,
            )
        else:
            embedding_importances = _embedding_importance(
                model, importance_metric=importance_metric
            )
        base_hidden = int(
            original_hidden_size
            or (
                embedding_importances.size
                if hasattr(embedding_importances, "size")
                else 0
            )
        )
        num_prune_embed_dims = int(round(pruning_ratio * max(0, base_hidden)))
        if original_hidden_size is not None:
            num_prune_embed_dims = max(
                0, min(num_prune_embed_dims, max(1, original_hidden_size) - 1)
            )
        embedding_dims_to_prune: List[int] = []
        if num_prune_embed_dims > 0:
            embedding_dims_to_prune = _select_indices(
                embedding_importances, num_prune_embed_dims, criterion
            )
            if not embedding_dims_to_prune and base_hidden:
                embedding_dims_to_prune = list(
                    range(max(0, base_hidden - num_prune_embed_dims), base_hidden)
                )
            logging.debug(
                "Pruning %d/%d embedding dimensions based on %s criterion.",
                num_prune_embed_dims,
                base_hidden,
                criterion,
            )
            print(
                f"Pruning {num_prune_embed_dims}/{base_hidden} embedding dimensions based on {criterion} criterion."
            )
        new_hidden_size = _prune_embedding_dimensions(model, embedding_dims_to_prune)

    if norm_enabled:
        layers = getattr(getattr(model, "model", None), "layers", []) or []
        norm_prune_ratio = pruning_ratio
        total_norm_channels = 0
        pruned_norm_channels = 0
        norm_importances: Dict[int, np.ndarray] = {}
        if use_dataset_for_norm:
            logging.info(
                "Collecting dataset-driven norm importance (%d prompts, dataset=%s/%s).",
                max_prompts,
                dataset_name,
                dataset_split,
            )
            norm_importances = _dataset_norm_importance(
                model,
                tokenizer,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                max_prompts=max_prompts,
                random_subset=random_subset,
                max_length=dataset_max_length,
                importance_metric=importance_metric,
            )
        for layer_idx, layer in enumerate(layers):
            layer_prune_norm = (
                norm_by_layer[layer_idx]
                if layer_idx < len(norm_by_layer)
                else bool(prune_norm)
            )
            if not layer_prune_norm:
                continue
            for norm_name in (
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn_layer_norm",
                "final_layer_norm",
            ):
                norm = getattr(layer, norm_name, None)
                weight = getattr(norm, "weight", None)
                if norm is None or weight is None:
                    continue
                total_norm_channels += weight.numel()
                pruned = _prune_norm_channels(
                    norm,
                    norm_prune_ratio,
                    norm_importances.get(id(norm)),
                    importance_metric=importance_metric,
                )
                pruned_norm_channels += pruned
                if pruned:
                    logging.debug(
                        "Layer %d %s: pruned %d/%d norm channels (%.2f%%).",
                        layer_idx,
                        norm_name,
                        pruned,
                        weight.numel(),
                        100.0 * pruned / max(1, weight.numel()),
                    )
        top_norm = getattr(getattr(model, "model", None), "norm", None)
        if top_norm is not None and getattr(top_norm, "weight", None) is not None:
            total_norm_channels += top_norm.weight.numel()
            pruned = _prune_norm_channels(
                top_norm,
                norm_prune_ratio,
                norm_importances.get(id(top_norm)),
                importance_metric=importance_metric,
            )
            pruned_norm_channels += pruned
            if pruned:
                logging.debug(
                    "Top-level norm: pruned %d/%d channels (%.2f%%).",
                    pruned,
                    top_norm.weight.numel(),
                    100.0 * pruned / max(1, top_norm.weight.numel()),
                )
        if total_norm_channels > 0:
            logging.info(
                "Pruned %d/%d (%.2f%%) normalization channels with ratio %.4f.",
                pruned_norm_channels,
                total_norm_channels,
                100.0 * pruned_norm_channels / max(1, total_norm_channels),
                norm_prune_ratio,
            )

    summary = _summaries(
        original_heads, new_num_heads, head_dim, new_hidden_size or original_hidden_size
    )
    summary["layer_stats"] = layer_stats
    return summary
