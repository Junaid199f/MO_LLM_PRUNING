from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from datasets import load_dataset


class WrappedGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1).pow(2) / self.nsamples

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()


class BiasGPT:
    def __init__(self, layer, metric: str):
        self.layer = layer
        self.dev = layer.weight.device
        self.in_dim = layer.weight.data.shape[1]
        self.metric = metric
        self.nsamples = 0
        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if metric == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        old_baseline_inp = self.baseline_inp.clone()
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.metric == "WIFN":
            inp = inp.type(torch.float32)
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1).pow(2) / (
                self.nsamples + batch_size
            )
        else:
            if self.nsamples == 0:
                self.fluc_inp = torch.zeros_like(self.fluc_inp)
            else:
                self.fluc_inp *= (self.nsamples - 1) / max(
                    1, self.nsamples + batch_size - 1
                )
                self.fluc_inp += torch.sum(
                    (inp - self.baseline_inp.unsqueeze(1))
                    * (inp - old_baseline_inp.unsqueeze(1)),
                    dim=1,
                ) / (self.nsamples + batch_size)

        self.nsamples += batch_size

    def free(self):
        self.baseline_inp = None
        if hasattr(self, "fluc_inp"):
            self.fluc_inp = None
        if hasattr(self, "scaler_inp"):
            self.scaler_inp = None
        torch.cuda.empty_cache()


_FLAP_METRICS = {
    "IFV": lambda wrapped, subset, name: wrapped[name].fluc_inp,
    "WIFV": lambda wrapped, subset, name: wrapped[name].fluc_inp
    * torch.sum(subset[name].weight.data.pow(2), dim=0),
    "WIFN": lambda wrapped, subset, name: (
        torch.abs(subset[name].weight.data)
        * torch.sqrt(wrapped[name].scaler_inp.reshape((1, -1)))
    ).mean(axis=0),
}


def _resolve_dataset_name(dataset_name, scoring_method: str) -> str:
    default = "c4" if scoring_method == "wanda-sp" else "wikitext2"
    if isinstance(dataset_name, str):
        name = dataset_name.lower()
        if "wikitext" in name:
            return "wikitext2"
        if "c4" in name:
            return "c4"
        return default
    if isinstance(dataset_name, dict):
        path = str(dataset_name.get("path", "")).lower()
        if "wikitext" in path:
            return "wikitext2"
        if "c4" in path:
            return "c4"
        return default
    if isinstance(dataset_name, Sequence):
        for item in dataset_name:
            resolved = _resolve_dataset_name(item, scoring_method)
            if resolved in {"wikitext2", "c4"}:
                return resolved
    return default


def _build_prompt(entry) -> Optional[str]:
    if isinstance(entry, str):
        return entry.strip() or None
    if not isinstance(entry, dict):
        return None

    text = entry.get("text") or entry.get("content")
    if isinstance(text, str) and text.strip():
        return text.strip()

    instruction = entry.get("instruction")
    input_text = entry.get("input")
    if isinstance(instruction, str) and instruction.strip():
        prompt = instruction.strip()
        if isinstance(input_text, str) and input_text.strip():
            prompt += "\n" + input_text.strip()
        return prompt

    for value in entry.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
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
        prompt = _build_prompt(entry)
        if prompt is None and not isinstance(entry, (str, dict)):
            raise ValueError(
                f"Unsupported prompt entry type in {path} at index {idx}: {type(entry)}"
            )
        if prompt:
            prompts.append(prompt)
    return prompts


def _local_prompt_files(dataset_name) -> List[Path]:
    items = (
        dataset_name
        if isinstance(dataset_name, Sequence)
        and not isinstance(dataset_name, (str, bytes, dict))
        else [dataset_name]
    )
    files: List[Path] = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("path")
        if item is None:
            continue
        path = Path(str(item))
        if path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return files


def _infer_model_seqlen(model, prompt_max_length: Optional[int]) -> int:
    seqlen = getattr(model, "seqlen", None)
    if not isinstance(seqlen, int) or seqlen <= 0:
        seqlen = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(seqlen, int) or seqlen <= 0:
        seqlen = 2048
    if prompt_max_length and prompt_max_length > 0:
        seqlen = min(seqlen, int(prompt_max_length))
    return max(8, seqlen)


def _get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def _get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        "allenai/c4",
        "default",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
    )
    valdata = load_dataset(
        "allenai/c4",
        "default",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    return trainloader, valenc


def _get_loaders(name="wikitext2", nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "wikitext2" in name:
        return _get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return _get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unsupported structured-pruning calibration dataset '{name}'.")


def _get_prompt_file_loaders(
    prompt_files: Sequence[Path], nsamples: int, seed: int, seqlen: int, tokenizer
):
    prompts: List[str] = []
    per_file_cap = max(
        1, (int(nsamples) + len(prompt_files) - 1) // max(1, len(prompt_files))
    )
    for prompt_file in prompt_files:
        prompts.extend(_load_prompt_json_file(prompt_file)[:per_file_cap])
    if not prompts:
        raise ValueError("No usable prompts found in local calibration prompt files.")

    random.seed(seed)
    if len(prompts) > nsamples:
        prompts = random.sample(prompts, nsamples)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )

    encoded = tokenizer(
        "\n\n".join(prompts),
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    if encoded.numel() == 0:
        raise ValueError("Local calibration prompt files produced no tokens.")

    chunks = []
    cursor = 0
    total_tokens = encoded.shape[1]
    while len(chunks) < nsamples and cursor < total_tokens:
        chunk = encoded[:, cursor : cursor + seqlen]
        attention_mask = torch.ones_like(chunk)
        if chunk.shape[1] < seqlen:
            pad_width = seqlen - chunk.shape[1]
            padding = torch.full((1, pad_width), int(pad_token_id), dtype=chunk.dtype)
            chunk = torch.cat([chunk, padding], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros((1, pad_width), dtype=attention_mask.dtype),
                ],
                dim=1,
            )
        target = chunk.clone()
        target[:, :-1] = -100
        chunks.append((chunk, target, attention_mask))
        cursor += seqlen

    while len(chunks) < nsamples:
        chunk, target, attention_mask = chunks[len(chunks) % max(1, len(chunks))]
        chunks.append((chunk.clone(), target.clone(), attention_mask.clone()))
    return chunks, None


def _prepare_calibration_input(model, dataloader, device):
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, "hf_device_map", {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader)
    seqlen = dataloader[0][0].shape[1]
    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            kwargs = {}
            if len(batch) > 2 and batch[2] is not None:
                kwargs["attention_mask"] = batch[2].to(device)
            model(batch[0].to(device), **kwargs)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids


def _forward_layer(layer, hidden_states, attention_mask, position_ids):
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return layer(hidden_states, **kwargs)[0]


def _maybe_to_device(tensor, device):
    if tensor is None:
        return None
    return tensor.to(device)


def _select_attention_heads_to_prune(
    attn_layer,
    head_scores: torch.Tensor,
    layer_pruning_ratio: float,
    criterion: str,
    select_indices_fn: Callable,
) -> List[int]:
    """Select valid attention heads, pruning whole KV groups for GQA models."""
    scores = head_scores.detach().float().cpu().numpy()
    num_heads = int(scores.size)
    if num_heads <= 0:
        return []

    kv_heads = int(getattr(attn_layer, "num_key_value_heads", num_heads) or num_heads)
    group_size = getattr(attn_layer, "num_key_value_groups", None)
    is_gqa = kv_heads > 0 and kv_heads < num_heads
    if not is_gqa:
        num_prune_heads = int(round(layer_pruning_ratio * num_heads))
        num_prune_heads = max(0, min(num_prune_heads, num_heads - 1))
        return select_indices_fn(scores, num_prune_heads, criterion)

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
    groups_to_prune = select_indices_fn(grouped_scores, num_prune_groups, criterion)
    return [
        int(group * int(group_size) + offset)
        for group in groups_to_prune
        for offset in range(int(group_size))
    ]


def apply_progressive_structured_pruning(
    model,
    tokenizer,
    *,
    scoring_method: str,
    dataset_name,
    max_prompts: int,
    prompt_max_length: Optional[int],
    seed: int,
    criterion: str,
    layer_keep_ratios,
    max_ratio: float,
    heads_by_layer: Sequence[bool],
    mlp_by_layer: Sequence[bool],
    flap_metric: str,
    select_indices_fn: Callable,
    prune_attention_fn: Callable,
    prune_mlp_fn: Callable,
) -> List[Dict[str, int]]:
    if scoring_method not in {"wanda-sp", "flap"}:
        raise ValueError(f"Unsupported structured scoring method '{scoring_method}'.")
    if scoring_method == "flap" and flap_metric not in _FLAP_METRICS:
        raise ValueError(f"Unsupported FLAP metric '{flap_metric}'.")

    seqlen = _infer_model_seqlen(model, prompt_max_length)
    nsamples = max(1, int(max_prompts))
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    first_param = next(model.parameters())
    default_device = first_param.device
    prompt_files = _local_prompt_files(dataset_name)
    if prompt_files:
        logging.info(
            "Using %d local calibration prompt file(s) for %s scoring.",
            len(prompt_files),
            scoring_method,
        )
        dataloader, _ = _get_prompt_file_loaders(
            prompt_files,
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            tokenizer=tokenizer,
        )
    else:
        calib_name = _resolve_dataset_name(dataset_name, scoring_method)
        dataloader, _ = _get_loaders(
            calib_name, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer
        )
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = _prepare_calibration_input(
            model, dataloader, default_device
        )

    layers = model.model.layers
    layer_stats: List[Dict[str, int]] = []
    for layer_idx, layer in enumerate(layers):
        subset = {
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.down_proj": getattr(
                layer.mlp, "down_proj", getattr(layer.mlp, "fc2", None)
            ),
        }
        subset = {name: module for name, module in subset.items() if module is not None}

        if f"model.layers.{layer_idx}" in getattr(model, "hf_device_map", {}):
            dev = model.hf_device_map[f"model.layers.{layer_idx}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = _maybe_to_device(attention_mask, dev)
            position_ids = _maybe_to_device(position_ids, dev)
        else:
            dev = default_device

        wrapped_layers = {}
        for name in subset:
            if scoring_method == "wanda-sp":
                wrapped_layers[name] = WrappedGPT(subset[name])
            else:
                wrapped_layers[name] = BiasGPT(subset[name], flap_metric)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = [
            subset[name].register_forward_hook(add_batch(name))
            for name in wrapped_layers
        ]
        for sample_idx in range(nsamples):
            with torch.no_grad():
                outs[sample_idx] = _forward_layer(
                    layer,
                    inps[sample_idx].unsqueeze(0),
                    attention_mask,
                    position_ids,
                )
        for handle in handles:
            handle.remove()

        layer_keep_ratio = max_ratio
        if layer_keep_ratios is not None:
            if isinstance(layer_keep_ratios, dict):
                layer_keep_ratio = layer_keep_ratios.get(layer_idx, max_ratio)
            elif layer_idx < len(layer_keep_ratios):
                layer_keep_ratio = layer_keep_ratios[layer_idx]
            if layer_keep_ratio is None:
                layer_keep_ratio = max_ratio
        layer_pruning_ratio = max(0.0, min(1.0, 1.0 - float(layer_keep_ratio)))

        head_dim = getattr(layer.self_attn, "head_dim", None)
        if not isinstance(head_dim, int) or head_dim <= 0:
            head_dim = subset["self_attn.o_proj"].weight.shape[1] // max(
                1, layer.self_attn.num_heads
            )

        num_heads = 0
        intermediate_size = 0
        num_prune_heads = 0
        num_prune_mlp_units = 0
        heads_to_prune = []
        mlp_to_prune = []
        if heads_by_layer[layer_idx] and "self_attn.o_proj" in subset:
            if scoring_method == "wanda-sp":
                head_metric = torch.abs(
                    subset["self_attn.o_proj"].weight.data
                ) * torch.sqrt(
                    wrapped_layers["self_attn.o_proj"].scaler_row.reshape((1, -1))
                )
                head_scores = head_metric.mean(dim=0).reshape(-1, head_dim).sum(dim=1)
            else:
                head_metric = _FLAP_METRICS[flap_metric](
                    wrapped_layers, subset, "self_attn.o_proj"
                ).pow(2)
                head_scores = head_metric.reshape(-1, head_dim).sum(dim=1)
            if not torch.isfinite(head_scores).all():
                logging.warning(
                    "Non-finite head scores detected in layer %d during %s scoring; sanitizing before selection.",
                    layer_idx,
                    scoring_method,
                )
                head_scores = torch.nan_to_num(
                    head_scores, nan=0.0, posinf=0.0, neginf=0.0
                )
            num_heads = int(head_scores.numel())
            heads_to_prune = _select_attention_heads_to_prune(
                layer.self_attn,
                head_scores,
                layer_pruning_ratio,
                criterion,
                select_indices_fn,
            )
            num_prune_heads = len(heads_to_prune)
            prune_attention_fn(layer.self_attn, list(heads_to_prune))

        if mlp_by_layer[layer_idx] and "mlp.down_proj" in subset:
            if scoring_method == "wanda-sp":
                mlp_metric = torch.abs(
                    subset["mlp.down_proj"].weight.data
                ) * torch.sqrt(
                    wrapped_layers["mlp.down_proj"].scaler_row.reshape((1, -1))
                )
                mlp_scores = mlp_metric.mean(dim=0)
            else:
                mlp_scores = _FLAP_METRICS[flap_metric](
                    wrapped_layers, subset, "mlp.down_proj"
                )
            if not torch.isfinite(mlp_scores).all():
                logging.warning(
                    "Non-finite MLP scores detected in layer %d during %s scoring; sanitizing before selection.",
                    layer_idx,
                    scoring_method,
                )
                mlp_scores = torch.nan_to_num(
                    mlp_scores, nan=0.0, posinf=0.0, neginf=0.0
                )
            intermediate_size = int(mlp_scores.numel())
            num_prune_mlp_units = int(round(layer_pruning_ratio * intermediate_size))
            num_prune_mlp_units = max(
                0, min(num_prune_mlp_units, intermediate_size - 1)
            )
            mlp_to_prune = select_indices_fn(
                mlp_scores.cpu().numpy(), num_prune_mlp_units, criterion
            )
            prune_mlp_fn(layer.mlp, list(mlp_to_prune))

        layer_stats.append(
            {
                "layer_idx": int(layer_idx),
                "heads_pruned": int(num_prune_heads > 0),
                "mlp_pruned": int(num_prune_mlp_units > 0),
                "total_heads": int(num_heads),
                "total_mlp_units": int(intermediate_size),
                "num_pruned_heads": int(num_prune_heads),
                "num_pruned_mlp_units": int(num_prune_mlp_units),
                "pruned_head_indices": [int(index) for index in heads_to_prune],
                "pruned_mlp_indices": [int(index) for index in mlp_to_prune],
            }
        )

        for wrapped in wrapped_layers.values():
            wrapped.free()

        subset_after = {
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.down_proj": getattr(
                layer.mlp, "down_proj", getattr(layer.mlp, "fc2", None)
            ),
        }
        subset_after = {
            name: module for name, module in subset_after.items() if module is not None
        }
        for sample_idx in range(nsamples):
            with torch.no_grad():
                outs[sample_idx] = _forward_layer(
                    layer,
                    inps[sample_idx].unsqueeze(0),
                    attention_mask,
                    position_ids,
                )
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return layer_stats
