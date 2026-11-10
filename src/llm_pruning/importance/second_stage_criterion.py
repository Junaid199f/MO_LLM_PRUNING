import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Global dictionaries to store activations and head importance.
# List to track hook handles, used by remove_all_hooks().
layer_activations = defaultdict(list)
head_importance = defaultdict(list)
hook_handles = []

_IMPORTANCE_METRICS = {"l1", "l2", "mean", "abs_mean"}


def _validate_importance_metric(metric: str) -> str:
    metric = (metric or "").lower()
    if metric not in _IMPORTANCE_METRICS:
        raise ValueError(
            f"Unsupported importance metric '{metric}'. Choose from {_IMPORTANCE_METRICS}."
        )
    return metric


def _reduce_importance(values: torch.Tensor, metric: str, reduce_dims) -> torch.Tensor:
    metric = _validate_importance_metric(metric)
    if metric == "l1":
        return values.abs().sum(dim=reduce_dims)
    if metric == "l2":
        return torch.sqrt(values.pow(2).mean(dim=reduce_dims))
    if metric == "mean":
        return values.mean(dim=reduce_dims)
    return values.abs().mean(dim=reduce_dims)


# Architecture settings
ARCHITECTURE_SETTINGS = {
    "llama": {
        "head_dim": 128,
        "attn_proj": "o_proj",  # Attention: layer.self_attn.o_proj
        "mlp_proj": "down_proj",  # MLP: layer.mlp.down_proj
    },
    "mistral": {
        "head_dim": 128,
        "attn_proj": "o_proj",  # Attention: layer.self_attn.o_proj
        "mlp_proj": "down_proj",  # MLP: layer.mlp.down_proj
    },
    "qwen2": {
        "head_dim": None,
        "attn_proj": "o_proj",  # Attention: layer.self_attn.o_proj
        "mlp_proj": "down_proj",  # MLP: layer.mlp.down_proj
    },
    "phi3": {
        "head_dim": None,
        "attn_proj": "o_proj",  # Attention: layer.self_attn.o_proj
        "mlp_proj": "down_proj",  # MLP: layer.mlp.down_proj
    },
    "phi4": {
        "head_dim": None,
        "attn_proj": "o_proj",  # Attention: layer.self_attn.o_proj
        "mlp_proj": "down_proj",  # MLP: layer.mlp.down_proj
    },
}


class _CacheGuard:
    """Temporarily toggle `use_cache` on the model and its generation config."""

    def __init__(self, model, enabled: bool):
        self.model = model
        self.enabled = enabled
        self.prev_model = getattr(model.config, "use_cache", None)
        self.prev_gen = getattr(
            getattr(model, "generation_config", None), "use_cache", None
        )

    def __enter__(self):
        self._set_attr(self.model.config, "use_cache", self.enabled)
        if hasattr(self.model, "generation_config"):
            self._set_attr(self.model.generation_config, "use_cache", self.enabled)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._restore(self.model.config, "use_cache", self.prev_model)
        if hasattr(self.model, "generation_config"):
            self._restore(self.model.generation_config, "use_cache", self.prev_gen)

    @staticmethod
    def _set_attr(obj, attr: str, value):
        if obj is not None:
            setattr(obj, attr, value)

    @staticmethod
    def _restore(obj, attr: str, value):
        if obj is None:
            return
        if value is None:
            if hasattr(obj, attr):
                delattr(obj, attr)
        else:
            setattr(obj, attr, value)


def store_head_importance(layer_number, num_heads, head_dim, importance_metric: str):
    """
    Returns a hook function that computes and stores head importance for a given layer.
    """

    def hook_fn(module, input, output):
        # input[0]: (batch_size, seq_length, hidden_dim)
        inp = input[0].float()
        batch_size, seq_length, _ = inp.shape
        hidden_dim = module.weight.shape[0]

        # Reshape input to (batch_size * seq_length, num_heads, head_dim)
        reshaped_inp = inp.view(-1, num_heads, head_dim)
        # Reshape weight matrix to (num_heads, head_dim, hidden_dim)
        weight = module.weight.T.view(
            num_heads, head_dim, hidden_dim
        ).float()  # .float() - FP16 can lead to overflow when calculating the average  # noqa
        # Compute per-head output contributions
        output_per_head = torch.einsum("bhd,hdo->bho", reshaped_inp, weight)
        # Calculate importance per head using the selected metric.
        head_imp = _reduce_importance(
            output_per_head, importance_metric, reduce_dims=(0, 2)
        )
        if importance_metric == "l1" and head_imp.numel() == num_heads:
            head_imp = head_imp / (batch_size * seq_length)

        if torch.isinf(head_imp).any():
            warnings.warn(f"Infinite values detected in layer {layer_number}.")

        head_importance[layer_number].append(head_imp.detach().cpu())

    return hook_fn


def register_attention_hooks(model, arch, importance_metric: str):
    """
    Registers hooks on the attention projection layers based on the architecture.
    For example, for "llama" it uses the 'o_proj' attribute and for "phi" it uses 'dense'.
    """
    settings = ARCHITECTURE_SETTINGS.get(arch)
    if settings is None:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    attn_proj_attr = settings["attn_proj"]

    global hook_handles
    hook_handles = []  # Clear any existing hooks
    for layer_number, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, attn_proj_attr):
            proj = getattr(layer.self_attn, attn_proj_attr)
        else:
            raise AttributeError(
                f"Layer {layer_number} self_attn does not have expected projection attribute "
                f"'{attn_proj_attr}' for architecture '{arch}'."
            )

        layer_num_heads = getattr(layer.self_attn, "num_heads", None)
        if not layer_num_heads:
            raise ValueError(f"Layer {layer_number} is missing num_heads information.")

        layer_head_dim = getattr(layer.self_attn, "head_dim", None)
        if not layer_head_dim or layer_head_dim * layer_num_heads != proj.in_features:
            hidden_dim = (
                getattr(layer.self_attn, "hidden_size", None) or proj.in_features
            )
            if hidden_dim % layer_num_heads != 0:
                raise ValueError(
                    f"Layer {layer_number} hidden dim {hidden_dim} is not divisible by {layer_num_heads} heads."
                )
            layer_head_dim = hidden_dim // layer_num_heads

        handle = proj.register_forward_hook(
            store_head_importance(
                layer_number, layer_num_heads, layer_head_dim, importance_metric
            )
        )
        hook_handles.append(handle)


def _truncate_prompt(tokenizer, prompt, prompt_max_length):
    if prompt_max_length is None or prompt_max_length <= 0:
        return prompt
    tokenized = tokenizer(
        prompt,
        max_length=prompt_max_length,
        truncation=True,
        add_special_tokens=False,
    )
    input_ids = tokenized.get("input_ids")
    if not input_ids:
        return prompt
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def generate_response(
    pipeline_obj, prompt, max_new_tokens=1, temperature=0.7, top_p=0.9
):
    """
    Generates a response using the given Hugging Face pipeline.
    """
    response = pipeline_obj(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=pipeline_obj.tokenizer.eos_token_id,
        pad_token_id=pipeline_obj.tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=1,
    )
    return response[0]["generated_text"]


def _build_prompt(entry):
    text = entry.get("text") or entry.get("content")
    if isinstance(text, str) and text.strip():
        return text

    instruction = entry.get("instruction")
    input_text = entry.get("input")
    if isinstance(instruction, str) and instruction.strip():
        prompt = instruction.strip()
        if isinstance(input_text, str) and input_text.strip():
            prompt += "\n" + input_text.strip()
        return f"Instruction: {prompt}\nResponse:"

    for value in entry.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _load_prompt_json_file(prompt_file):
    path = Path(prompt_file)
    if not path.is_file() or path.suffix.lower() != ".json":
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Prompt file must contain a list of prompt objects: {path}")

    prompts = []
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


def _normalize_dataset_specs(dataset_name, split):
    if isinstance(dataset_name, str):
        return [{"path": dataset_name, "split": split}]

    specs = []
    if isinstance(dataset_name, dict):
        dataset_name = [dataset_name]

    for item in dataset_name:
        if isinstance(item, str):
            specs.append({"path": item, "split": split})
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
            spec.setdefault("split", split)
            specs.append(spec)
            continue
        raise TypeError(f"Unsupported dataset spec type: {type(item)}")
    return specs


def load_prompts_from_dataset(
    dataset_name="yahma/alpaca-cleaned",
    split="train",
    max_prompts=None,
    random_subset=False,
):
    """
    Loads and returns a list of prompts from a Hugging Face dataset.
    """
    specs = _normalize_dataset_specs(dataset_name, split)
    per_spec_cap = None
    if max_prompts is not None and len(specs) > 1:
        per_spec_cap = max(1, (int(max_prompts) + len(specs) - 1) // len(specs))
    prompts = []
    for spec in specs:
        path = spec["path"]
        prompt_file = Path(str(path))
        if prompt_file.is_file() and prompt_file.suffix.lower() == ".json":
            file_prompts = _load_prompt_json_file(prompt_file)
            if per_spec_cap is not None and len(file_prompts) > per_spec_cap:
                file_prompts = (
                    random.sample(file_prompts, per_spec_cap)
                    if random_subset
                    else file_prompts[:per_spec_cap]
                )
            prompts.extend(file_prompts)
            continue
        name = spec.get("config")
        spec_split = spec.get("split", split)
        if name is None:
            dataset = load_dataset(path, split=spec_split)
        else:
            dataset = load_dataset(path, name=name, split=spec_split)
        for entry in dataset:
            prompt = _build_prompt(entry)
            if prompt:
                prompts.append(prompt)

    if max_prompts is not None and max_prompts < len(prompts):
        prompts = (
            random.sample(prompts, max_prompts)
            if random_subset
            else prompts[:max_prompts]
        )
    return prompts


def compute_average_head_importance():
    """
    Computes the average head importance for each layer over all samples.
    """
    layer_means = {}
    for layer, values in head_importance.items():
        vals = torch.stack(values, dim=0)
        layer_mean = vals.mean(dim=0)
        layer_means[layer] = layer_mean.numpy()
    return layer_means


def measure_second_stage_heads_importance(
    model,
    tokenizer,
    arch,
    dataset_name="yahma/alpaca-cleaned",
    split="train",
    max_prompts=None,
    random_subset=False,
    prompt_max_length=None,
    importance_metric: str = "l1",
):
    """
    Measures attention head importance using the second-stage method.

    Args:
        model: The transformer model.
        tokenizer: The corresponding tokenizer.
        arch: Model architecture ('llama') determining hook details.
        dataset_name: Dataset name for loading prompts.
        split: Dataset split.
        max_prompts: Maximum number of prompts to process.
        random_subset: Whether to randomly sample prompts.
        prompt_max_length: Optional max prompt length (in tokens) for truncation.

    Returns:
        A dict mapping each layer to its average head importance vector.
    """
    num_heads = model.config.num_attention_heads
    settings = ARCHITECTURE_SETTINGS.get(arch)
    if settings is None:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    head_dim_default = settings.get("head_dim")
    inferred_head_dim = getattr(
        model.model.layers[0].self_attn, "head_dim", head_dim_default
    )
    print(
        f"Model has {num_heads} heads with a head dimension of {inferred_head_dim}.\n"
    )
    importance_metric = _validate_importance_metric(importance_metric)
    register_attention_hooks(model, arch, importance_metric)

    prompts = load_prompts_from_dataset(
        dataset_name=dataset_name,
        split=split,
        max_prompts=max_prompts,
        random_subset=random_subset,
    )
    print(f"Number of prompts to process: {len(prompts)}\n")

    with _CacheGuard(model, False):
        print("Initializing text generation pipeline...")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Pipeline initialized successfully.\n")

        print("Generating responses and collecting head importance...\n")
        for prompt in tqdm(prompts, desc="Processing prompts"):
            prompt = _truncate_prompt(tokenizer, prompt, prompt_max_length)
            _ = generate_response(generator, prompt)

    print("\nResponse generation and head importance collection completed.\n")

    mean_importances = compute_average_head_importance()
    for layer in sorted(mean_importances.keys()):
        importance = mean_importances[layer]
        importance_str = ", ".join([f"{imp:.2f}" for imp in importance])
        print(f"Layer {layer} mean head importance: [{importance_str}]")

    remove_all_hooks()

    return mean_importances


def store_second_stage_mlp_input_average(layer_number, importance_metric: str):
    """
    Returns a hook function that stores the average (absolute) activation for an MLP layer.
    """

    def hook_fn(module, input, output):
        activations = input[0].float()
        abs_activations = activations.abs()
        if importance_metric == "l1":
            scores = abs_activations.sum(dim=(0, 1))
        elif importance_metric == "l2":
            scores = torch.sqrt(activations.pow(2).mean(dim=(0, 1)))
        elif importance_metric == "mean":
            scores = activations.mean(dim=(0, 1))
        else:
            scores = abs_activations.mean(dim=(0, 1))
        layer_activations[layer_number].append(scores.detach().cpu().numpy())

    return hook_fn


def register_mlp_hooks(model, arch, importance_metric: str):
    """
    Registers hooks on the MLP projection layers based on the architecture.
    """
    settings = ARCHITECTURE_SETTINGS.get(arch)
    if settings is None:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    mlp_proj_attr = settings["mlp_proj"]

    global hook_handles
    for layer_number, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, mlp_proj_attr):
            proj = getattr(mlp, mlp_proj_attr)
        else:
            raise AttributeError(
                f"MLP layer {layer_number} does not have expected projection attribute "
                f"'{mlp_proj_attr}' for architecture '{arch}'."
            )
        handle = proj.register_forward_hook(
            store_second_stage_mlp_input_average(layer_number, importance_metric)
        )
        hook_handles.append(handle)


def compute_average_neuron_importance():
    """
    Computes the average neuron importance per MLP layer over all samples.
    """
    combined_importance = {}
    for layer, importance_list in layer_activations.items():
        stacked = np.stack(importance_list, axis=0)
        mean_importance = stacked.mean(axis=0)
        combined_importance[layer] = mean_importance.tolist()
    return combined_importance


def measure_second_stage_mlps_importance(
    model,
    tokenizer,
    arch,
    dataset_name="yahma/alpaca-cleaned",
    split="train",
    max_prompts=None,
    random_subset=False,
    prompt_max_length=None,
    importance_metric: str = "l1",
):
    """
    Measures MLP neuron importance using the second-stage method.

    Args:
        model: The transformer model.
        tokenizer: The corresponding tokenizer.
        arch: Model architecture ('llama' or 'phi') determining hook details.
        dataset_name: Dataset name for prompts.
        split: Dataset split.
        max_prompts: Maximum number of prompts to process.
        random_subset: Whether to randomly sample prompts.
        prompt_max_length: Optional max prompt length (in tokens) for truncation.

    Returns:
        A dict mapping each layer to its average neuron importance vector.
    """
    importance_metric = _validate_importance_metric(importance_metric)
    register_mlp_hooks(model, arch, importance_metric)

    prompts = load_prompts_from_dataset(
        dataset_name=dataset_name,
        split=split,
        max_prompts=max_prompts,
        random_subset=random_subset,
    )
    print(f"Number of prompts to process: {len(prompts)}\n")

    with _CacheGuard(model, False):
        print("Initializing text generation pipeline...")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Pipeline initialized successfully.\n")

        print("Generating responses and collecting activations...\n")
        for prompt in tqdm(prompts, desc="Processing prompts"):
            prompt = _truncate_prompt(tokenizer, prompt, prompt_max_length)
            _ = generate_response(generator, prompt)

    print("\nResponse generation and activation collection completed.\n")

    print("Computing average neuron importance...\n")
    mean_importances = compute_average_neuron_importance()
    for layer in sorted(mean_importances.keys()):
        importance = mean_importances[layer]
        importance_str = ", ".join([f"{imp:.4f}" for imp in importance[:10]])
        print(
            f"Layer {layer} mean neuron importance (first 10 neurons): [{importance_str}]"
        )

    remove_all_hooks()

    # Clear the global dictionaries for subsequent runs.
    head_importance.clear()
    layer_activations.clear()

    return mean_importances


def remove_all_hooks():
    """
    Removes all registered hooks.
    """
    global hook_handles
    for handle in hook_handles:
        handle.remove()
    hook_handles = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure second-stage heads and MLP neurons importance"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Pretrained model name or path"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=list(ARCHITECTURE_SETTINGS.keys()),
        required=True,
        help="Model architecture (e.g., 'llama' or 'phi')",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["heads", "mlps"],
        required=True,
        help="Task: measure 'heads' or 'mlps' importance",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yahma/alpaca-cleaned",
        help="Dataset to use for prompts",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process",
    )
    parser.add_argument(
        "--random_subset",
        action="store_true",
        help="Randomly sample prompts if max_prompts is set",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda:0")

    if args.task == "heads":
        measure_second_stage_heads_importance(
            model,
            tokenizer,
            args.arch,
            dataset_name=args.dataset,
            split=args.split,
            max_prompts=args.max_prompts,
            random_subset=args.random_subset,
        )
    elif args.task == "mlps":
        measure_second_stage_mlps_importance(
            model,
            tokenizer,
            args.arch,
            dataset_name=args.dataset,
            split=args.split,
            max_prompts=args.max_prompts,
            random_subset=args.random_subset,
        )
