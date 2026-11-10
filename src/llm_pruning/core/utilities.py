"""Model loading, masks, parameter counts, and calibration sampling."""

import random
import logging
import numpy as np
import torch
from types import MethodType
from transformers import AutoModelForCausalLM


def get_model_layers(model):
    """Return the main transformer block list across common HF architectures."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            return inner.decoder.layers
        if hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
            return inner.transformer.h
        if hasattr(inner, "gpt_neox") and hasattr(inner.gpt_neox, "layers"):
            return inner.gpt_neox.layers
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError(
        f"Unsupported model architecture: cannot locate transformer layers on {type(model).__name__}"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    logging.info(f"Seed for reproducibility: {seed}")


def printModelStats(model, model_type):
    model_total_params = sum(p.numel() for p in model.parameters())
    main_model_total_params = sum(
        p.numel() for p in get_model_layers(model).parameters()
    )
    logging.info(f"[{model_type}] Full number of parameters = {model_total_params}")
    logging.info(
        f"[{model_type}] Main model number of parameters = {main_model_total_params}"
    )


def get_parameter_counts(model):
    """Return total parameter count and how many remain non-zero."""
    total_params = 0
    nonzero_params = 0
    with torch.no_grad():
        for param in model.parameters():
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()
    return total_params, nonzero_params


# Backward-compat alias for older imports


def loadModel(model_name, cache_dir=None, device_map="auto"):
    logging.info("Loading the model")

    if (
        "llama" in model_name.lower()
        or "phi-3" in model_name.lower()
        or "phi-4" in model_name.lower()
        or "qwen2" in model_name.lower()
    ):
        dtype = torch.float32
    else:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    return model


def remove_accelerate_hooks_if_present(model):
    """Detach Accelerate hooks that still expect the original tensor shapes."""
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        return model

    hooks_present = hasattr(model, "_hf_hook")
    if not hooks_present:
        for module in model.modules():
            if hasattr(module, "_hf_hook"):
                hooks_present = True
                break

    if not hooks_present:
        return model

    remove_hook_from_module(model, recurse=True)

    if hasattr(model, "hf_device_map"):
        try:
            delattr(model, "hf_device_map")
        except AttributeError:
            model.hf_device_map = None

    return model


def maskModel(model, attnMask, mlpMask):
    for i in range(len(attnMask)):
        if attnMask[i] == 1 and mlpMask[i] == 1:  # mask the entire block

            def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                # Mirror the tuple structure expected by HF blocks so cache handling does not break.
                output_attentions = kwargs.get("output_attentions", False)
                use_cache = kwargs.get("use_cache", False)
                outputs = (hidden_states,)
                if output_attentions:
                    outputs += (None,)
                if use_cache:
                    outputs += (None,)
                return outputs

            get_model_layers(model)[i].forward_bak = get_model_layers(model)[i].forward
            get_model_layers(model)[i].forward = MethodType(
                identity_forward, get_model_layers(model)[i]
            )

        elif attnMask[i] == 1 and mlpMask[i] == 0:  # mask the attention only
            if model.config.model_type in ("phi", "phi3", "phi4"):

                def identity_forward(
                    self, hidden_states: torch.Tensor, *args, **kwargs
                ):
                    return torch.zeros_like(hidden_states), None, None

            else:

                def identity_forward(
                    self, hidden_states: torch.Tensor, *args, **kwargs
                ):
                    return 0, None, None

            get_model_layers(model)[i].self_attn.forward_bak = get_model_layers(model)[
                i
            ].self_attn.forward
            get_model_layers(model)[i].self_attn.forward = MethodType(
                identity_forward, get_model_layers(model)[i].self_attn
            )

        elif attnMask[i] == 0 and mlpMask[i] == 1:  # mask the mlp only
            if model.config.model_type in ("phi", "phi3", "phi4"):

                def identity_forward(
                    self, hidden_states: torch.Tensor, *args, **kwargs
                ):
                    return torch.zeros_like(hidden_states)

            else:

                def identity_forward(
                    self, hidden_states: torch.Tensor, *args, **kwargs
                ):
                    return 0

            get_model_layers(model)[i].mlp.forward_bak = get_model_layers(model)[
                i
            ].mlp.forward
            get_model_layers(model)[i].mlp.forward = MethodType(
                identity_forward, get_model_layers(model)[i].mlp
            )


def unmaskModel(model, attnMask, mlpMask):
    for i in range(len(attnMask)):
        if attnMask[i] == 1 and mlpMask[i] == 1:  # unmask the entire block
            block = get_model_layers(model)[i]
            if hasattr(block, "forward_bak"):
                block.forward = block.forward_bak
                delattr(block, "forward_bak")

        elif attnMask[i] == 1 and mlpMask[i] == 0:  # unmask attention only
            attn = get_model_layers(model)[i].self_attn
            if hasattr(attn, "forward_bak"):
                attn.forward = attn.forward_bak
                delattr(attn, "forward_bak")

        elif attnMask[i] == 0 and mlpMask[i] == 1:  # unmask FFN only
            mlp = get_model_layers(model)[i].mlp
            if hasattr(mlp, "forward_bak"):
                mlp.forward = mlp.forward_bak
                delattr(mlp, "forward_bak")


def count_params_with_masks(model, attn_mask, mlp_mask):
    """Return total parameters and the subset left active after masking."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for idx, block in enumerate(get_model_layers(model)):
        if attn_mask[idx] and mlp_mask[idx]:
            inactive += sum(p.numel() for p in block.parameters())
            continue
        if attn_mask[idx]:
            inactive += sum(p.numel() for p in block.self_attn.parameters())
        if mlp_mask[idx]:
            inactive += sum(p.numel() for p in block.mlp.parameters())
    return total, total - inactive


def get_calibration(dataset, tokenizer, num_samples, seq_len=2048, seed=0):
    samples_indices = list(range(len(dataset)))
    if seed != 0:
        random.shuffle(samples_indices)

    samples = dataset.select(samples_indices)
    input_ids = tokenizer(
        "\n\n".join(samples["text"]), return_tensors="pt", add_special_tokens=False
    ).input_ids

    # Split the calibration into num_samples of seq_len length
    calibration = []
    for i in range(num_samples):
        calibration.append(input_ids[:, i * seq_len : (i + 1) * seq_len])

    return calibration
