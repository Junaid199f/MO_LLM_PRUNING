import argparse
import json
import logging
import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from llm_pruning.refinement.second_stage import apply_saved_second_stage_pruning
from llm_pruning.core.utilities import (
    loadModel,
    maskModel,
    remove_accelerate_hooks_if_present,
)


def fine_tune_model(
    pruned_model_dir: str = "pruned_model",
    tuned_model_dir: str = "tuned_model",
    arch: str = "llama",
    num_train_epochs: float = 2,
    save_steps: int = 200,
    logging_steps: int = 10,
    num_samples: int = None,
    target_modules: list = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
    warmup_ratio: float = 0.03,
    optimizer: str = "adamw_torch",
    report_to: str = None,
    model=None,
    tokenizer=None,
):
    """
    Fine-tune a model using LoRA.
    Parameters:
      - pruned_model_dir (str): Directory containing the pruned model and tokenizer.
      - tuned_model_dir (str): Directory where the fine-tuned model will be saved.
      - arch (str): Model architecture, currently only "llama" is supported.
      - num_train_epochs (float): Number of training epochs.
      - save_steps (int): Frequency (in steps) at which checkpoints are saved.
      - logging_steps (int): Frequency (in steps) for logging training progress.
      - num_samples (int): If provided, limits the training dataset to the first num_samples examples.
      - target_modules (list): Optional list of target modules for LoRA. If not provided, defaults based on `arch`.
      - lora_rank (int): LoRA rank (r).
      - lora_alpha (int): LoRA alpha scaling.
      - lora_dropout (float): LoRA dropout.
      - batch_size (int): Per-device training batch size.
      - learning_rate (float): Learning rate.
      - warmup_ratio (float): Warmup ratio for the scheduler.
      - optimizer (str): HF Trainer optimizer name (e.g., adamw_torch).
      - report_to (str): Reporting backend (e.g., wandb).
      - model/tokenizer: Optional pre-loaded objects to reuse and avoid reloading from disk.

    Returns:
      - The fine-tuned model.
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if target_modules is None:
        if arch == "llama":
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ]
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def _has_meta_params(mdl):
        return any(getattr(p, "is_meta", False) for p in mdl.parameters())

    def _clear_quantization_metadata(config_obj, *, label: str) -> bool:
        if config_obj is None:
            return False
        removed = False
        if (
            hasattr(config_obj, "quantization_config")
            and getattr(config_obj, "quantization_config", None) is not None
        ):
            logging.warning(
                "Dropping %s quantization_config before recovery LoRA.", label
            )
            config_obj.quantization_config = None
            removed = True
        return removed

    def _sanitize_model_configs(mdl):
        if mdl is None:
            return
        _clear_quantization_metadata(getattr(mdl, "config", None), label="model")
        _clear_quantization_metadata(
            getattr(mdl, "generation_config", None), label="generation"
        )

    def _disable_peft_awq_integration() -> None:
        def _always_false():
            return False

        def _dispatch_none(*args, **kwargs):
            return None

        try:
            import peft.import_utils as peft_import_utils

            if hasattr(peft_import_utils, "is_auto_awq_available"):
                peft_import_utils.is_auto_awq_available = _always_false
        except Exception:
            pass

        try:
            from peft.utils import import_utils as peft_utils_import_utils

            if hasattr(peft_utils_import_utils, "is_auto_awq_available"):
                peft_utils_import_utils.is_auto_awq_available = _always_false
        except Exception:
            pass

        try:
            from peft.tuners.lora import model as peft_lora_model

            if hasattr(peft_lora_model, "is_auto_awq_available"):
                peft_lora_model.is_auto_awq_available = _always_false
            if hasattr(peft_lora_model, "dispatch_awq"):
                peft_lora_model.dispatch_awq = _dispatch_none
        except Exception:
            pass

        try:
            from peft.tuners.lora import awq as peft_lora_awq

            if hasattr(peft_lora_awq, "is_auto_awq_available"):
                peft_lora_awq.is_auto_awq_available = _always_false
            if hasattr(peft_lora_awq, "dispatch_awq"):
                peft_lora_awq.dispatch_awq = _dispatch_none
        except Exception:
            pass

    if model is not None:
        model = remove_accelerate_hooks_if_present(model)
        _sanitize_model_configs(model)
        if _has_meta_params(model):
            logging.warning(
                "Model contains meta tensors; reloading from disk without device_map."
            )
            model = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pruned_model_dir, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        # Reloading structurally pruned checkpoints with vanilla HF config reconstruction
        # is only safe when the saved config still matches the tensor shapes.
        config = AutoConfig.from_pretrained(pruned_model_dir, trust_remote_code=True)
        _clear_quantization_metadata(config, label="saved")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pruned_model_dir,
                config=config,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to reload the saved pruned model for recovery fine-tuning. "
                "For structurally pruned checkpoints with layer-specific shapes, "
                "reuse the in-memory pruned model instead of reloading from disk."
            ) from e
        _sanitize_model_configs(model)

    # Disable cache during training to avoid legacy cache errors in recent Transformers.
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    # Ensure the in-memory pruned model is on one concrete device. Single-process
    # Trainer DataParallel is fragile with PEFT-wrapped/pruned models because
    # replicated inputs can land on cuda:1 while some nested weights remain on cuda:0.
    target_device = (
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device=target_device, dtype=torch.float16)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to the model.
    _disable_peft_awq_integration()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load the Alpaca Cleaned Dataset.
    dataset = load_dataset("yahma/alpaca-cleaned")["train"]

    # If a sample limit is provided, select only the first num_samples examples.
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    # Preprocess the dataset using the tokenizer.
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
    )

    # Training Arguments.
    training_args = TrainingArguments(
        output_dir=tuned_model_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        evaluation_strategy="no",
        optim=optimizer,
        report_to=report_to,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "Multiple GPUs are visible, but recovery fine-tuning reached the single-process "
            "fine_tune_model() path. This would use fragile Trainer DataParallel for the "
            "PEFT/pruned model. Make sure main.py is passing --recovery_finetune_gpus with "
            "at least two GPU ids and that the job is running the updated main.py torchrun path."
        )

    # Initialize Trainer and start training.
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=lambda features: causal_lm_data_collator(features, tokenizer),
    )
    trainer.train()

    # Save adapters directly in distributed recovery runs. A structurally pruned
    # Llama has per-layer shapes that cannot be reconstructed from a single HF
    # config, so downstream workers must replay pruning then load this adapter.
    model_to_save = trainer.model
    if hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module
    if dist.is_available() and dist.is_initialized():
        if trainer.is_world_process_zero():
            os.makedirs(tuned_model_dir, exist_ok=True)
            model_to_save.save_pretrained(tuned_model_dir)
            tokenizer.save_pretrained(tuned_model_dir)
        dist.barrier()
        return model_to_save

    # Merge LoRA weights and unload adapters for in-process recovery.
    model_to_save = model_to_save.merge_and_unload()

    if trainer.is_world_process_zero():
        os.makedirs(tuned_model_dir, exist_ok=True)
        model_to_save.save_pretrained(tuned_model_dir)
        tokenizer.save_pretrained(tuned_model_dir)

    return model_to_save


def causal_lm_data_collator(features, tokenizer):
    """
    Pad variable-length causal-LM examples into a batch.

    Labels must use -100 as the padding value so padded tokens are ignored by
    the loss function.
    """
    label_pad_token_id = -100
    max_length = max(len(feature["input_ids"]) for feature in features)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        input_ids = list(feature["input_ids"])
        attention_mask = list(feature["attention_mask"])
        labels = list(feature["labels"])
        pad_length = max_length - len(input_ids)

        batch["input_ids"].append(input_ids + [pad_token_id] * pad_length)
        batch["attention_mask"].append(attention_mask + [0] * pad_length)
        batch["labels"].append(labels + [label_pad_token_id] * pad_length)

    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def preprocess_function(examples, tokenizer):
    """
    Preprocess a single example from the dataset.
    It creates a prompt from the instruction and input, tokenizes both prompt and output,
    and constructs the corresponding labels.

    Parameters:
      - examples (dict): A single example containing "instruction", "input", and "output".
      - tokenizer: The tokenizer instance used for tokenization.

    Returns:
      - A dictionary with tokenized "input_ids", "attention_mask", and "labels".
    """
    instruction = examples["instruction"]
    input_text = examples["input"]
    response = examples["output"]

    if input_text.strip() != "":
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

    prompt_ids = tokenizer(prompt, truncation=True, max_length=512)
    response_ids = tokenizer(response, truncation=True, max_length=512)

    input_ids = (
        prompt_ids["input_ids"] + response_ids["input_ids"] + [tokenizer.eos_token_id]
    )
    attention_mask = [1] * len(input_ids)
    labels = (
        [-100] * len(prompt_ids["input_ids"])
        + response_ids["input_ids"]
        + [tokenizer.eos_token_id]
    )

    max_length = 512
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_replayed_candidate_model(
    base_model_name: str, cache_dir: str, candidate_json: str
):
    """Load the base model and replay saved stage-1/stage-2 pruning decisions."""
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = "auto"

    with open(candidate_json, "r", encoding="utf-8") as handle:
        candidate = json.load(handle)

    attn_mask = candidate.get("attn")
    mlp_mask = candidate.get("mlp")
    if attn_mask is None or mlp_mask is None:
        raise ValueError(
            f"Candidate replay file is missing attn/mlp masks: {candidate_json}"
        )

    model = loadModel(base_model_name, cache_dir, device_map=device_map)
    maskModel(model, attn_mask, mlp_mask)

    saved_stage2_layer_stats = candidate.get("stage2_layer_stats") or []
    if saved_stage2_layer_stats:
        logging.info(
            "Replaying saved stage-2 pruning decisions from %d layer records before DDP fine-tuning.",
            len(saved_stage2_layer_stats),
        )
        apply_saved_second_stage_pruning(model, saved_stage2_layer_stats)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a model using LoRA with configurable training epochs, sample size, and step frequency."
    )
    parser.add_argument(
        "--pruned_model_dir",
        type=str,
        default="pruned_model",
        help="Directory of the pruned model.",
    )
    parser.add_argument(
        "--tuned_model_dir",
        type=str,
        default="tuned_model",
        help="Directory to save the tuned model.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["llama"],
        default="llama",
        help="Model architecture: currently only 'llama' is supported.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=2,
        help="Number of training epochs for fine-tuning.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Number of steps between saving model checkpoints.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging training progress.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use from the dataset per epoch.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,
        help="Comma-separated list of target modules.",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device train batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate."
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.03, help="Warmup ratio."
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw_torch", help="Trainer optimizer name."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Reporting service to use (default is 'wandb' for llama).",
    )  # TODO acho que está sem wandb não?
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model to load before replaying pruning decisions.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--replay_candidate_json",
        type=str,
        default=None,
        help="JSON file containing saved pruning masks/stage-2 stats to replay.",
    )
    args = parser.parse_args()

    # If a custom target modules list is provided, split it into a list.
    if args.target_modules:
        modules = [x.strip() for x in args.target_modules.split(",")]
    else:
        modules = None

    replay_model = None
    replay_tokenizer = None
    if args.replay_candidate_json:
        if not args.base_model_name:
            raise ValueError(
                "--base_model_name is required with --replay_candidate_json."
            )
        replay_model = load_replayed_candidate_model(
            args.base_model_name,
            args.cache_dir,
            args.replay_candidate_json,
        )
        replay_tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    fine_tune_model(
        pruned_model_dir=args.pruned_model_dir,
        tuned_model_dir=args.tuned_model_dir,
        arch=args.arch,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_samples=args.num_samples,
        target_modules=modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        optimizer=args.optimizer,
        report_to=args.report_to,
        model=replay_model,
        tokenizer=replay_tokenizer,
    )
