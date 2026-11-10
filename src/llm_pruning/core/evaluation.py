import os
import math
import logging
import contextlib
import torch
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from lm_eval import evaluator


# Perplexity evaluation on the supported text benchmarks.
def evaluation_ppl(
    model,
    wikitext_input_ids,
    c4_val_input_ids,
    fineweb_edu_input_ids,
    code_input_ids=None,
    gsm8k_input_ids=None,
    seq_len=2048,
    batch_size=1,
):
    datasets = [
        ("wikitext2", wikitext_input_ids),
        ("c4", c4_val_input_ids),
        ("fineweb_edu", fineweb_edu_input_ids),
        ("code", code_input_ids),
        ("gsm8k", gsm8k_input_ids),
    ]

    results = {}
    for name, ids in datasets:
        if ids is None:
            logging.warning("Skipping perplexity for %s: no input IDs provided.", name)
            continue
        ppl = evaluate_perplexity(model, ids, seq_len=seq_len, batch_size=batch_size)
        results[name] = ppl
        logging.info("Perplexity (%s) = %.4f", name, ppl)
    return results


# The following code was sourced and adapted from the repository:
# https://github.com/IST-DASLab/EvoPress
@torch.no_grad()
def evaluate_perplexity(
    model, input_ids, seq_len=2048, batch_size=1, enable_tqdm=True, device="cuda"
):
    # Divide the test dataset into samples of exactly seq_len tokens
    num_samples = input_ids.numel() // seq_len
    data = []
    for i in range(num_samples):
        data.append(input_ids[:, i * seq_len : (i + 1) * seq_len])

    # Running estimate of negative log-likelihood
    nll_running = None
    tokens_processed = 0

    if enable_tqdm:
        ppl_range = tqdm(
            range(0, num_samples, batch_size), desc=f"Calculating perplexity"
        )
    else:
        ppl_range = range(0, num_samples, batch_size)

    # Loop through each batch
    for i in ppl_range:
        j = min(i + batch_size, num_samples)
        inputs = torch.cat(data[i:j]).to(device)
        # Forward pass through the model
        # Disable cache to avoid legacy cache conversions on models loaded with accelerate/device maps
        lm_logits = model(inputs, use_cache=False).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        if not torch.isfinite(loss):
            logging.warning("Perplexity loss is non-finite; returning inf.")
            return float("inf")
        # Calculate negative log likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = loss if nll_running is None else (a * loss + b * nll_running)
        # Update number of processed tokens
        tokens_processed += shift_labels.numel()

    if nll_running is None:
        return float("inf")
    # Compute perplexity
    if not torch.isfinite(nll_running):
        logging.warning("Perplexity running NLL is non-finite; returning inf.")
        return float("inf")
    ppl = nll_running.exp().item()
    if not math.isfinite(ppl):
        logging.warning("Perplexity value is non-finite after exp; returning inf.")
        return float("inf")
    return ppl


# Downstream tasks evaluation with Lm Eval Harness
def evaluation_downstream(
    model, model_name, num_fewshot=0, task_list=None, batch_size=1
):
    if task_list is None:
        task_list = [
            "hellaswag",
            "mmlu",
            "piqa",
            "winogrande",
            "arc_challenge",
            "arc_easy",
        ]
    # task_list = ["arc_easy"]
    results = eval_zero_shot(
        model_name,
        model,
        task_list,
        num_fewshot=num_fewshot,
        use_accelerate=True,
        access_token=os.environ.get("HF_TOKEN"),
        batch_size=batch_size,
    )

    logging.info("Zero-shot evaluation results")
    for task in results["results"].keys():
        task_res = results["results"][task]
        alias = task_res.get("alias", task)
        score = None
        for key in ("acc_norm", "acc,none", "acc"):
            if key in task_res:
                score = task_res[key]
                break
        if score is None:
            logging.info("%s : %s", alias, task_res)
        else:
            logging.info("%s : %s", alias, score)
    return results


@contextlib.contextmanager
def _disable_model_cache(model):
    """Temporarily disable KV cache for evaluation paths that call model() directly."""
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    prev_model_use_cache = getattr(config, "use_cache", None)
    prev_generation_use_cache = getattr(generation_config, "use_cache", None)

    try:
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = False
        yield
    finally:
        if (
            config is not None
            and hasattr(config, "use_cache")
            and prev_model_use_cache is not None
        ):
            config.use_cache = prev_model_use_cache
        if (
            generation_config is not None
            and hasattr(generation_config, "use_cache")
            and prev_generation_use_cache is not None
        ):
            generation_config.use_cache = prev_generation_use_cache


# The following code was sourced and adapted from the repository:
# https://github.com/eliacunegatti/NeuroAL
def eval_zero_shot(
    model_name,
    model,
    task_list=["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"],
    num_fewshot=0,
    use_accelerate=False,
    access_token=None,
    batch_size=1,
):
    model_args = f"token={access_token}"
    limit = None
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"use_accelerate=True"

    logging.info(f"Testing tasks: {task_list}")
    with _disable_model_cache(model):
        model_obj = HFLM(pretrained=model, batch_size=batch_size)
        results = evaluator.simple_evaluate(
            model=model_obj,
            model_args=model_args,
            tasks=task_list,
            num_fewshot=num_fewshot,
            batch_size=None,
            device=None,
            limit=limit,
            check_integrity=False,
        )

    return results


@torch.no_grad()
def evaluate_inference_time(model, sample, calibration_dataset=None, batch_size=1):
    if calibration_dataset is not None and batch_size > 1:
        batch_samples = calibration_dataset[:batch_size]
        if len(batch_samples) < batch_size:
            raise ValueError(
                f"Requested batch_size={batch_size}, but only {len(batch_samples)} calibration samples are available."
            )
        sample = torch.cat(batch_samples, dim=0)
    sample = sample.to("cuda")

    # model warmup
    for _ in range(10):
        _ = model(sample, use_cache=False)

    # Cuda events counter
    n_runs = 10
    times = []
    throughputs = []
    # Treat "tokens processed" as batch_size * seq_len for a forward pass.
    token_count = sample.numel()
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        _ = model(sample, use_cache=False)
        torch.cuda.synchronize()
        end.record()

        torch.cuda.synchronize()

        elapsed_time_ms = start.elapsed_time(end)
        elapsed_time_s = elapsed_time_ms / 1000.0
        tokens_per_s = token_count / max(elapsed_time_s, 1e-9)
        logging.info(f"Inference time: {elapsed_time_s} s")
        logging.info(f"Throughput: {tokens_per_s:.2f} tokens/s")
        times.append(elapsed_time_s)
        throughputs.append(tokens_per_s)

    average_time = sum(times) / n_runs
    average_throughput = sum(throughputs) / n_runs
    logging.info(f"Average Inference Time: {average_time:.6f} seconds")
    logging.info(f"Average Throughput: {average_throughput:.2f} tokens/s")
    return average_time


@torch.no_grad()
def generate_response(prompt, model, tokenizer, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_length": max_length,
        "num_beams": 5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        if tokenizer.pad_token_id != tokenizer.eos_token_id:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    outputs = model.generate(**inputs, **generation_kwargs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


@torch.no_grad()
def qualitative_results(model, tokenizer, max_length=64):
    prompt1 = "Who is Albert Einstein?"
    prompt2 = "The theory of relativity"

    logging.info("Prompt 1: Who is Albert Einstein?")
    logging.info("-" * 20)
    logging.info(generate_response(prompt1, model, tokenizer, max_length))
    logging.info("Prompt 2: Explain the theory of relativity")
    logging.info("-" * 20)
    logging.info(generate_response(prompt2, model, tokenizer, max_length))
