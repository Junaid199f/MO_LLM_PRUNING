"""Calibration loading, prompt parsing, and quality evaluation."""

from llm_pruning.config import BO_INVALID_SCORE_PENALTY
import json
import logging
import math
from pathlib import Path
import torch
from typing import Dict, List, Optional, Sequence, Tuple
from llm_pruning.core.data_loader_utils import (
    load_c4,
    load_fineweb_edu,
    load_gsm8k,
    load_wikitext2,
)
from llm_pruning.core.utilities import get_calibration, maskModel, unmaskModel


def prepare_wikitext_inputs(tokenizer, use_local: bool, split: str = "test"):
    """Tokenize the requested Wikitext2 split."""
    dataset = load_wikitext2(use_local, split=split)
    logging.info("Tokenizing Wikitext2 %s split.", split)
    return tokenizer(
        "\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False
    ).input_ids


def prepare_c4_validation_inputs(tokenizer, use_local: bool):
    """Tokenize the C4 validation split."""
    dataset = load_c4(train=False, local=use_local)
    logging.info("Tokenizing C4 validation split.")
    return tokenizer(
        "\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False
    ).input_ids


def prepare_fineweb_edu_inputs(tokenizer, use_local: bool):
    """Tokenize the fineweb-edu split used for perplexity evaluation."""
    dataset = load_fineweb_edu(local=use_local)
    logging.info("Tokenizing fineweb-edu split.")
    return tokenizer(
        "\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False
    ).input_ids


def _load_prompt_texts_from_file(prompt_file: str) -> List[str]:
    """Load usable prompt texts from a JSON prompt file."""
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"BO evaluation prompt file not found: {prompt_file}")
    if path.stat().st_size == 0:
        raise ValueError(f"BO evaluation prompt file is empty: {prompt_file}")
    with open(path, "r", encoding="utf-8-sig") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in BO evaluation prompt file {prompt_file} "
                f"(size={path.stat().st_size} bytes): {exc}"
            ) from exc
    if not isinstance(payload, list):
        raise ValueError(
            f"BO evaluation prompt file must contain a list: {prompt_file}"
        )

    texts = []
    for idx, entry in enumerate(payload):
        if isinstance(entry, str):
            text = entry.strip()
        elif isinstance(entry, dict):
            raw_text = entry.get("text")
            text = raw_text.strip() if isinstance(raw_text, str) else ""
        else:
            raise ValueError(
                f"Unsupported BO prompt entry type in {prompt_file} at index {idx}: {type(entry)}"
            )
        if text:
            texts.append(text)

    if not texts:
        raise ValueError(
            f"BO evaluation prompt file contains no usable text: {prompt_file}"
        )
    return texts


def load_prompt_texts(prompt_files: Sequence[str]) -> List[str]:
    """Load and flatten prompt texts from the mixed calibration prompt files."""
    all_texts = []
    for prompt_file in prompt_files:
        all_texts.extend(_load_prompt_texts_from_file(prompt_file))
    if not all_texts:
        raise ValueError("No usable prompt text found in the mixed calibration set.")
    return all_texts


def prepare_prompt_mix_inputs(tokenizer, prompt_files: Sequence[str]) -> torch.Tensor:
    """Tokenize the combined mixed calibration prompt set into a single stream."""
    texts = load_prompt_texts(prompt_files)
    logging.info(
        "Tokenizing mixed calibration prompt set with %d prompts from %d files.",
        len(texts),
        len(prompt_files),
    )
    return tokenizer(
        "\n\n".join(texts),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


def build_prompt_mix_corpus(prompt_files: Sequence[str]) -> List[Dict[str, str]]:
    """Build a text corpus compatible with stage-1 KL calibration sampling."""
    texts = load_prompt_texts(prompt_files)
    logging.info(
        "Building mixed calibration corpus with %d prompts from %d files.",
        len(texts),
        len(prompt_files),
    )
    return [{"text": text} for text in texts]


def prepare_bo_prompt_eval_inputs(
    tokenizer,
    prompt_files: Sequence[str],
) -> List[Tuple[str, torch.Tensor]]:
    """Tokenize BO evaluation prompt files containing prompt-only JSON entries."""
    datasets = []
    for prompt_file in prompt_files:
        path = Path(prompt_file)
        texts = _load_prompt_texts_from_file(prompt_file)

        logging.info(
            "Tokenizing BO evaluation prompt file %s with %d prompts.",
            path.name,
            len(texts),
        )
        input_ids = tokenizer(
            "\n\n".join(texts),
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        datasets.append((path.stem, input_ids))
    return datasets


@torch.no_grad()
def _bo_compute_running_nll(
    model, input_ids, seq_len=2048, batch_size=1, enable_tqdm=False, device="cuda"
):
    """Return the average token NLL used to score BO prompt datasets."""
    total_tokens = int(input_ids.numel())
    if total_tokens < 2:
        logging.warning(
            "BO prompt dataset has fewer than 2 tokens; skipping perplexity evaluation."
        )
        return None
    effective_seq_len = min(int(seq_len), total_tokens)
    if effective_seq_len < 2:
        logging.warning(
            "BO prompt dataset effective seq_len is < 2; skipping perplexity evaluation."
        )
        return None

    num_samples = max(1, total_tokens // effective_seq_len)
    data = []
    for i in range(num_samples):
        start = i * effective_seq_len
        end = min((i + 1) * effective_seq_len, total_tokens)
        chunk = input_ids[:, start:end]
        if chunk.numel() >= 2:
            data.append(chunk)

    if not data:
        logging.warning(
            "BO prompt dataset produced no valid token chunks; skipping perplexity evaluation."
        )
        return None

    nll_running = None
    tokens_processed = 0
    ppl_range = range(0, len(data), batch_size)

    for i in ppl_range:
        j = min(i + batch_size, len(data))
        inputs = torch.cat(data[i:j]).to(device)
        lm_logits = model(inputs, use_cache=False).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        if not torch.isfinite(shift_logits).all():
            logging.warning("Perplexity logits are non-finite; returning inf.")
            return None
        shift_logits = shift_logits.float()
        shift_labels = inputs[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        if not torch.isfinite(loss):
            logging.warning("Perplexity loss is non-finite; returning inf.")
            return None
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = loss if nll_running is None else (a * loss + b * nll_running)
        tokens_processed += shift_labels.numel()

    if nll_running is None:
        return None
    if not torch.isfinite(nll_running):
        logging.warning("Perplexity running NLL is non-finite; returning inf.")
        return None
    return nll_running


@torch.no_grad()
def _evaluate_perplexity_suite_local(
    model,
    datasets: Sequence[Tuple[str, torch.Tensor]],
    *,
    seq_len=2048,
    batch_size=1,
    device="cuda",
) -> Tuple[float, Dict[str, float]]:
    """Aggregate BO prompt perplexity with an arithmetic mean."""
    per_dataset = {}
    ppl_values = []

    for name, input_ids in datasets:
        if input_ids is None:
            logging.warning("Skipping perplexity for %s: no input IDs provided.", name)
            continue
        nll = _bo_compute_running_nll(
            model,
            input_ids,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
        )
        if nll is None:
            per_dataset[name] = float("inf")
            return BO_INVALID_SCORE_PENALTY, per_dataset
        ppl = nll.exp().item()
        if not math.isfinite(ppl):
            logging.warning(
                "Perplexity value is non-finite after exp for %s; returning inf.", name
            )
            per_dataset[name] = float("inf")
            return BO_INVALID_SCORE_PENALTY, per_dataset
        per_dataset[name] = ppl
        ppl_values.append(float(ppl))

    if not ppl_values:
        return BO_INVALID_SCORE_PENALTY, per_dataset

    aggregate = sum(ppl_values) / len(ppl_values)
    if not math.isfinite(aggregate):
        logging.warning("Aggregate perplexity score is non-finite; returning inf.")
        return BO_INVALID_SCORE_PENALTY, per_dataset
    logging.info(
        "Aggregate perplexity score (%s) = %.4f",
        ", ".join(name for name, _ in datasets),
        aggregate,
    )
    return aggregate, per_dataset


@torch.no_grad()
def _evaluate_nll_suite_local(
    model,
    datasets: Sequence[Tuple[str, torch.Tensor]],
    *,
    seq_len=2048,
    batch_size=1,
    device="cuda",
) -> Tuple[Optional[float], Dict[str, float]]:
    """Aggregate BO prompt average token NLL with an arithmetic mean."""
    per_dataset = {}
    nll_values = []

    for name, input_ids in datasets:
        if input_ids is None:
            logging.warning("Skipping NLL for %s: no input IDs provided.", name)
            continue
        nll = _bo_compute_running_nll(
            model,
            input_ids,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
        )
        if nll is None:
            per_dataset[name] = float("inf")
            return None, per_dataset
        nll_value = float(nll.item())
        if not math.isfinite(nll_value):
            logging.warning("NLL value is non-finite for %s; returning inf.", name)
            per_dataset[name] = float("inf")
            return None, per_dataset
        per_dataset[name] = nll_value
        nll_values.append(nll_value)

    if not nll_values:
        return None, per_dataset

    aggregate = sum(nll_values) / len(nll_values)
    if not math.isfinite(aggregate):
        logging.warning("Aggregate NLL score is non-finite; returning inf.")
        return None, per_dataset
    return float(aggregate), per_dataset


def _normalize_stage1_scores(scores: Sequence[float]) -> List[float]:
    """Normalize measured ablation deltas into [0, 1] without tie-breaking heuristics."""
    if not scores:
        return []
    finite_scores = [float(score) for score in scores if math.isfinite(float(score))]
    if not finite_scores:
        return [1.0 for _ in scores]

    min_score = min(finite_scores)
    max_score = max(finite_scores)
    scale = max_score - min_score
    if scale <= 1e-12:
        return [0.0 if math.isfinite(float(score)) else 1.0 for score in scores]
    return [
        (
            float(max(0.0, min(1.0, (float(score) - min_score) / scale)))
            if math.isfinite(float(score))
            else 1.0
        )
        for score in scores
    ]


@torch.no_grad()
def _compute_stage1_block_importance_scores(
    model,
    datasets: Sequence[Tuple[str, torch.Tensor]],
    *,
    num_blocks: int,
    seq_len: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
) -> Dict[str, object]:
    """Estimate per-block stage-1 importance from the configured calibration prompt set."""
    if not datasets:
        raise ValueError(
            "Stage-1 importance penalty requires non-empty calibration prompt datasets."
        )
    if num_blocks <= 0:
        raise ValueError(
            "Stage-1 importance penalty requires at least one model block."
        )

    baseline_nll, baseline_nll_breakdown = _evaluate_nll_suite_local(
        model,
        datasets,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    baseline_score, _ = _evaluate_perplexity_suite_local(
        model,
        datasets,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    logging.info(
        "Stage-1 importance baseline calibration-prompt perplexity = %.4f across %d datasets.",
        baseline_score,
        len(datasets),
    )
    print(
        "Stage-1 importance baseline calibration-prompt perplexity = "
        f"{baseline_score:.4f} across {len(datasets)} datasets."
    )
    if baseline_nll is None:
        raise RuntimeError("Stage-1 importance baseline NLL could not be computed.")
    logging.info(
        "Stage-1 importance baseline calibration-prompt NLL = %.8f across %d datasets.",
        baseline_nll,
        len(baseline_nll_breakdown),
    )

    zero_attn_mask = [0] * num_blocks
    zero_mlp_mask = [0] * num_blocks
    attn_scores: List[float] = []
    mlp_scores: List[float] = []

    def _score_single_block(
        *, attn_idx: Optional[int] = None, mlp_idx: Optional[int] = None
    ) -> float:
        attn_mask = list(zero_attn_mask)
        mlp_mask = list(zero_mlp_mask)
        if attn_idx is not None:
            attn_mask[int(attn_idx)] = 1
        if mlp_idx is not None:
            mlp_mask[int(mlp_idx)] = 1
        maskModel(model, attn_mask, mlp_mask)
        try:
            score, _ = _evaluate_nll_suite_local(
                model,
                datasets,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
            )
        finally:
            unmaskModel(model, attn_mask, mlp_mask)
        if score is None or not math.isfinite(score):
            return float(BO_INVALID_SCORE_PENALTY)
        return max(0.0, float(score - baseline_nll))

    for layer_idx in range(num_blocks):
        attn_delta = _score_single_block(attn_idx=layer_idx)
        mlp_delta = _score_single_block(mlp_idx=layer_idx)
        attn_scores.append(attn_delta)
        mlp_scores.append(mlp_delta)
        print(
            "Stage-1 importance layer "
            f"{layer_idx + 1}/{num_blocks}: attn_delta={attn_delta:.6f}, mlp_delta={mlp_delta:.6f}"
        )
        if (
            (layer_idx + 1) == 1
            or (layer_idx + 1) == num_blocks
            or (layer_idx + 1) % 5 == 0
        ):
            logging.info(
                "Computed stage-1 importance scores for %d/%d layers.",
                layer_idx + 1,
                num_blocks,
            )
            print(
                f"Computed stage-1 importance scores for {layer_idx + 1}/{num_blocks} layers."
            )

    normalized_attn_scores = _normalize_stage1_scores(attn_scores)
    normalized_mlp_scores = _normalize_stage1_scores(mlp_scores)

    if normalized_attn_scores and max(normalized_attn_scores) <= 1e-12:
        logging.warning(
            "Stage-1 attention ablation deltas were flat on the calibration set; normalized scores remain tied."
        )
    if normalized_mlp_scores and max(normalized_mlp_scores) <= 1e-12:
        logging.warning(
            "Stage-1 MLP ablation deltas were flat on the calibration set; normalized scores remain tied."
        )

    logging.info(
        "Stage-1 importance penalty scores prepared for %d attention and %d MLP blocks.",
        len(normalized_attn_scores),
        len(normalized_mlp_scores),
    )
    print(
        "Stage-1 importance penalty scores prepared for "
        f"{len(normalized_attn_scores)} attention and {len(normalized_mlp_scores)} MLP blocks."
    )
    return {
        "baseline_score": float(baseline_score),
        "baseline_nll": float(baseline_nll),
        "attn_raw": [float(score) for score in attn_scores],
        "mlp_raw": [float(score) for score in mlp_scores],
        "attn": normalized_attn_scores,
        "mlp": normalized_mlp_scores,
    }


def prepare_gsm8k_inputs(tokenizer, use_local: bool):
    """Tokenize the GSM8K split for perplexity evaluation."""
    try:
        dataset = load_gsm8k(local=use_local)
    except Exception as exc:
        logging.warning(
            "Skipping GSM8K perplexity dataset because it could not be loaded: %s", exc
        )
        return None
    if "question" not in dataset.column_names or "answer" not in dataset.column_names:
        logging.warning(
            "Skipping GSM8K perplexity dataset because required fields are missing. Columns: %s",
            dataset.column_names,
        )
        return None
    logging.info("Tokenizing GSM8K split.")
    prompts = [
        f"Q: {q}\nA: {a}" for q, a in zip(dataset["question"], dataset["answer"])
    ]
    return tokenizer(
        "\n\n".join(prompts), return_tensors="pt", add_special_tokens=False
    ).input_ids


def build_calibration(
    tokenizer, use_local: bool, num_samples: int, seq_len: int, dataset: str = "c4"
):
    """Create a calibration dataset from the chosen corpus."""
    dataset_name = dataset.lower()
    if dataset_name == "c4":
        calibration_corpus = load_c4(train=True, local=use_local)
    elif dataset_name == "wikitext2":
        calibration_corpus = load_wikitext2(use_local, split="train")
    elif dataset_name == "fineweb_edu":
        calibration_corpus = load_fineweb_edu(local=use_local)
    else:
        raise ValueError(f"Unsupported calibration dataset '{dataset}'.")

    logging.info(
        "Building calibration set from %s (%d samples, seq_len=%d).",
        dataset_name,
        num_samples,
        seq_len,
    )
    calibration = get_calibration(
        calibration_corpus,
        tokenizer,
        num_samples=num_samples,
        seq_len=seq_len,
    )
    return calibration_corpus, calibration
