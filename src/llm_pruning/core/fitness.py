# The following code was sourced and adapted from the repository:
# https://github.com/IST-DASLab/EvoPress
import torch
from transformers import AutoTokenizer


def collect_samples_with_join(
    calibration_dataset,
    tokenizer: AutoTokenizer,
    num_samples: int,
    sequence_length: int,
    text_key: str = "text",
):
    data = []
    data_iter = iter(calibration_dataset)

    samples_collected = 0
    current_sample = torch.empty((1, 0), dtype=torch.int64)
    for sample in data_iter:
        tokenized_sample = tokenizer(
            sample[text_key], return_tensors="pt", add_special_tokens=False
        ).input_ids
        current_sample = torch.cat([current_sample, tokenized_sample], dim=1)
        if current_sample.numel() >= sequence_length:
            samples_collected += 1
            data.append(
                current_sample[:, :sequence_length]
            )  # trim to sequence length; this introduces bias
            current_sample = torch.empty(
                (1, 0), dtype=torch.int64
            )  # reset current sample
        else:
            # add 2 new lines to the current sample
            current_sample = torch.cat(
                [
                    current_sample,
                    tokenizer(
                        "\n\n", return_tensors="pt", add_special_tokens=False
                    ).input_ids,
                ],
                dim=1,
            )
        # Stop if sufficient number of samples are collected
        if samples_collected >= num_samples:
            break
    return data


@torch.no_grad()
def compute_kl_div(model, data, target_logits, batch_size=1):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    kl_div_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in range(0, num_samples, batch_size):
        torch.cuda.empty_cache()
        j = min(i + batch_size, num_samples)

        inputs = torch.cat(data[i:j]).to(device)
        targets = torch.cat(target_logits[i:j]).to(device)
        # Forward pass through the model
        lm_logits = model(inputs, use_cache=False).logits

        # Don't predict last token (not required, can be removed)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, :-1, :]

        # Squeeze on GPU
        torch.cuda.empty_cache()
        for i in range(0, shift_logits.shape[1], 1024):
            j = min(i + 1024, shift_logits.shape[1])
            shift_logits_batch = shift_logits[:, i:j, :]
            shift_targets_batch = shift_targets[:, i:j, :]
            loss_batch = torch.nn.functional.kl_div(
                shift_logits_batch.reshape(-1, shift_logits_batch.size(-1)).log_softmax(
                    dim=-1
                ),
                shift_targets_batch.reshape(
                    -1, shift_targets_batch.size(-1)
                ).log_softmax(dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            # Calculate negative log likelihood
            a = shift_targets_batch.numel() / (
                tokens_processed + shift_targets_batch.numel()
            )
            b = tokens_processed / (tokens_processed + shift_targets_batch.numel())
            kl_div_running = a * loss_batch + b * kl_div_running
            # Update number of processed tokens
            tokens_processed += shift_targets_batch.numel()
            del shift_logits_batch, shift_targets_batch, loss_batch
            torch.cuda.empty_cache()

    return kl_div_running.item()


def compute_fitness(model, data, target_logits):
    return compute_kl_div(model, data, target_logits)
