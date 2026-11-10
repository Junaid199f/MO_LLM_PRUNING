import statistics
from typing import Dict, List, Tuple

import torch
from llm_pruning.utils import count_params, set_seed


def measure_latency(
    model,
    tokenizer,
    device: str = "cuda",
    input_prompt: str = "The Leaning Tower of Pisa is known for",
    batch_size: int = 4,
    num_all_runs: int = 30,
    num_warmup_runs: int = 10,
    top_p: float = 0.95,
    top_k: float = 50,
    temperature: float = 1.0,
    max_seq_len: int = 128,
    seed: int = 1234,
    gen_log_path: str = None,
) -> Tuple[Dict[str, float], List[float], List[float]]:
    """Run a simple batched generation benchmark and return latency/throughput stats."""
    if seed is not None:
        set_seed(seed)

    device_str = str(device)
    if "cuda" not in device_str:
        raise NotImplementedError(
            "Latency benchmark currently supports CUDA devices only."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for latency benchmarking.")

    model.eval()
    original_pad_token_id = tokenizer.pad_token_id
    original_padding_side = tokenizer.padding_side

    try:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        inputs = tokenizer(
            [input_prompt] * batch_size, return_tensors="pt", padding=True
        )["input_ids"].to(device_str)
        input_len = inputs[0].size(0)

        if gen_log_path:
            open(gen_log_path, "w", encoding="utf8").close()

        time_list: List[float] = []
        throughput_list: List[float] = []

        with torch.cuda.device(device_str):
            for i in range(num_all_runs):
                starter, ender = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                starter.record()
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=inputs,
                        do_sample=True,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        max_length=(input_len + max_seq_len),
                        min_length=(input_len + max_seq_len),
                        return_dict_in_generate=True,
                    )
                ender.record()
                torch.cuda.synchronize()
                batch_time = starter.elapsed_time(ender) / 1000

                if i < num_warmup_runs:
                    continue

                if gen_log_path:
                    output = tokenizer.batch_decode(generation_output.sequences)
                    for b_idx, s in enumerate(output):
                        tmp_size = len(generation_output.sequences[b_idx]) - input_len
                        with open(gen_log_path, "a", encoding="utf8") as f:
                            f.write(
                                f"=== output {i} b{b_idx} | leng gen {tmp_size} + input {input_len}\n"
                            )
                            f.write(f"{s}\n")

                time_list.append(batch_time)
                throughput_list.append(batch_size * max_seq_len / batch_time)

        mem = torch.cuda.memory_allocated(device=device_str) / 1024 / 1024
        nparams = count_params(model)
        time_mean = statistics.mean(time_list)
        time_std = statistics.pstdev(time_list)
        throughput_mean = statistics.mean(throughput_list)
        throughput_std = statistics.pstdev(throughput_list)
        nbatches = len(throughput_list)

        metrics = {
            "time_mean(sec)": time_mean,
            "time_std": time_std,
            "th_mean(tokens/s)": throughput_mean,
            "th_std": throughput_std,
            "mem": mem,
            "out_len": max_seq_len,
            "in_len": input_len,
            "nbatches": nbatches,
            "batchsz": batch_size,
            "nparam": nparams,
        }
    finally:
        tokenizer.pad_token_id = original_pad_token_id
        tokenizer.padding_side = original_padding_side

    return metrics, time_list, throughput_list
