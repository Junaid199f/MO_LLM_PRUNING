# Multi-Objective Structured Pruning of LLMs

Code accompanying **Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization**, by Muhammad Junaid Ali, Smail Niar, and El Ghazali Talbi ([arXiv:2607.22583](https://arxiv.org/abs/2607.22583)).

The method first uses NSGA-II to search attention/MLP sub-block masks, optimizing calibration quality against latency or model size. It then applies Bayesian layer-wise sparsity allocation to retained Pareto candidates and structurally prunes attention-head groups and MLP neurons.

## Installation

Use Python 3.11 and a CUDA-capable NVIDIA GPU. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[research]"
```

## Run

Start with the quick profile:

```bash
llm-prune --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --sparsity 0.375 --dry-run
llm-prune --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --sparsity 0.375
```

`--dry-run` prints the full configuration without loading models or creating files. `python main.py` and `python -m llm_pruning` provide the same interface.

Use the larger search budget reported in the paper:

```bash
llm-prune --model meta-llama/Llama-2-7b-hf --profile paper --sparsity 0.375 --gpus 0,1
llm-prune --model meta-llama/Llama-2-7b-hf --profile paper --sparsity 0.5 --gpus 0,1
```

| Setting | Quick | Paper budget |
| --- | ---: | ---: |
| NSGA-II population | 4 | 40 |
| Generations | 2 | 40 |
| Calibration samples | 32 | 1,024 |
| Sequence length | 512 | 2,048 |
| Total BO trials per candidate | 4 | 40 |
| Random warm-up trials, included in total | 2 | 20 |



Common options:

| Option | Purpose |
| --- | --- |
| `--calibration mixed` | Use the bundled mixed-task prompts for coarse calibration and fine-stage importance |
| `--importance wanda-sp` | Choose `importance`, `wanda-sp`, `flap`, `gradient`, `taylor`, or `wactigrad` |
| `--objective params` | Optimize active parameter count instead of stage-1 latency |
| `--gpus 0,1` | Distribute Pareto candidates across visible GPUs |
| `--evaluate` | Evaluate final perplexity and downstream tasks |
| `--save-model` | Export the selected model; disabled by default |
| `--stage1-only` | Stop after coarse Pareto search |
| `--output-dir outputs/my-run` | Set the run directory |
| `--fresh` | Ignore an existing compatible checkpoint |




See [advanced usage](docs/USAGE.md) for examples and checkpoint replay.

## Repository layout

```text
src/llm_pruning/
  cli.py, profiles.py       Public command and explicit search budgets
  pipeline.py              Two-stage workflow
  core/                    NSGA-II, masks, model loading, evaluation
  refinement/              Structured head/MLP pruning and scoring backends
  importance/              Activation-based importance hooks
  bayesian.py, budget.py   Bayesian search and budget arithmetic
  allocation.py            Layer-wise allocation comparisons
  calibration.py           Calibration preparation
  candidates.py            Saved Pareto candidates
  checkpoints.py           Resume state
  evaluation_workers.py    Parallel benchmark evaluation
  stage2_workers.py        Parallel Pareto refinement
  replay.py, export.py      Candidate reconstruction and export
  recovery.py, fine_tune.py Optional LoRA recovery
  metrics.py, reporting.py  CSV summaries and plots
  data/                    Existing calibration prompt subsets
```




## Citation and attribution

```bibtex
@misc{ali2026multiobjective,
  title={Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization},
  author={Muhammad Junaid Ali and Smail Niar and El Ghazali Talbi},
  year={2026},
  eprint={2607.22583},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

The original GPL v3 [LICENSE](LICENSE) and inherited source attribution are preserved. See [NOTICE.md](NOTICE.md) for the 2SSP foundation and adapted sources. Models and datasets retain their own terms.

