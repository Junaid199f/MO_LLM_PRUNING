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

On Windows, activate with `.venv\Scripts\Activate.ps1`. Use a CUDA-enabled PyTorch build appropriate for your GPU. Gated models require access through your Hugging Face account.

To inspect commands and run the dependency-free checks without installing the research dependencies, use `python -m pip install -e .` instead. The dependency pins extend the original workspace's requirements; the complete ML environment and GPU execution have not been validated here.

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

The quick profile checks the pipeline; it is not a numerical reproduction. The paper profile is a large search and can require substantial CPU RAM for calibration logits and model copies. Search budgets alone do not establish exact reproduction; [implementation notes](docs/IMPLEMENTATION.md) record the known differences.

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

Each run keeps checkpoints, CSV summaries, plots, and optional exports in its output directory. Repeating the same command resumes a compatible checkpoint. GPU indices are relative to `CUDA_VISIBLE_DEVICES`.

The full research interface remains available for custom budgets, baselines, ablations, trial-level GPU parallelism, and LoRA recovery:

```bash
llm-prune-advanced --help
python -m llm_pruning.advanced --help
```

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

The release excludes backups, logs, generated results, weights, external repository checkouts, scheduler-specific scripts, and the experimental third-stage KV search. The original working folder is separate and unchanged.

## Data and model replay

WikiText-2, C4, and FineWeb-Edu are loaded through Hugging Face Datasets. The bundled mixed-prompt calibration files retain the original workspace contents. Their source revisions, sampling history, and split separation are not fully recorded; see [data notes](docs/DATA.md). Downstream evaluation uses the installed `lm_eval` package.

Per-layer structural pruning can produce shapes that a standard Hugging Face configuration cannot describe. Keep the base model ID, masks, stage-2 records, and tokenizer for replay. Do not assume a plain `save_pretrained` export loads correctly with an unmodified loader.

## Validation

```bash
python -m unittest discover -s tests -v
```

The 20 tests cover budget conservation, layer retention limits, argument validation, output isolation, package imports, and bundled data. A wheel build and a temporary installation without research dependencies were also checked, including all 32 bundled calibration JSON files. GPU execution and the paper's numerical results remain untested in this environment.

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

Use **this folder** as the GitHub repository root.
