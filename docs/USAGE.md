# Advanced research interface

The public `llm-prune` command always enables the two-stage proposed method with head and MLP pruning and Bayesian allocation. Use `llm-prune-advanced` for other experiments. The advanced interface preserves the original argument names and defaults.

## Custom search

```bash
llm-prune-advanced --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --method proposed --sparsity 0.375 --proposed_pop_size 8 --proposed_n_gen 5 --second_stage_heads --second_stage_mlp --second_stage_bayes --second_stage_bayes_trials 10 --second_stage_bayes_warmup 4 --no-save_selected_stage2_model
```

Unlike the public command, the advanced interface writes relative outputs to the current working directory. Set checkpoint and artifact paths explicitly when running several experiments.

## Stage ablations

```bash
llm-prune-advanced --model MODEL --sparsity 0.375 --stage1_only
llm-prune-advanced --model MODEL --sparsity 0.375 --stage2_only --second_stage_heads --second_stage_mlp --second_stage_bayes
llm-prune-advanced --model MODEL --method blockpruner --sparsity 0.375
```

Stage-2 importance methods are selected with `--second_stage_method`. Allocation comparisons are selected with `--second_stage_allocation_method`; see `--help` for the supported names.

## GPU parallelism

`--second_stage_bayes_candidate_parallel_gpus 0,1` refines separate Pareto candidates concurrently. `--second_stage_bayes_parallel_gpus 0,1` evaluates trials within one candidate concurrently. Avoid assigning the same GPUs to nested parallel modes. GPU indices are local to the visible CUDA devices.

## Resume and replay

Use the same arguments and checkpoint path to resume a compatible run. `--force_fresh_run` ignores existing state. `--proposed_load_solutions FILE` accepts saved coarse Pareto CSV/JSON masks.

For checkpoint evaluation, use `--pipeline_checkpoint_path FILE --checkpoint_eval_only` together with the original model and pruning arguments. Changing the run signature can invalidate resume state. Keep candidate masks and `stage2_layer_stats`: these records are used by `llm_pruning.replay` to reconstruct structurally pruned models.

Optional recovery is enabled with `--recovery_finetune`; its distributed launcher invokes `python -m llm_pruning.fine_tune` through torchrun. Only the implemented architecture-specific recovery paths are supported.

## Calibration controls

`--num_calibration_samples`, `--stage1_calibration_tokens`, and `--stage1_tokens_per_evaluation` are independent controls. The paper-budget profile requests 1,024 full 2,048-token sequences for stage-1 KL targets; storing those targets can be costly. Reduce the calibration-token budget for memory-limited experiments and record the changed setting.

The default BO quality inputs remain the bundled mixed prompts even when WikiText-2 supplies coarse calibration and fine-stage importance. For a distinct BO validation corpus, provide your own prompt-only JSON files with `--bo_eval_prompt_files`.

`--local_datasets` reads datasets saved with Hugging Face `save_to_disk` from paths in `src/llm_pruning/core/data_loader_utils.py`. The local paths are relative to the advanced command's working directory.
