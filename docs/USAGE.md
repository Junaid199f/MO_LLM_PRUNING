Advanced Research Interface

The llm-prune command runs the proposed two-stage pruning method, including attention head pruning, MLP pruning, and Bayesian optimization. For additional experiments and custom configurations, use llm-prune-advanced.

Custom Search

The following example runs the proposed method on TinyLlama with 37.5% sparsity:

llm-prune-advanced --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --method proposed --sparsity 0.375 --proposed_pop_size 8 --proposed_n_gen 5 --second_stage_heads --second_stage_mlp --second_stage_bayes --second_stage_bayes_trials 10 --second_stage_bayes_warmup 4 --no-save_selected_stage2_model

The advanced command saves outputs relative to the current working directory. When running multiple experiments, use separate paths to avoid overwriting results.

Stage Ablation Experiments

The advanced interface supports experiments with individual pruning stages and baseline methods.

# Stage 1 only
llm-prune-advanced --model MODEL --sparsity 0.375 --stage1_only

# Stage 2 only
llm-prune-advanced --model MODEL --sparsity 0.375 --stage2_only --second_stage_heads --second_stage_mlp --second_stage_bayes

# BlockPruner baseline
llm-prune-advanced --model MODEL --method blockpruner --sparsity 0.375

You can select different importance methods using --second_stage_method and allocation methods using --second_stage_allocation_method. Run llm-prune-advanced --help to see the available options.

GPU Parallelism

The advanced interface supports running experiments across multiple GPUs.

--second_stage_bayes_candidate_parallel_gpus 0,1 runs separate Pareto candidates in parallel.

--second_stage_bayes_parallel_gpus 0,1 runs Bayesian optimization trials for a single candidate in parallel.

Avoid using the same GPUs for both parallel modes at the same time. GPU indices refer to the visible CUDA devices.

Resume and Evaluate Experiments

To resume an interrupted experiment, use the same configuration and checkpoint path.

Use --force_fresh_run to start a new experiment without loading previous results.

Previously saved Pareto solutions can be loaded using:

--proposed_load_solutions FILE

To evaluate a saved checkpoint without repeating the pruning process, use:

--pipeline_checkpoint_path FILE --checkpoint_eval_only

Keep the original model and pruning settings when resuming or evaluating checkpoints. Saved candidate masks and stage2_layer_stats are needed to reconstruct pruned models using llm_pruning.replay.

Optional fine-tuning after pruning can be enabled with --recovery_finetune. The recovery process uses llm_pruning.fine_tune and supports the implemented model architectures.

Calibration Settings

The calibration process can be adjusted using:

--num_calibration_samples: Number of calibration samples.

--stage1_calibration_tokens: Number of calibration tokens.

--stage1_tokens_per_evaluation: Number of tokens used per evaluation.

The paper configuration uses 1,024 sequences of 2,048 tokens for Stage 1. This may require considerable GPU memory, so smaller settings can be used when hardware resources are limited.

By default, Bayesian optimization uses the included mixed prompts for quality evaluation. To use your own validation prompts, provide JSON files through --bo_eval_prompt_files.

Local Datasets

To use datasets stored locally, enable:

--local_datasets

Datasets must be saved using Hugging Face's save_to_disk function. Their locations are defined in:

src/llm_pruning/core/data_loader_utils.py

Local dataset paths are interpreted relative to the directory where llm-prune-advanced is executed.
