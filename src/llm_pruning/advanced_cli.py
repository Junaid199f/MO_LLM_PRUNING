"""Command-line options for the two-stage pruning pipeline."""

import argparse
import sys
from llm_pruning.config import DEFAULT_BO_EVAL_PROMPT_FILES, DEFAULT_DOWNSTREAM_TASKS


def parse_args(argv=None):
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hardware-aware two-stage structured pruning of language models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name or path.",
    )
    method_group = parser.add_mutually_exclusive_group(required=False)
    method_group.add_argument(
        "--method",
        type=str,
        choices=["proposed", "blockpruner"],
        help="Choose which pruning strategy to run.",
    )
    method_group.add_argument(
        "--pruning_method",
        type=str,
        choices=["proposed", "blockpruner"],
        dest="method",
        help="Alias for --method to preserve the original CLI.",
    )
    parser.set_defaults(method="proposed")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Global target sparsity (fraction of blocks to prune across stages).",
    )
    parser.add_argument(
        "--sparsity_alpha",
        type=float,
        default=0.2,
        help="Fraction of the global sparsity budget targeted in stage 1; stage 2 uses the remaining budget.",
    )
    parser.add_argument(
        "--sparsity_rate",
        type=float,
        dest="sparsity",
        help="Alias for --sparsity to preserve the original CLI.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Model cache directory."
    )
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        choices=["c4", "wikitext2", "fineweb_edu"],
        default="wikitext2",
        help="Dataset used to build the stage-1 calibration set for KL/perplexity.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length for calibration and perplexity computations.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=256,
        help="Number of calibration samples to build from --calibration_dataset.",
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--evaluate_perplexity",
        action="store_true",
        help="Report perplexity on Wikitext2 for the pruned candidates.",
    )
    parser.add_argument(
        "--evaluate_perplexity_stage1",
        action="store_true",
        help="Report perplexity on Wikitext2 after stage-1 pruning (before stage-2).",
    )
    parser.add_argument(
        "--evaluate_inference",
        action="store_true",
        help="Measure inference latency on a calibration sample (CUDA only).",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1,
        help="Batch size for inference-time measurement; uses calibration samples when >1.",
    )
    parser.add_argument(
        "--evaluate_perplexity_all",
        default=False,
        action="store_true",
        help="Report perplexity on Wikitext2, C4, and fineweb-edu for the pruned candidates.",
    )
    parser.add_argument(
        "--perplexity_batch_size",
        type=int,
        default=1,
        help="Batch size for perplexity evaluation.",
    )
    parser.add_argument(
        "--perplexity_parallel_gpus",
        type=str,
        default="",
        help=(
            "Comma-separated CUDA GPU ids for dataset-parallel final perplexity evaluation "
            "when replaying final candidates, e.g. '0,1,2'. Use 'auto' for all visible GPUs. "
            "Leave empty to run perplexity evaluation in the current process."
        ),
    )
    parser.add_argument(
        "--evaluate_downstream",
        action="store_true",
        help="Run lm-eval-harness downstream tasks on each candidate.",
    )
    parser.add_argument(
        "--downstream_tasks",
        type=str,
        default=",".join(DEFAULT_DOWNSTREAM_TASKS),
        help=(
            "Comma-separated lm-eval task list for downstream evaluation. "
            "Defaults to HellaSwag, MMLU, ARC-Easy, ARC-Challenge, WinoGrande, and PIQA."
        ),
    )
    parser.add_argument(
        "--downstream_batch_size",
        type=str,
        default="1",
        help=(
            "Per-worker lm-eval batch size for downstream evaluation. "
            "Use an integer or lm-eval's 'auto' setting."
        ),
    )
    parser.add_argument(
        "--downstream_parallel_gpus",
        type=str,
        default="",
        help=(
            "Comma-separated CUDA GPU ids for task-parallel downstream evaluation "
            "when replaying final candidates, e.g. '0,1,2'. Use 'auto' for all visible GPUs. "
            "Leave empty to run downstream evaluation in the current process."
        ),
    )
    parser.add_argument(
        "--proposed_save_pareto",
        type=str,
        default="proposed_pareto_front.csv",
        dest="proposed_save_pareto",
        help="Where to save the Pareto front CSV emitted by the proposed method (timestamp is appended).",
    )
    parser.add_argument(
        "--proposed_pareto_plot",
        type=str,
        default="proposed_pareto_front.png",
        dest="proposed_pareto_plot",
        help="Where to save the Pareto scatter-plot (timestamp is appended).",
    )
    parser.add_argument(
        "--pipeline_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Stable JSON checkpoint used to persist stage-1/stage-2 progress and resume "
            "interrupted runs. Defaults to checkpoints/<model>_<method>_..._pipeline_state.json."
        ),
    )
    parser.add_argument(
        "--pipeline_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the JSON pipeline checkpoint used for resume/eval-only flows. "
            "Use --no-pipeline_checkpoint for storage-light ablation runs that should "
            "only keep CSV/plot artifacts."
        ),
    )
    parser.add_argument(
        "--training_log_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable writing a per-run timestamped CSV summary with one row per final candidate, "
            "including PPL/downstream results when enabled."
        ),
    )
    parser.add_argument(
        "--training_log_csv_path",
        type=str,
        default=None,
        help=(
            "Optional base CSV path or directory for the training summary. A timestamp is appended "
            "so each run writes a different CSV file. Defaults to logs/<model>_<method>_..._training_log_<timestamp>.csv."
        ),
    )
    parser.add_argument(
        "--force_fresh_run",
        action="store_true",
        help=(
            "Ignore any existing pipeline checkpoint for this run and start from scratch. "
            "A new checkpoint will still be written as the run progresses."
        ),
    )
    parser.add_argument(
        "--checkpoint_eval_only",
        action="store_true",
        help=(
            "Skip optimization and run evaluation only from the selected candidate stored "
            "in the pipeline checkpoint."
        ),
    )
    parser.add_argument(
        "--save_selected_stage2_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save the final selected stage-2-pruned model to disk so it can be benchmarked "
            "later on other hardware."
        ),
    )
    parser.add_argument(
        "--selected_stage2_model_dir",
        type=str,
        default="selected_stage2_models",
        help="Root directory where the selected stage-2-pruned model will be exported.",
    )
    parser.add_argument(
        "--proposed_eval_saved_pareto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load all solutions from the saved Pareto front CSV and re-evaluate them (e.g., Hellaswag) instead of rerunning search.",
    )
    parser.add_argument(
        "--proposed_pareto_selection",
        type=str,
        choices=["best_ppl", "best_latency", "sorted", "all"],
        default="best_ppl",
        help=(
            "Final handling for proposed Pareto candidates after BO. 'best_ppl' keeps "
            "the candidate with the lowest final BO calibration score from the "
            "JSON prompt sets. 'best_latency' keeps the candidate with the lowest "
            "stage-2 BO/final latency score. 'sorted' reports all final candidates "
            "sorted by latency, then BO calibration score, and evaluates them on downstream tasks. "
            "'all' reports all final candidates in candidate order and evaluates them "
            "on downstream tasks."
        ),
    )
    parser.add_argument(
        "--proposed_return_pareto",
        action="store_true",
        default=False,
        help="Return the Pareto payload (with all solutions) for downstream final evaluation.",
    )
    parser.add_argument(
        "--proposed_alpha",
        type=float,
        default=0.5,
        help="Alpha weight (0-1) for S = alpha*acc_norm - (1-alpha)*param_norm when ranking Pareto solutions.",
    )
    parser.add_argument(
        "--proposed_first_objective",
        type=str,
        choices=["kl", "perplexity", "kl_mix", "ppl_mix"],
        default="kl",
        help=(
            "Primary objective for proposed search: calibration KL/perplexity on the "
            "default dataset, or KL/perplexity on the mixed calibration prompt set."
        ),
    )
    parser.add_argument(
        "--proposed_second_objective",
        type=str,
        choices=["params", "latency"],
        default="latency",
        help="Second objective for proposed search: active parameters or latency delta vs. baseline.",
    )
    parser.add_argument(
        "--proposed_pop_size",
        type=int,
        default=5,
        help="Population size for the proposed multi-objective search (NSGA-II).",
    )
    parser.add_argument(
        "--proposed_n_gen",
        type=int,
        default=10,
        help="Number of generations for the proposed multi-objective search (NSGA-II).",
    )
    parser.add_argument(
        "--proposed_debug_random_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use random target logits for KL objective (debug only; keeps compute low but results are meaningless).",
    )
    parser.add_argument(
        "--proposed_debug_random_fitness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use random fitness scores during search (debug only; do not enable for real runs).",
    )
    parser.add_argument(
        "--stage1_use_surrogate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use surrogate models for stage-1 objectives instead of direct evaluation.",
    )
    parser.add_argument(
        "--stage1_surrogate_true_fraction",
        type=float,
        default=0.2,
        help="Fraction of each generation evaluated with true objectives when using surrogates.",
    )
    parser.add_argument(
        "--stage1_calibration_tokens",
        type=int,
        default=262144,
        help=(
            "Approximate number of calibration tokens used to build the stage-1 KL set. "
            "Lower this on limited hardware to avoid storing excessive target logits."
        ),
    )
    parser.add_argument(
        "--stage1_calibration_sequence_length",
        type=int,
        default=None,
        help=(
            "Sequence length used to build the stage-1 KL calibration set. "
            "Defaults to --seq_len when omitted."
        ),
    )
    parser.add_argument(
        "--stage1_tokens_per_evaluation",
        type=int,
        default=32768,
        help=(
            "Approximate number of calibration tokens sampled for each stage-1 fitness "
            "evaluation. Reduce this to cut compute and memory further."
        ),
    )
    parser.add_argument(
        "--stage1_importance_penalty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Precompute per-layer MHA/MLP importance on the stage-1 calibration prompt set "
            "(preferring calibration200 when available) and add a penalty when stage-1 "
            "NSGA prunes higher-importance blocks."
        ),
    )
    parser.add_argument(
        "--stage1_importance_penalty_weight",
        type=float,
        default=0.1,
        help=(
            "Weight applied to the optional stage-1 importance penalty. Higher values "
            "push NSGA to prefer pruning lower-importance MHA/MLP blocks."
        ),
    )
    parser.add_argument(
        "--proposed_enforce_block_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce a fixed pruning budget across attention/MLP masks during proposed search.",
    )
    parser.add_argument(
        "--second_stage_max_ratio",
        type=float,
        default=0.7,
        help=(
            "Maximum fraction (0 < x <= 1.0) of original heads/key/value dimensions kept during the "
            "second-stage search (values < 1.0 exclude the unpruned configuration)."
        ),
    )
    parser.add_argument(
        "--second_stage_layer_keep_ratios",
        type=str,
        default=None,
        help=(
            "Optional per-layer keep ratios for stage-2 pruning (overrides --second_stage_max_ratio). "
            "Accepts a list like '[0.9, 0.8, ...]', a dict like '{0: 0.9, 3: 0.7}', "
            "or 'random' to sample a ratio per layer."
        ),
    )
    parser.add_argument(
        "--second_stage_allocation_method",
        type=str,
        choices=[
            "auto",
            "proposed",
            "manual",
            "uniform",
            "er",
            "er_plus",
            "owl",
            "dsa",
            "dlp",
            "lsa",
            "beta",
        ],
        default="auto",
        help=(
            "Layerwise sparsity-allocation method for stage-2. "
            "'proposed' uses the current BO-based proposed method, "
            "'manual' uses --second_stage_layer_keep_ratios, 'uniform' keeps all layers equal, "
            "'er'/'er_plus' use ER-style structural allocation heuristics, 'owl' uses a weight-outlier proxy, "
            "'dsa' uses aggregated stage-2 importance statistics, 'dlp' combines weight and activation signals, "
            "'lsa' uses a reconstruction-error proxy, and 'beta' redistributes keep ratios around the "
            "global stage-2 keep ratio using a scalar beta/alpha parameter."
        ),
    )
    parser.add_argument(
        "--second_stage_allocation_beta",
        type=float,
        default=0.5,
        help="Beta parameter for second_stage_allocation_method=beta. Range is clamped to [0, 1].",
    )
    parser.add_argument(
        "--second_stage_allocation_alpha",
        type=float,
        default=None,
        help="Alias for --second_stage_allocation_beta for beta-based layerwise allocation baselines.",
    )
    parser.add_argument(
        "--second_stage_bayes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Bayesian optimization to allocate per-layer pruning ratios for stage-2 "
            "while preserving a global prune ratio "
            "(monotone: lower pruning in early layers, higher pruning in deeper layers)."
        ),
    )
    parser.add_argument(
        "--second_stage_bayes_trials",
        type=int,
        default=25,
        help="Number of Bayesian optimization trials for per-layer pruning allocation.",
    )
    parser.add_argument(
        "--second_stage_bayes_warmup",
        type=int,
        default=5,
        help="Number of random warmup trials before Bayesian optimization.",
    )
    parser.add_argument(
        "--second_stage_bayes_objective",
        type=str,
        choices=["ppl", "latency"],
        default="ppl",
        help=(
            "Objective optimized by stage-2 Bayesian optimization. 'ppl' minimizes "
            "prompt-set perplexity; 'latency' minimizes CUDA forward-pass latency."
        ),
    )
    parser.add_argument(
        "--second_stage_bayes_latency_batch_size",
        type=int,
        default=1,
        help="Batch size for stage-2 BO latency scoring when --second_stage_bayes_objective=latency.",
    )
    parser.add_argument(
        "--second_stage_bayes_latency_seq_len",
        type=int,
        default=64,
        help="Tokenized input length for stage-2 BO latency scoring.",
    )
    parser.add_argument(
        "--second_stage_bayes_latency_runs",
        type=int,
        default=5,
        help="Measured CUDA forward-pass runs per stage-2 BO latency trial.",
    )
    parser.add_argument(
        "--second_stage_bayes_latency_warmup",
        type=int,
        default=2,
        help="Warmup CUDA forward-pass runs per stage-2 BO latency trial.",
    )
    parser.add_argument(
        "--second_stage_bayes_latency_prompt",
        type=str,
        default="The Leaning Tower of Pisa is known for",
        help="Prompt used to build the fixed input batch for stage-2 BO latency scoring.",
    )
    parser.add_argument(
        "--second_stage_bayes_parallel_gpus",
        type=str,
        default="",
        help=(
            "Comma-separated CUDA GPU ids for parallel stage-2 BO trial evaluation, e.g. '0,1'. "
            "Use 'auto' for all visible GPUs. Leave empty for sequential BO."
        ),
    )
    parser.add_argument(
        "--second_stage_bayes_candidate_parallel_gpus",
        type=str,
        default="",
        help=(
            "Comma-separated CUDA GPU ids for candidate-parallel stage-2 BO, e.g. '0,1,2'. "
            "Each Pareto candidate runs an independent BO loop on one GPU. Use 'auto' for all visible GPUs."
        ),
    )
    parser.add_argument(
        "--bo_eval_prompt_files",
        nargs="+",
        default=list(DEFAULT_BO_EVAL_PROMPT_FILES),
        help=(
            "JSON prompt files used to score BO candidates. Each file should contain a list "
            "of prompt objects with a 'text' field."
        ),
    )
    parser.add_argument(
        "--second_stage_bayes_min_keep_ratio",
        type=float,
        default=0.01,
        help=(
            "Minimum keep ratio enforced for every layer during BO-based stage-2 sparsity "
            "allocation. Prevents any layer from being pruned to nearly zero."
        ),
    )
    parser.add_argument(
        "--second_stage_bayes_early_layer_count",
        type=int,
        default=2,
        help=(
            "Number of earliest layers given stronger BO keep-ratio protection during "
            "stage-2 sparsity allocation."
        ),
    )
    parser.add_argument(
        "--second_stage_bayes_early_min_keep_ratio",
        type=float,
        default=None,
        help=(
            "Optional hard minimum keep ratio enforced for the first "
            "--second_stage_bayes_early_layer_count layers during BO allocation. "
            "If omitted, early-layer protection is applied with a relative prune cap instead."
        ),
    )
    parser.add_argument(
        "--second_stage_bayes_early_max_prune_factor",
        type=float,
        default=0.80,
        help=(
            "Relative cap for pruning the first --second_stage_bayes_early_layer_count layers "
            "during BO allocation. For example, with target prune ratio 0.7 and factor 0.75, "
            "the protected early layers can prune at most 0.525."
        ),
    )
    parser.add_argument(
        "--second_stage_importance_metric",
        type=str,
        choices=["l1", "l2", "mean", "abs_mean"],
        default="l1",
        help="Importance metric used for heads/MLP/norm/embedding scoring during stage-2 pruning.",
    )
    parser.add_argument(
        "--second_stage_method",
        type=str,
        choices=[
            "importance",
            "wanda-sp",
            "wanda_sp",
            "taylor",
            "gradient",
            "wactigrad",
            "flap",
        ],
        default="importance",
        help=(
            "Scoring backend for stage-2 structured pruning: the original importance-based method, "
            "WANDA-SP, Taylor-only salience, fused WActiGrad scoring (selected by "
            "'gradient' or 'wactigrad'), or FLAP."
        ),
    )
    parser.add_argument(
        "--second_stage_flap_metric",
        type=str,
        choices=["IFV", "WIFV", "WIFN"],
        default="WIFV",
        help="FLAP metric variant to use when --second_stage_method=flap.",
    )
    parser.add_argument(
        "--second_stage_heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable attention head pruning during the second-stage search.",
    )
    parser.add_argument(
        "--second_stage_mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable MLP neuron pruning during the second-stage search.",
    )
    parser.add_argument(
        "--second_stage_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable embedding-dimension pruning during the second-stage search.",
    )
    parser.add_argument(
        "--stage2_only",
        action="store_true",
        help="Skip stage-1 pruning/search and run only the stage-2 pruning step.",
    )
    parser.add_argument(
        "--stage1_only",
        action="store_true",
        help=(
            "Run only the stage-1 pruning/search, save the pipeline checkpoint, "
            "and exit before any stage-2 candidate evaluation."
        ),
    )
    parser.add_argument(
        "--second_stage_prompt_set",
        type=str,
        choices=["alpaca", "calibration_mix", "c4_train", "wikitext2", "fineweb_edu"],
        default="calibration_mix",
        help=(
            "Prompt source for second-stage importance "
            "(alpaca, local calibration/*.json mix, C4 train, Wikitext2 train, or fineweb-edu). "
            "importance, wanda-sp, and flap always use the local calibration mix."
        ),
    )
    parser.add_argument(
        "--second_stage_num_prompts",
        type=int,
        default=512,
        help="Number of prompts used to compute second-stage importance.",
    )
    parser.add_argument(
        "--second_stage_prompt_max_length",
        type=int,
        default=512,
        help="Max token length for second-stage prompts (truncates long samples).",
    )
    parser.add_argument(
        "--second_stage_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace per-layer normalization with identity during second-stage pruning.",
    )
    parser.add_argument(
        "--proposed_load_solutions",
        type=str,
        dest="proposed_load_solutions",
        help="Path to a CSV/JSON file containing pre-computed proposed-method masks.",
    )
    parser.add_argument(
        "--proposed_structural_prune",
        action="store_true",
        dest="proposed_structural_prune",
        help="Convert proposed-method masks into structural pruning (irreversible).",
    )
    parser.add_argument(
        "--recovery_finetune",
        action="store_true",
        help="Run recovery fine-tuning after second-stage pruning.",
    )
    parser.add_argument(
        "--recovery_bayes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Bayesian optimization to tune recovery LoRA parameters "
            "(r, alpha, dropout, learning rate, warmup ratio)."
        ),
    )
    parser.add_argument(
        "--recovery_bayes_trials",
        type=int,
        default=15,
        help="Number of Bayesian optimization trials for recovery LoRA tuning.",
    )
    parser.add_argument(
        "--recovery_bayes_warmup",
        type=int,
        default=3,
        help="Number of random warmup trials before BO for recovery tuning.",
    )
    parser.add_argument(
        "--recovery_arch",
        type=str,
        choices=["llama"],
        default="llama",
        help="Model architecture for recovery fine-tuning (passed to fine_tune.py).",
    )
    parser.add_argument(
        "--recovery_epochs",
        type=float,
        default=1,
        help="Number of epochs for recovery fine-tuning.",
    )
    parser.add_argument(
        "--recovery_save_steps",
        type=int,
        default=200,
        help="Checkpoint save interval (steps) for recovery fine-tuning.",
    )
    parser.add_argument(
        "--recovery_logging_steps",
        type=int,
        default=25,
        help="Logging interval (in steps) for recovery fine-tuning.",
    )
    parser.add_argument(
        "--recovery_warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for recovery fine-tuning (used when BO is disabled).",
    )
    parser.add_argument(
        "--recovery_num_samples",
        type=int,
        default=50,
        help="Optional limit on the number of Alpaca examples for recovery fine-tuning.",
    )
    parser.add_argument(
        "--recovery_target_modules",
        type=str,
        default=None,
        help="Comma-separated target modules for LoRA recovery fine-tuning.",
    )
    parser.add_argument(
        "--recovery_report_to",
        type=str,
        default=None,
        help="Reporting backend for recovery fine-tuning (passed to fine_tune.py).",
    )
    parser.add_argument(
        "--recovery_finetune_gpus",
        type=str,
        default="",
        help=(
            "Comma-separated GPU ids for recovery fine-tuning. If two or more ids are "
            "provided, recovery fine-tuning is launched with torchrun/DDP in a subprocess."
        ),
    )
    parser.add_argument(
        "--recovery_output_dir",
        type=str,
        default="recovery_finetuning",
        help="Directory for recovery fine-tuning checkpoints.",
    )
    parser.add_argument(
        "--local_datasets",
        action="store_true",
        help="Load datasets from the ./data directory instead of Hugging Face Hub.",
    )
    normalized_argv = []
    for token in sys.argv[1:] if argv is None else argv:
        normalized = token.lstrip("-")
        if token.startswith("---") and normalized and not normalized[0].isdigit():
            normalized_argv.append(f"--{normalized}")
            continue
        normalized_argv.append(token)
    return parser.parse_args(normalized_argv)
