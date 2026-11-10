"""A small interface for the complete two-stage pruning method."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .profiles import PROFILES


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Hugging Face model ID or local path"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.375,
        help="Target parameter sparsity, between 0 and 1",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="quick",
        help="quick checks the pipeline; paper uses the reported search budgets",
    )
    parser.add_argument(
        "--calibration", choices=["wikitext2", "mixed"], default="wikitext2"
    )
    parser.add_argument(
        "--importance",
        choices=["importance", "wanda-sp", "flap", "gradient", "taylor", "wactigrad"],
        default="importance",
    )
    parser.add_argument(
        "--objective",
        choices=["latency", "params"],
        default="latency",
        help="Stage-1 efficiency objective",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="Visible GPU indices for parallel Pareto candidates, e.g. 0,1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run final perplexity and downstream evaluation",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Export the selected model as well as replay metadata",
    )
    parser.add_argument(
        "--stage1-only", action="store_true", help="Stop after coarse Pareto search"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore a previous checkpoint for this run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full research command without loading models or creating files",
    )
    return parser


def parse_args(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if not 0 < args.sparsity < 1:
        parser.error("--sparsity must be strictly between 0 and 1")
    if args.gpus:
        parts = args.gpus.split(",")
        if not all(part.strip().isdigit() for part in parts):
            parser.error("--gpus must contain comma-separated nonnegative GPU indices")
        indices = [int(part.strip()) for part in parts]
        if len(set(indices)) != len(indices):
            parser.error("--gpus cannot repeat the same GPU index")
        args.gpus = ",".join(map(str, indices))
    model_path = Path(args.model).expanduser()
    if model_path.is_dir():
        args.model = str(model_path.resolve())
    slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.model)
    if args.output_dir is None:
        args.output_dir = (
            Path("outputs") / slug / f"{args.profile}_{args.sparsity}_seed_{args.seed}"
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def build_research_args(args):
    """Translate the public options into explicit, reproducible research settings."""
    budget = PROFILES[args.profile]
    output = args.output_dir
    command = [
        "--model",
        args.model,
        "--method",
        "proposed",
        "--sparsity",
        str(args.sparsity),
        "--seed",
        str(args.seed),
        "--calibration_dataset",
        "wikitext2",
        "--seq_len",
        str(budget.sequence_length),
        "--num_calibration_samples",
        str(budget.samples),
        "--stage1_calibration_tokens",
        str(budget.calibration_tokens),
        "--stage1_calibration_sequence_length",
        str(budget.sequence_length),
        "--stage1_tokens_per_evaluation",
        str(budget.tokens_per_evaluation),
        "--proposed_first_objective",
        "kl_mix" if args.calibration == "mixed" else "kl",
        "--proposed_second_objective",
        args.objective,
        "--proposed_pop_size",
        str(budget.population),
        "--proposed_n_gen",
        str(budget.generations),
        "--second_stage_heads",
        "--second_stage_mlp",
        "--second_stage_bayes",
        "--second_stage_bayes_trials",
        str(budget.bo_trials),
        "--second_stage_bayes_warmup",
        str(budget.bo_warmup),
        "--second_stage_bayes_objective",
        "ppl",
        "--second_stage_method",
        args.importance,
        "--second_stage_prompt_set",
        "calibration_mix" if args.calibration == "mixed" else "wikitext2",
        "--second_stage_num_prompts",
        str(budget.importance_prompts),
        "--pipeline_checkpoint_path",
        str(output / "pipeline_checkpoint.json"),
        "--proposed_save_pareto",
        str(output / "pareto.csv"),
        "--proposed_pareto_plot",
        str(output / "pareto.png"),
        "--training_log_csv_path",
        str(output / "training.csv"),
        "--selected_stage2_model_dir",
        str(output / "selected_model"),
        "--recovery_output_dir",
        str(output / "recovery"),
        (
            "--save_selected_stage2_model"
            if args.save_model
            else "--no-save_selected_stage2_model"
        ),
    ]
    if args.gpus:
        command += ["--second_stage_bayes_candidate_parallel_gpus", args.gpus]
    if args.evaluate:
        command += ["--evaluate_perplexity_all", "--evaluate_downstream"]
    if args.stage1_only:
        command.append("--stage1_only")
    if args.fresh:
        command.append("--force_fresh_run")
    return command


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def main(argv=None):
    args = parse_args(argv)
    research_args = build_research_args(args)
    command = [sys.executable, "-m", "llm_pruning.advanced", *research_args]
    print(
        subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command),
        flush=True,
    )
    if args.dry_run:
        return
    from .advanced_cli import parse_args as parse_research_args
    from .pipeline import main as run_pipeline

    settings = parse_research_args(research_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with working_directory(args.output_dir):
        run_pipeline(settings)


if __name__ == "__main__":
    main()
