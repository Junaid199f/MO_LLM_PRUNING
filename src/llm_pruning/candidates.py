"""Candidate mask parsing and saved Pareto-front loading."""

from llm_pruning.reporting import append_timestamp_to_path
from llm_pruning.allocation import _resolve_model_compute_device
from llm_pruning.checkpoints import _build_run_artifact_filename
import ast
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List
from llm_pruning.core.proposed_multiobjective import proposed_multiobjective
from llm_pruning.core.pruning import blockpruner


def _resolve_default_run_artifact_path(
    args,
    *,
    configured_path: str,
    default_basename: str,
    suffix: str,
    extension: str,
) -> str:
    """
    Replace generic default artifact names with model/sparsity-specific ones.

    Custom user-provided paths are preserved.
    """
    configured = Path(configured_path)
    if configured.name != default_basename:
        return configured_path
    return str(
        configured.with_name(
            _build_run_artifact_filename(args, suffix=suffix, extension=extension)
        )
    )


def _parse_mask_field(field):
    """Normalize mask fields read from CSV/JSON."""
    if isinstance(field, str):
        field = field.strip()
        if not field:
            return []
        try:
            field = ast.literal_eval(field)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Unable to parse mask entry: {field}") from exc
    if isinstance(field, (list, tuple)):
        return [int(x) for x in field]
    raise ValueError(f"Unsupported mask format: {type(field)}")


def _parse_layer_keep_ratios(field):
    """Parse per-layer keep ratios from CLI input."""
    if field is None:
        return None
    if isinstance(field, str) and field.strip().lower() in {"random", "rand"}:
        return "random"
    if isinstance(field, str):
        field = field.strip()
        if not field:
            return None
        try:
            field = ast.literal_eval(field)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Unable to parse layer keep ratios: {field}") from exc
    if isinstance(field, dict):
        parsed = {}
        for key, value in field.items():
            parsed[int(key)] = float(value) if value is not None else None
        return parsed
    if isinstance(field, (list, tuple)):
        return [float(value) if value is not None else None for value in field]
    raise ValueError(f"Unsupported layer keep ratios format: {type(field)}")


def _parse_stage2_layer_stats(field):
    """Parse serialized stage-2 per-layer pruning records from CSV/JSON."""
    if field is None:
        return []
    if isinstance(field, str):
        field = field.strip()
        if not field:
            return []
        try:
            field = ast.literal_eval(field)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Unable to parse stage2_layer_stats: {field}") from exc
    if not isinstance(field, (list, tuple)):
        raise ValueError(f"Unsupported stage2_layer_stats format: {type(field)}")

    parsed = []
    for item in field:
        if not isinstance(item, dict):
            continue
        parsed_item = dict(item)
        if "layer_idx" in parsed_item and parsed_item["layer_idx"] is not None:
            parsed_item["layer_idx"] = int(parsed_item["layer_idx"])
        for key in (
            "total_heads",
            "total_mlp_units",
            "num_pruned_heads",
            "num_pruned_mlp_units",
            "heads_pruned",
            "mlp_pruned",
        ):
            if key in parsed_item and parsed_item[key] is not None:
                parsed_item[key] = int(parsed_item[key])
        for key in ("pruned_head_indices", "pruned_mlp_indices"):
            parsed_item[key] = [int(index) for index in parsed_item.get(key, []) or []]
        parsed.append(parsed_item)
    return parsed


def load_proposed_solutions(path: str):
    """Load Proposed pruning masks from a CSV/JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Proposed solutions file not found: {path}")

    ext = file_path.suffix.lower()
    records = []

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            if isinstance(data.get("mask_candidates"), list):
                records = data["mask_candidates"]
            elif isinstance(data.get("solutions"), list):
                records = data["solutions"]
            elif isinstance(data.get("selected"), list):
                records = data["selected"]
            else:
                records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(
                f"Unexpected JSON structure in {path}; expected list or dict with "
                "'mask_candidates', 'solutions', or 'selected'."
            )
    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row:
                    records.append(row)
    else:
        raise ValueError(f"Unsupported solutions format '{ext}'. Provide CSV or JSON.")

    solutions = []
    for entry in records:
        attn_mask = _parse_mask_field(entry.get("attn"))
        mlp_field = entry.get("mlp", [])
        mlp_mask = _parse_mask_field(mlp_field)
        if not mlp_mask and mlp_field in (None, "", []):
            mlp_mask = list(attn_mask)
        solution = {"attn": attn_mask, "mlp": mlp_mask}
        if "stage2_keep_ratio" in entry:
            try:
                if entry["stage2_keep_ratio"] not in (None, ""):
                    solution["stage2_keep_ratio"] = float(entry["stage2_keep_ratio"])
            except (TypeError, ValueError):
                pass
        if "stage2_layer_keep_ratios" in entry:
            try:
                solution["stage2_layer_keep_ratios"] = _parse_layer_keep_ratios(
                    entry["stage2_layer_keep_ratios"]
                )
            except ValueError:
                pass
        if "stage2_allocation_method" in entry and entry[
            "stage2_allocation_method"
        ] not in (None, ""):
            solution["stage2_allocation_method"] = str(
                entry["stage2_allocation_method"]
            )
        if "stage2_allocation_params" in entry:
            try:
                raw_value = entry["stage2_allocation_params"]
                if raw_value not in (None, ""):
                    if isinstance(raw_value, str):
                        solution["stage2_allocation_params"] = ast.literal_eval(
                            raw_value
                        )
                    else:
                        solution["stage2_allocation_params"] = raw_value
            except (TypeError, ValueError, SyntaxError):
                pass
        if "stage2_layer_stats" in entry:
            try:
                solution["stage2_layer_stats"] = _parse_stage2_layer_stats(
                    entry["stage2_layer_stats"]
                )
            except ValueError:
                pass
        if "stage2_importance_metric" in entry and entry[
            "stage2_importance_metric"
        ] not in (None, ""):
            solution["stage2_importance_metric"] = entry["stage2_importance_metric"]
        if "fitness" in entry:
            try:
                solution["fitness"] = float(entry["fitness"])
            except (TypeError, ValueError):
                pass
        if "active_params" in entry:
            try:
                solution["active_params"] = float(entry["active_params"])
            except (TypeError, ValueError):
                pass
        if "latency" in entry:
            try:
                solution["latency"] = float(entry["latency"])
            except (TypeError, ValueError):
                pass
        solutions.append(solution)

    if not solutions:
        raise ValueError(f"No proposed solutions could be read from {path}.")
    return solutions


def load_proposed_history_points(path: str) -> List[Dict[str, object]]:
    """Load stage-1 search history records used for Pareto plotting."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: List[Dict[str, object]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            record: Dict[str, object] = {}
            for key in ("fitness", "active_params", "latency"):
                if key in row:
                    try:
                        record[key] = float(row[key])
                    except (TypeError, ValueError):
                        record[key] = None
            if "first_objective" in row:
                record["first_objective"] = row["first_objective"]
            records.append(record)
    return records


def build_mask_candidates(
    args,
    model,
    tokenizer,
    calibration_corpus,
    first_calibration_sample,
    target_blocks,
    stage1_first_objective,
    stage1_ppl_input_ids=None,
    stage1_importance_scores=None,
):
    """Generate pruning mask candidates based on the selected method."""
    if args.method == "blockpruner":
        attn_mask, mlp_mask = blockpruner(
            model, target_blocks, first_calibration_sample
        )
        return [{"attn": attn_mask, "mlp": mlp_mask}]

    if args.method == "proposed":
        if args.proposed_eval_saved_pareto:
            logging.info(
                "Loading proposed Pareto front from %s for evaluation.",
                args.proposed_save_pareto,
            )
            candidates = load_proposed_solutions(args.proposed_save_pareto)
            if not candidates:
                return []
            return candidates

        if args.proposed_load_solutions:
            logging.info(
                "Loading proposed solutions from %s", args.proposed_load_solutions
            )
            candidates = load_proposed_solutions(args.proposed_load_solutions)
            if not candidates:
                return []
            return candidates

        # Generate model/sparsity-specific timestamped output paths to avoid collisions.
        save_pareto_base = _resolve_default_run_artifact_path(
            args,
            configured_path=args.proposed_save_pareto,
            default_basename="proposed_pareto_front.csv",
            suffix="pareto_front",
            extension=".csv",
        )
        pareto_plot_base = _resolve_default_run_artifact_path(
            args,
            configured_path=args.proposed_pareto_plot,
            default_basename="proposed_pareto_front.png",
            suffix="pareto_front",
            extension=".png",
        )
        save_pareto_path = append_timestamp_to_path(save_pareto_base)
        pareto_plot_path = append_timestamp_to_path(pareto_plot_base)
        logging.info(
            "Saving Pareto artifacts to %s (CSV) and %s (plot).",
            str(Path(save_pareto_path).resolve()),
            str(Path(pareto_plot_path).resolve()),
        )
        args.proposed_save_pareto = save_pareto_path
        args.proposed_pareto_plot = pareto_plot_path

        second_objective = args.proposed_second_objective
        model_device = _resolve_model_compute_device(model)
        if second_objective == "latency" and model_device.type != "cuda":
            logging.warning(
                "Latency objective requires CUDA, but model is on %s; falling back to params.",
                model_device,
            )
            second_objective = "params"

        result = proposed_multiobjective(
            model,
            target_blocks,
            tokenizer,
            calibration_corpus,
            drop_entire_block=False,
            calibration_tokens=args.stage1_calibration_tokens,
            calibration_sequence_length=(
                args.stage1_calibration_sequence_length
                if args.stage1_calibration_sequence_length is not None
                else args.seq_len
            ),
            tokens_per_evaluation=args.stage1_tokens_per_evaluation,
            save_pareto_path=save_pareto_path,
            pareto_plot_path=pareto_plot_path,
            pareto_selection="all",
            debug_random_fitness=args.proposed_debug_random_fitness,
            debug_random_targets=args.proposed_debug_random_targets,
            pop_size=args.proposed_pop_size,
            n_gen=args.proposed_n_gen,
            first_objective=stage1_first_objective,
            second_objective=second_objective,
            enforce_block_budget=args.proposed_enforce_block_budget,
            ppl_input_ids=stage1_ppl_input_ids,
            ppl_seq_len=args.seq_len,
            return_pareto=args.proposed_return_pareto,
            stage1_use_surrogate=args.stage1_use_surrogate,
            stage1_surrogate_true_fraction=args.stage1_surrogate_true_fraction,
            stage1_importance_scores=stage1_importance_scores,
            stage1_importance_penalty_weight=args.stage1_importance_penalty_weight,
        )
        if args.proposed_return_pareto:
            selected = result.get("selected", [])
            if isinstance(selected, dict):
                selected = [selected]
            candidates = []
            for sol in selected:
                candidates.append(
                    {
                        "attn": sol.get("attn"),
                        "mlp": sol.get("mlp"),
                        "fitness": sol.get("fitness"),
                        "active_params": sol.get("active_params"),
                        "latency": sol.get("latency"),
                    }
                )
            return candidates

        candidates = []
        for sol in result:
            candidates.append(
                {
                    "attn": sol.get("attn"),
                    "mlp": sol.get("mlp"),
                    "fitness": sol.get("fitness"),
                    "active_params": sol.get("active_params"),
                    "latency": sol.get("latency"),
                }
            )
        return candidates

    return []
