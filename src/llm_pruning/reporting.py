"""Artifact paths and pruning visualizations."""

import logging
import math
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


def append_timestamp_to_path(path: str) -> str:
    """Append a timestamp to the filename portion of a path."""
    if not path:
        return path

    path_obj = Path(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(path_obj.suffixes)
    base = path_obj.name[: -len(suffix)] if suffix else path_obj.name
    new_name = f"{base.rstrip('.')}_{timestamp}{suffix}"
    return str(path_obj.with_name(new_name))


def _slugify_filename(value: str) -> str:
    """Convert a free-form identifier into a filesystem-friendly filename stem."""
    if not value:
        return "model"
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "model"


def ensure_directory_exists(path: str) -> Path:
    """Create a directory if it does not already exist and return its Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _layer_prune_ratios_from_keep_ratios(keep_ratios: Sequence[float]) -> List[float]:
    """Convert layer keep ratios to clipped layer pruning ratios."""
    return [max(0.0, min(1.0, 1.0 - float(keep_ratio))) for keep_ratio in keep_ratios]


def save_layerwise_allocation_plot(
    keep_ratios: Sequence[float],
    *,
    candidate_idx: int,
    model_name: str,
    graphs_dir: str = "graphs",
) -> Optional[str]:
    """Save a classic per-layer pruning-ratio plot for one candidate."""
    if not keep_ratios:
        logging.warning(
            "Skipping layerwise allocation plot for candidate %d: no keep ratios provided.",
            candidate_idx,
        )
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(
            "Skipping layerwise allocation plot for candidate %d because matplotlib is unavailable: %s",
            candidate_idx,
            exc,
        )
        return None

    graph_dir_path = ensure_directory_exists(graphs_dir)
    model_stem = _slugify_filename(Path(str(model_name)).name)
    output_path = append_timestamp_to_path(
        str(
            graph_dir_path
            / f"{model_stem}_candidate_{candidate_idx}_layerwise_pruning_ratios.png"
        )
    )

    layers = list(range(len(keep_ratios)))
    prune_ratios = _layer_prune_ratios_from_keep_ratios(keep_ratios)
    target_ratio = sum(prune_ratios) / len(prune_ratios) if prune_ratios else 0.0

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhline(
        0.0,
        color="#2f5d2f",
        linestyle=":",
        linewidth=1.8,
        label="Full Keep Baseline (0% pruning)",
    )
    ax.axhline(
        1.0,
        color="#2f5d2f",
        linestyle=":",
        linewidth=1.8,
        label="Full Prune Baseline (100% pruning)",
    )
    ax.axhline(
        target_ratio,
        color="#6b6b6b",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label=f"Mean Pruning Ratio ({target_ratio:.3f})",
    )
    ax.plot(
        layers,
        prune_ratios,
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.8,
        label="Layerwise Pruning Ratio",
    )
    ax.set_xlabel(f"Layer Index (0-{len(layers) - 1}) ->")
    ax.set_ylabel("Pruning Ratio")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Candidate {candidate_idx} Layerwise Pruning Allocation")
    ax.set_xticks(layers)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logging.info(
        "Saved layerwise pruning-ratio plot for candidate %d to %s",
        candidate_idx,
        output_path,
    )
    return output_path


def save_stagewise_pruning_plot(
    stagewise_records: Sequence[Dict[str, float]],
    *,
    model_name: str,
    graphs_dir: str = "graphs",
) -> Optional[str]:
    """Save a stacked bar plot showing stage-1 and stage-2 pruning per candidate."""
    if not stagewise_records:
        logging.warning("Skipping stagewise pruning plot: no records were provided.")
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(
            "Skipping stagewise pruning plot because matplotlib is unavailable: %s", exc
        )
        return None

    graph_dir_path = ensure_directory_exists(graphs_dir)
    model_stem = _slugify_filename(Path(str(model_name)).name)
    output_path = append_timestamp_to_path(
        str(graph_dir_path / f"{model_stem}_stagewise_pruning.png")
    )

    candidate_labels = [
        f"C{int(record['candidate_idx'])}" for record in stagewise_records
    ]
    stage1_pct = []
    stage2_pct = []
    stage1_params = []
    stage2_params = []
    for record in stagewise_records:
        total_params = float(record.get("total_params", 0.0))
        stage1_value = float(record.get("stage1_pruned_params", 0.0))
        stage2_value = float(record.get("stage2_pruned_params", 0.0))
        stage1_params.append(stage1_value)
        stage2_params.append(stage2_value)
        if total_params > 0.0:
            stage1_pct.append(100.0 * stage1_value / total_params)
            stage2_pct.append(100.0 * stage2_value / total_params)
        else:
            stage1_pct.append(0.0)
            stage2_pct.append(0.0)

    fig_width = max(8.0, 1.8 * len(stagewise_records))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x_positions = np.arange(len(candidate_labels))

    ax.bar(x_positions, stage1_pct, color="#4C78A8", alpha=0.9, label="Stage 1 pruned")
    ax.bar(
        x_positions,
        stage2_pct,
        bottom=stage1_pct,
        color="#F58518",
        alpha=0.9,
        label="Stage 2 pruned",
    )

    for idx, (s1_pct, s2_pct, s1_params, s2_params) in enumerate(
        zip(stage1_pct, stage2_pct, stage1_params, stage2_params)
    ):
        if s1_pct > 0.0:
            ax.text(
                idx,
                s1_pct / 2.0,
                f"{s1_pct:.1f}%\n{s1_params / 1e6:.2f}M",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )
        if s2_pct > 0.0:
            ax.text(
                idx,
                s1_pct + (s2_pct / 2.0),
                f"{s2_pct:.1f}%\n{s2_params / 1e6:.2f}M",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )
        total_pct = s1_pct + s2_pct
        ax.text(
            idx,
            total_pct + 1.0,
            f"Total {total_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    max_total_pct = max((s1 + s2) for s1, s2 in zip(stage1_pct, stage2_pct))
    ax.set_ylim(0.0, min(105.0, max(10.0, max_total_pct + 8.0)))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(candidate_labels)
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Pruned Parameters (% of original model)")
    ax.set_title("Stage-1 vs Stage-2 Pruning")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved stagewise pruning plot to %s", output_path)
    return output_path


def save_layer_component_pruning_plot(
    *,
    candidate_idx: int,
    model_name: str,
    stage1_attn_mask: Sequence[int],
    stage1_mlp_mask: Sequence[int],
    stage2_layer_stats: Optional[Sequence[Dict[str, object]]] = None,
    graphs_dir: str = "graphs",
) -> Optional[str]:
    """Save a plot showing stage-1 block drops and stage-2 pruning with 100 bins per layer."""
    if not stage1_attn_mask and not stage1_mlp_mask and not stage2_layer_stats:
        logging.warning(
            "Skipping layer component pruning plot for candidate %d: no layer data provided.",
            candidate_idx,
        )
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        logging.warning(
            "Skipping layer component pruning plot for candidate %d because matplotlib is unavailable: %s",
            candidate_idx,
            exc,
        )
        return None

    num_layers = max(
        len(stage1_attn_mask),
        len(stage1_mlp_mask),
        len(stage2_layer_stats or []),
    )
    if num_layers <= 0:
        return None

    stage1_attn = [
        (
            int(bool(stage1_attn_mask[layer_idx]))
            if layer_idx < len(stage1_attn_mask)
            else 0
        )
        for layer_idx in range(num_layers)
    ]
    stage1_mlp = [
        int(bool(stage1_mlp_mask[layer_idx])) if layer_idx < len(stage1_mlp_mask) else 0
        for layer_idx in range(num_layers)
    ]
    stage2_stats_by_layer = {}
    for layer_stat in stage2_layer_stats or []:
        layer_idx = int(layer_stat.get("layer_idx", -1))
        if 0 <= layer_idx < num_layers:
            stage2_stats_by_layer[layer_idx] = layer_stat

    graph_dir_path = ensure_directory_exists(graphs_dir)
    model_stem = _slugify_filename(Path(str(model_name)).name)
    output_path = append_timestamp_to_path(
        str(
            graph_dir_path
            / f"{model_stem}_candidate_{candidate_idx}_layer_component_pruning.png"
        )
    )

    stage1_heatmap = np.array(
        [
            stage1_attn,
            stage1_mlp,
        ],
        dtype=np.int32,
    )
    num_bins = 100
    stage2_attn_bins = np.zeros((num_bins, num_layers), dtype=np.float32)
    stage2_mlp_bins = np.zeros((num_bins, num_layers), dtype=np.float32)

    def _mark_pruned_positions(
        bin_grid, layer_idx: int, total_items: int, pruned_indices: Sequence[int]
    ) -> None:
        if total_items <= 0 or not pruned_indices:
            return
        valid_indices = sorted(
            {
                int(raw_index)
                for raw_index in pruned_indices
                if 0 <= int(raw_index) < total_items
            }
        )
        if not valid_indices:
            return
        if total_items <= num_bins:
            if total_items == 1:
                bin_grid[0, layer_idx] = 1
                return
            for item_index in valid_indices:
                bin_position = int(
                    round(item_index * (num_bins - 1) / float(total_items - 1))
                )
                bin_position = max(0, min(num_bins - 1, bin_position))
                bin_grid[bin_position, layer_idx] = 1
            return

        pruned_lookup = set(valid_indices)
        for bin_position in range(num_bins):
            sample_index = int(
                round(bin_position * (total_items - 1) / float(num_bins - 1))
            )
            if sample_index in pruned_lookup:
                bin_grid[bin_position, layer_idx] = 1

    for layer_idx in range(num_layers):
        if stage1_attn[layer_idx]:
            stage2_attn_bins[:, layer_idx] = np.nan
        if stage1_mlp[layer_idx]:
            stage2_mlp_bins[:, layer_idx] = np.nan
        layer_stat = stage2_stats_by_layer.get(layer_idx, {})
        if not stage1_attn[layer_idx]:
            _mark_pruned_positions(
                stage2_attn_bins,
                layer_idx,
                int(layer_stat.get("total_heads", 0)),
                layer_stat.get("pruned_head_indices", []),
            )
        if not stage1_mlp[layer_idx]:
            _mark_pruned_positions(
                stage2_mlp_bins,
                layer_idx,
                int(layer_stat.get("total_mlp_units", 0)),
                layer_stat.get("pruned_mlp_indices", []),
            )

    cmap = ListedColormap(["#F6F7F9", "#D04F45"])
    cmap.set_bad(color="#D9D9D9")
    fig_width = max(10.0, 0.38 * num_layers)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(fig_width, 8.8),
        gridspec_kw={"height_ratios": [1.2, 3.8, 3.8]},
        sharex=True,
    )
    ax_stage1, ax_stage2_mha, ax_stage2_mlp = axes

    ax_stage1.imshow(stage1_heatmap, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax_stage1.set_yticks([0, 1])
    ax_stage1.set_yticklabels(["Stage 1 MHA", "Stage 1 MLP"])
    ax_stage1.set_title(f"Candidate {candidate_idx} Layerwise MHA/MLP Pruning")
    ax_stage1.set_xticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax_stage1.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax_stage1.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax_stage1.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(stage1_heatmap.shape[0]):
        for col_idx in range(stage1_heatmap.shape[1]):
            value = int(stage1_heatmap[row_idx, col_idx])
            ax_stage1.text(
                col_idx,
                row_idx,
                "P" if value else ".",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value else "#4F5B66",
            )

    ax_stage2_mha.imshow(
        np.ma.masked_invalid(stage2_attn_bins),
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
    )
    ax_stage2_mha.set_ylabel("Stage 2 MHA\npruned bins")
    ax_stage2_mha.set_yticks([0, 24, 49, 74, 99])
    ax_stage2_mha.set_yticklabels(["1", "25", "50", "75", "100"])
    ax_stage2_mha.set_xticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax_stage2_mha.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
    ax_stage2_mha.tick_params(which="minor", bottom=False, left=False)

    ax_stage2_mlp.imshow(
        np.ma.masked_invalid(stage2_mlp_bins),
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
    )
    ax_stage2_mlp.set_ylabel("Stage 2 MLP\npruned bins")
    ax_stage2_mlp.set_yticks([0, 24, 49, 74, 99])
    ax_stage2_mlp.set_yticklabels(["1", "25", "50", "75", "100"])
    ax_stage2_mlp.set_xticks(np.arange(num_layers))
    ax_stage2_mlp.set_xticklabels(
        [str(layer_idx) for layer_idx in range(num_layers)], rotation=0
    )
    ax_stage2_mlp.set_xticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax_stage2_mlp.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
    ax_stage2_mlp.tick_params(which="minor", bottom=False, left=False)
    ax_stage2_mlp.set_xlabel("Layer")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logging.info(
        "Saved layer component pruning plot for candidate %d to %s",
        candidate_idx,
        output_path,
    )
    return output_path


def save_stage2_pareto_refinement_plot(
    candidates: Sequence[Dict[str, object]],
    *,
    model_name: str,
    stage1_all_solutions: Optional[Sequence[Dict[str, object]]] = None,
    graphs_dir: str = "graphs",
) -> Optional[str]:
    """Save a latency/PPL plot for stage-1 search and post-stage-2 refined solutions."""
    if not candidates:
        logging.warning(
            "Skipping stage-2 Pareto refinement plot: no candidates were provided."
        )
        return None

    def _safe_float(value: object) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _pareto_front(
        points: Sequence[Tuple[float, float, object]],
    ) -> List[Tuple[float, float, object]]:
        """Return the 2D minimization Pareto front sorted by latency."""
        front: List[Tuple[float, float, object]] = []
        best_y = float("inf")
        for point in sorted(points, key=lambda item: (item[0], item[1])):
            if point[1] < best_y:
                front.append(point)
                best_y = point[1]
        return front

    stage1_all_points: List[Tuple[float, float, str]] = []
    if stage1_all_solutions:
        for idx, solution in enumerate(stage1_all_solutions, start=1):
            first_objective = (
                str(solution.get("first_objective", "") or "").strip().lower()
            )
            if first_objective and first_objective != "perplexity":
                continue
            latency = _safe_float(solution.get("latency"))
            ppl = _safe_float(solution.get("fitness"))
            if latency is None or ppl is None:
                continue
            stage1_all_points.append((latency, ppl, f"S{idx}"))

    metric_labels = {
        str(candidate.get("stage1_ppl_label", "") or "").strip()
        for candidate in candidates
        if str(candidate.get("stage1_ppl_label", "") or "").strip()
    }
    metric_labels.update(
        {
            str(candidate.get("final_plot_ppl_label", "") or "").strip()
            for candidate in candidates
            if str(candidate.get("final_plot_ppl_label", "") or "").strip()
        }
    )
    metric_labels.update(
        {
            str(solution.get("first_objective", "") or "").strip().lower()
            for solution in (stage1_all_solutions or [])
            if str(solution.get("first_objective", "") or "").strip()
        }
    )

    metric_label_blob = " ".join(metric_labels).lower()
    if "kl" in metric_label_blob:
        y_axis_label = "KL Divergence"
    elif "perplexity" in metric_label_blob or "ppl" in metric_label_blob:
        y_axis_label = "PPL"
    else:
        y_axis_label = "Fitness"

    stage1_points: List[Tuple[float, float, int]] = []
    stage2_points: List[Tuple[float, float, int]] = []
    stage2_actual_points: List[Tuple[float, float, int]] = []
    stage2_fallback_points: List[Tuple[float, float, int]] = []
    skipped_stage1 = 0
    skipped_stage2 = 0
    fallback_stage2 = 0
    for candidate in candidates:
        candidate_idx = int(candidate.get("candidate_idx", 0) or 0)
        stage1_latency = _safe_float(candidate.get("latency"))
        stage1_ppl = _safe_float(candidate.get("stage1_ppl"))
        if stage1_latency is None or stage1_ppl is None:
            skipped_stage1 += 1
        else:
            stage1_points.append((stage1_latency, stage1_ppl, candidate_idx))

        stage2_label = str(candidate.get("final_plot_ppl_label", "") or "").lower()
        stage2_latency = _safe_float(candidate.get("final_latency"))
        if stage2_latency is None:
            stage2_latency = stage1_latency

        stage2_ppl = _safe_float(candidate.get("final_plot_ppl"))
        if stage2_ppl is None:
            stage2_ppl = stage1_ppl

        if stage2_latency is None or stage2_ppl is None:
            skipped_stage2 += 1
            continue

        point = (stage2_latency, stage2_ppl, candidate_idx)
        stage2_points.append(point)
        if (
            "fallback" in stage2_label
            or _safe_float(candidate.get("final_latency")) is None
        ):
            stage2_fallback_points.append(point)
            fallback_stage2 += 1
        else:
            stage2_actual_points.append(point)

    if not stage1_points:
        logging.warning(
            "Skipping stage-2 Pareto refinement plot because stage-1 latency/PPL values are unavailable."
        )
        return None
    if not stage2_points:
        logging.warning(
            "Skipping stage-2 Pareto refinement plot because no real post-stage-2 latency/PPL values were found."
        )
        return None
    if skipped_stage1:
        logging.warning(
            "Stage-2 Pareto plot skipped %d candidate(s) with missing stage-1 latency/PPL.",
            skipped_stage1,
        )
    if skipped_stage2:
        logging.warning(
            "Stage-2 Pareto plot skipped %d candidate(s) with missing post-stage-2 latency/PPL.",
            skipped_stage2,
        )
    if fallback_stage2:
        logging.warning(
            "Stage-2 Pareto plot reused fallback stage-1 metrics for %d candidate(s) because final stage-2 metrics were unavailable.",
            fallback_stage2,
        )

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(
            "Skipping stage-2 Pareto refinement plot because matplotlib is unavailable: %s",
            exc,
        )
        return None

    stage1_front = _pareto_front(stage1_points)
    stage2_front = _pareto_front(stage2_points)
    graph_dir_path = ensure_directory_exists(graphs_dir)
    model_stem = _slugify_filename(Path(str(model_name)).name)
    output_path = append_timestamp_to_path(
        str(graph_dir_path / f"{model_stem}_pareto_front_stage1_vs_stage2.png")
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    if stage1_all_points:
        ax.scatter(
            [point[0] for point in stage1_all_points],
            [point[1] for point in stage1_all_points],
            marker="x",
            color="#9A9A9A",
            alpha=0.55,
            label="Stage 1 all solutions",
        )

    ax.scatter(
        [point[0] for point in stage1_points],
        [point[1] for point in stage1_points],
        marker="o",
        color="#F28E8B",
        alpha=0.75,
        label="Stage 1 selected solutions",
    )
    ax.plot(
        [point[0] for point in stage1_front],
        [point[1] for point in stage1_front],
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        marker="o",
        markersize=6,
        label="Stage 1 Pareto front",
    )
    if stage2_actual_points:
        ax.scatter(
            [point[0] for point in stage2_actual_points],
            [point[1] for point in stage2_actual_points],
            marker="D",
            color="#6BAF92",
            alpha=0.85,
            label="Post-Stage-2 solutions",
        )
    if stage2_fallback_points:
        ax.scatter(
            [point[0] for point in stage2_fallback_points],
            [point[1] for point in stage2_fallback_points],
            marker="D",
            color="#A8CFA8",
            alpha=0.85,
            label="Post-Stage-2 solutions (fallback metrics)",
        )
    ax.plot(
        [point[0] for point in stage2_front],
        [point[1] for point in stage2_front],
        color="#0A8A1F",
        linestyle="-",
        linewidth=1.8,
        marker="D",
        markersize=7,
        label="Post-Stage-2 Pareto front",
    )

    for latency, ppl, candidate_idx in stage2_points:
        ax.text(
            latency,
            ppl,
            f"C{candidate_idx}",
            fontsize=8,
            ha="left",
            va="bottom",
            color="#0A5C18",
        )

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel(y_axis_label)
    ax.set_title(f"Stage-1 vs Stage-2 Pareto Front ({y_axis_label} vs Latency)")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved stage-2 Pareto refinement plot to %s", output_path)
    return output_path
