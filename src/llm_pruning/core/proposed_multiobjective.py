"""
Multi-objective variant of Proposed that optimises a primary objective
(calibration KL or perplexity) alongside a secondary objective (active
parameters or latency) using pymoo.

Requires pymoo >= 0.6.
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mutation import Mutation
from tqdm import tqdm

from .proposed_utils import collect_samples_with_join, compute_fitness
from .evaluation import evaluate_perplexity, evaluate_inference_time
from .utilities import count_params_with_masks, maskModel, unmaskModel


def _predict_surrogate(model, x: np.ndarray) -> float:
    if model is None:
        raise ValueError("Surrogate model is not loaded.")
    features = np.asarray(x, dtype=np.float32).reshape(1, -1)
    if hasattr(model, "predict"):
        pred = model.predict(features)
        return float(np.asarray(pred).reshape(-1)[0])
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            tensor = torch.from_numpy(features)
            pred = model(tensor)
            return float(pred.detach().cpu().reshape(-1)[0])
    if callable(model):
        pred = model(features)
        return float(np.asarray(pred).reshape(-1)[0])
    raise TypeError(
        "Unsupported surrogate model type; expected predict() or torch module."
    )


def _fit_xgboost_regressor(X: np.ndarray, y: np.ndarray):
    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError(
            "xgboost is required for stage-1 surrogate training. Install xgboost."
        ) from exc
    if X.size == 0:
        raise ValueError("Empty training data for surrogate.")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=0,
    )
    model.fit(X, y)
    return model


def _build_calibration_minibatch(
    calibration_data: List[torch.Tensor],
    target_logits: Optional[List[torch.Tensor]],
    num_tokens: Optional[int],
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Sample a minibatch of calibration sequences totaling roughly num_tokens tokens.

    Returns the selected calibration tensors and their corresponding target logits
    (if available). When num_tokens is None we return the full calibration set.
    """
    if num_tokens is None:
        return calibration_data, target_logits
    if num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if not calibration_data:
        raise ValueError("calibration_data must contain at least one sample")

    minibatch_data: List[torch.Tensor] = []
    minibatch_logits: Optional[List[torch.Tensor]] = (
        [] if target_logits is not None else None
    )
    tokens_used = 0
    sampled_indices = set()

    while (
        tokens_used < num_tokens
        and len(sampled_indices) < len(calibration_data)
        and calibration_data
    ):
        candidate_idx = random.randint(0, len(calibration_data) - 1)
        if candidate_idx in sampled_indices:
            continue
        sampled_indices.add(candidate_idx)

        sample = calibration_data[candidate_idx]
        sample_tokens = sample.shape[1]
        remaining = num_tokens - tokens_used

        if sample_tokens > remaining:
            minibatch_data.append(sample[:, :remaining])
            if minibatch_logits is not None and target_logits is not None:
                minibatch_logits.append(target_logits[candidate_idx][:, :remaining])
            tokens_used = num_tokens
        else:
            minibatch_data.append(sample)
            if minibatch_logits is not None and target_logits is not None:
                minibatch_logits.append(target_logits[candidate_idx])
            tokens_used += sample_tokens

    if minibatch_logits is None:
        return minibatch_data, None
    return minibatch_data, minibatch_logits


class ProposedMaskMutation(Mutation):
    """
    Mutation operator that mirrors the swap-based mutation used in Proposed while
    supporting a shared pruning budget across attention and MLP masks.

    When ``enforce_budget`` is enabled the combined number of pruned attention and
    MLP submodules is kept equal to ``num_prune`` (or exactly ``num_prune`` blocks
    when pruning whole blocks). Otherwise mutations behave like the vanilla
    Proposed swaps.
    """

    def __init__(
        self,
        num_layers: int,
        num_prune: int,
        *,
        drop_entire_block: bool,
        max_mutations: int = 3,
        enforce_budget: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_prune = num_prune
        self.drop_entire_block = drop_entire_block
        self.max_mutations = max(1, max_mutations)
        self.enforce_budget = enforce_budget

    def _repair_segment(self, mask: np.ndarray, target_ones: int) -> None:
        """Project ``mask`` onto a binary vector with exactly ``target_ones`` ones."""
        current_pruned = int(mask.sum())
        if current_pruned == target_ones:
            return
        indices_one = np.where(mask == 1)[0]
        indices_zero = np.where(mask == 0)[0]
        if current_pruned > target_ones and len(indices_one) > 0:
            to_flip = min(current_pruned - target_ones, len(indices_one))
            if to_flip > 0:
                chosen = np.random.choice(indices_one, size=to_flip, replace=False)
                mask[chosen] = 0
        elif current_pruned < target_ones and len(indices_zero) > 0:
            to_flip = min(target_ones - current_pruned, len(indices_zero))
            if to_flip > 0:
                chosen = np.random.choice(indices_zero, size=to_flip, replace=False)
                mask[chosen] = 1

    def _repair_budget(
        self, attn_mask: np.ndarray, mlp_mask: Optional[np.ndarray] = None
    ) -> None:
        """Adjust masks so the enforced pruning budget is respected."""
        if not self.enforce_budget:
            return
        if self.drop_entire_block or mlp_mask is None:
            self._repair_segment(attn_mask, self.num_prune)
            return

        total_pruned = int(attn_mask.sum() + mlp_mask.sum())
        if total_pruned == self.num_prune:
            return

        if total_pruned > self.num_prune:
            over = total_pruned - self.num_prune
            indices = [(0, idx) for idx in np.where(attn_mask == 1)[0]] + [
                (1, idx) for idx in np.where(mlp_mask == 1)[0]
            ]
            if indices:
                chosen = random.sample(indices, k=min(over, len(indices)))
                for mask_id, idx in chosen:
                    if mask_id == 0:
                        attn_mask[idx] = 0
                    else:
                        mlp_mask[idx] = 0
        else:
            needed = self.num_prune - total_pruned
            indices = [(0, idx) for idx in np.where(attn_mask == 0)[0]] + [
                (1, idx) for idx in np.where(mlp_mask == 0)[0]
            ]
            if not indices:
                return
            chosen = random.sample(indices, k=min(needed, len(indices)))
            for mask_id, idx in chosen:
                if mask_id == 0:
                    attn_mask[idx] = 1
                else:
                    mlp_mask[idx] = 1

    def _swap_within_mask(self, mask: np.ndarray) -> None:
        """Randomly swap one kept index with a pruned index inside mask."""
        indices_keep = np.where(mask == 0)[0]
        indices_pruned = np.where(mask == 1)[0]

        if len(indices_keep) == 0 or len(indices_pruned) == 0:
            return

        remove_idx = random.choice(indices_keep.tolist())
        add_idx = random.choice(indices_pruned.tolist())
        mask[remove_idx] = 1
        mask[add_idx] = 0

    def _transfer_between_masks(self, source: np.ndarray, target: np.ndarray) -> None:
        """Move one pruned index from ``source`` to ``target`` while keeping totals."""
        src_pruned = np.where(source == 1)[0]
        tgt_kept = np.where(target == 0)[0]
        if len(src_pruned) == 0 or len(tgt_kept) == 0:
            return
        remove_idx = random.choice(src_pruned.tolist())
        add_idx = random.choice(tgt_kept.tolist())
        source[remove_idx] = 0
        target[add_idx] = 1

    def _do(self, problem, X, **kwargs):
        Y = np.copy(X)

        for i in range(len(Y)):
            vector = Y[i]
            binary_vec = (vector >= 0.5).astype(np.int64)
            attn_mask = binary_vec[: self.num_layers]

            mlp_mask = None
            if not self.drop_entire_block:
                mlp_mask = binary_vec[self.num_layers :]

            # Ensure initial masks respect the prune budget if desired.
            self._repair_budget(attn_mask, mlp_mask)

            num_flips = min(
                random.randint(1, self.max_mutations),
                random.randint(1, self.max_mutations),
            )

            for _ in range(num_flips):
                if self.drop_entire_block:
                    self._swap_within_mask(attn_mask)
                else:
                    mutation_type = random.randint(0, 2)
                    if mutation_type == 0:
                        self._swap_within_mask(attn_mask)
                    elif mutation_type == 1 and mlp_mask is not None:
                        self._swap_within_mask(mlp_mask)
                    else:
                        assert mlp_mask is not None
                        if random.random() < 0.5:
                            self._transfer_between_masks(attn_mask, mlp_mask)
                        else:
                            self._transfer_between_masks(mlp_mask, attn_mask)

            # Re-apply repair in case mutations broke feasibility.
            self._repair_budget(attn_mask, mlp_mask)

            if self.drop_entire_block:
                # MLP mask mirrors attention mask when pruning whole blocks.
                binary_vec[: self.num_layers] = attn_mask
            elif mlp_mask is not None:
                binary_vec[: self.num_layers] = attn_mask
                binary_vec[self.num_layers :] = mlp_mask

            Y[i] = binary_vec.astype(np.float64)

        return Y


class ProposedMaskProblem(ElementwiseProblem):
    """
    pymoo problem that evaluates a binary mask over attention and MLP submodules.

    It minimises two objectives:
      1. Either calibration fitness (KL divergence) or perplexity, chosen via first_objective.
      2. Either the number of active parameters or latency delta vs. baseline.
    """

    def __init__(
        self,
        model,
        calibration_data: List[torch.Tensor],
        target_logits: Optional[List[torch.Tensor]],
        num_prune: int,
        drop_entire_block: bool = False,
        history: Optional[List[Dict[str, object]]] = None,
        enforce_constraints: bool = False,
        tokens_per_evaluation: Optional[int] = None,
        first_objective: str = "kl",
        second_objective: str = "params",
        baseline_latency: Optional[float] = None,
        latency_sample: Optional[torch.Tensor] = None,
        ppl_input_ids: Optional[torch.Tensor] = None,
        ppl_seq_len: int = 2048,
        debug_random_fitness: bool = False,
        use_surrogate: bool = False,
        pop_size: int = 1,
        surrogate_true_fraction: float = 0.2,
        stage1_importance_scores: Optional[Dict[str, object]] = None,
        stage1_importance_penalty_weight: float = 0.0,
    ):
        self.model = model
        self.calibration_data = calibration_data
        self.target_logits = target_logits
        self.num_layers = len(model.model.layers)
        self.num_prune = num_prune
        self.drop_entire_block = drop_entire_block
        self.history = history if history is not None else []
        self.enforce_constraints = enforce_constraints
        self.tokens_per_evaluation = tokens_per_evaluation
        self.first_objective = first_objective.lower()
        self.second_objective = second_objective.lower()
        self.baseline_latency = baseline_latency
        self.latency_sample = latency_sample
        self.ppl_input_ids = ppl_input_ids
        self.ppl_seq_len = ppl_seq_len
        self.debug_random_fitness = debug_random_fitness
        self.use_surrogate = use_surrogate
        self.pop_size = max(1, int(pop_size))
        self.surrogate_true_fraction = max(
            0.0, min(1.0, float(surrogate_true_fraction))
        )
        self.surrogate_true_per_gen = max(
            1, int(round(self.pop_size * self.surrogate_true_fraction))
        )
        self.stage1_importance_penalty_weight = max(
            0.0, float(stage1_importance_penalty_weight)
        )
        self._surrogate_obj1 = None
        self._surrogate_obj2 = None
        self._surrogate_X: List[np.ndarray] = []
        self._surrogate_y1: List[float] = []
        self._surrogate_y2: List[float] = []
        self._eval_count = 0
        self.attn_block_importance = self._resolve_stage1_importance_scores(
            stage1_importance_scores,
            key="attn",
        )
        self.mlp_block_importance = self._resolve_stage1_importance_scores(
            stage1_importance_scores,
            key="mlp",
        )

        num_variables = self.num_layers if drop_entire_block else 2 * self.num_layers
        num_constraints = 0
        if enforce_constraints:
            num_constraints = 2

        super().__init__(
            n_var=num_variables,
            n_obj=2,
            n_ieq_constr=num_constraints,
            xl=0,
            xu=1,
        )

    def _resolve_stage1_importance_scores(
        self,
        stage1_importance_scores: Optional[Dict[str, object]],
        *,
        key: str,
    ) -> np.ndarray:
        if not stage1_importance_scores or self.stage1_importance_penalty_weight <= 0.0:
            return np.zeros(self.num_layers, dtype=np.float32)
        raw_scores = stage1_importance_scores.get(key)
        if raw_scores is None:
            logging.warning(
                "Stage-1 importance scores missing key '%s'; disabling that penalty term.",
                key,
            )
            return np.zeros(self.num_layers, dtype=np.float32)
        values = np.asarray(list(raw_scores), dtype=np.float32).reshape(-1)
        if values.size != self.num_layers:
            logging.warning(
                "Stage-1 importance scores for '%s' have length %d, expected %d; disabling that penalty term.",
                key,
                values.size,
                self.num_layers,
            )
            return np.zeros(self.num_layers, dtype=np.float32)
        values = np.where(np.isfinite(values), values, 1.0).astype(np.float32)
        return np.clip(values, 0.0, 1.0)

    def _compute_stage1_importance_penalty(
        self,
        attn_mask: List[int],
        mlp_mask: List[int],
    ) -> float:
        if self.stage1_importance_penalty_weight <= 0.0:
            return 0.0

        contributions: List[float] = []
        if len(attn_mask) == self.num_layers:
            contributions.extend(
                float(self.attn_block_importance[idx])
                for idx, is_pruned in enumerate(attn_mask)
                if int(is_pruned) != 0
            )
        if len(mlp_mask) == self.num_layers:
            contributions.extend(
                float(self.mlp_block_importance[idx])
                for idx, is_pruned in enumerate(mlp_mask)
                if int(is_pruned) != 0
            )
        if not contributions:
            return 0.0
        return float(sum(contributions) / len(contributions))

    def _vector_to_masks(self, x: np.ndarray) -> Tuple[List[int], List[int]]:
        """Convert an optimisation vector into attention and MLP masks."""
        binary_vec = (x >= 0.5).astype(int)
        attn_mask = binary_vec[: self.num_layers].tolist()

        if self.drop_entire_block:
            mlp_mask = copy.deepcopy(attn_mask)
        else:
            mlp_mask = binary_vec[self.num_layers :].tolist()

        return attn_mask, mlp_mask

    def _evaluate(self, x, out, *args, **kwargs):
        attn_mask, mlp_mask = self._vector_to_masks(x)
        latency_val: Optional[float] = None
        importance_penalty = 0.0

        maskModel(self.model, attnMask=attn_mask, mlpMask=mlp_mask)
        try:
            with torch.no_grad():
                idx_in_gen = self._eval_count % self.pop_size
                gen_idx = self._eval_count // self.pop_size
                use_true = (
                    (not self.use_surrogate)
                    or (gen_idx == 0)
                    or (idx_in_gen < self.surrogate_true_per_gen)
                )
                if use_true:
                    if self.first_objective == "kl":
                        minibatch_data, minibatch_logits = _build_calibration_minibatch(
                            self.calibration_data,
                            self.target_logits,
                            self.tokens_per_evaluation,
                        )
                        if self.debug_random_fitness:
                            # Debug path: avoid KL divergence computation and return a random score.
                            objective_one = float(np.random.random())
                        else:
                            objective_one = compute_fitness(
                                self.model,
                                minibatch_data,
                                minibatch_logits,
                            )
                    elif self.first_objective == "perplexity":
                        if self.ppl_input_ids is None:
                            raise RuntimeError(
                                "Perplexity objective requires ppl_input_ids."
                            )
                        if self.debug_random_fitness:
                            objective_one = float(np.random.random())
                        else:
                            objective_one = evaluate_perplexity(
                                self.model,
                                self.ppl_input_ids,
                                seq_len=self.ppl_seq_len,
                                enable_tqdm=False,
                                device=str(next(self.model.parameters()).device),
                            )
                    else:
                        raise RuntimeError(
                            f"Unsupported first_objective '{self.first_objective}'."
                        )
                else:
                    if self._surrogate_obj1 is None:
                        raise RuntimeError(
                            "Surrogate model for objective 1 is not ready."
                        )
                    objective_one = float(_predict_surrogate(self._surrogate_obj1, x))

                total_params, active_params = count_params_with_masks(
                    self.model, attn_mask, mlp_mask
                )

            latency_delta = None
            if use_true:
                objective_two = float(active_params)
                if self.second_objective == "latency":
                    if self.latency_sample is None:
                        raise RuntimeError(
                            "Latency objective requires a latency sample tensor."
                        )
                    latency_val = evaluate_inference_time(
                        self.model, self.latency_sample
                    )
                    if self.baseline_latency is not None:
                        latency_delta = self.baseline_latency - latency_val
                    objective_two = float(latency_val)
            else:
                if self._surrogate_obj2 is None:
                    raise RuntimeError("Surrogate model for objective 2 is not ready.")
                objective_two = float(_predict_surrogate(self._surrogate_obj2, x))
        finally:
            unmaskModel(self.model, attnMask=attn_mask, mlpMask=mlp_mask)

        if self.stage1_importance_penalty_weight > 0.0:
            importance_penalty = self._compute_stage1_importance_penalty(
                attn_mask, mlp_mask
            )
            objective_one = float(objective_one) + (
                self.stage1_importance_penalty_weight * importance_penalty
            )

        if self.second_objective == "latency":
            latency_str = (
                f"latency={float(objective_two):.6f}s"
                if not use_true
                else f"latency={latency_val:.6f}s"
            )
        else:
            latency_str = f"active_params={int(active_params)}"
        suffix_parts = []
        if latency_delta is not None:
            suffix_parts.append(f"delta={latency_delta:.6f}s")
        if importance_penalty > 0.0:
            suffix_parts.append(f"importance_penalty={importance_penalty:.6f}")
        print(
            "[Proposed] {}={:.6f}, {}{}".format(
                "KL" if self.first_objective == "kl" else "Perplexity",
                float(objective_one),
                latency_str,
                f" ({', '.join(suffix_parts)})" if suffix_parts else "",
            )
        )
        out["F"] = np.array([objective_one, objective_two], dtype=np.float64)

        history_entry = {
            "attn": attn_mask,
            "mlp": mlp_mask,
            "fitness": float(objective_one),
            "active_params": float(active_params),
            "first_objective": self.first_objective,
        }
        if self.stage1_importance_penalty_weight > 0.0:
            history_entry["stage1_importance_penalty"] = float(importance_penalty)
            history_entry["stage1_importance_penalty_weight"] = float(
                self.stage1_importance_penalty_weight
            )
        if self.second_objective == "latency":
            if not use_true:
                history_entry["latency"] = float(objective_two)
            else:
                history_entry["latency"] = (
                    float(latency_val) if latency_val is not None else None
                )
            if latency_delta is not None:
                history_entry["latency_delta"] = float(latency_delta)
        self.history.append(history_entry)

        if use_true and self.use_surrogate:
            obj1 = float(objective_one)
            obj2 = float(objective_two)
            if np.isfinite(obj1) and np.isfinite(obj2):
                self._surrogate_X.append(np.asarray(x, dtype=np.float32).copy())
                self._surrogate_y1.append(obj1)
                self._surrogate_y2.append(obj2)
            else:
                logging.warning(
                    "Skipping surrogate sample with non-finite objectives: obj1=%s obj2=%s",
                    obj1,
                    obj2,
                )

        self._eval_count += 1
        if self.use_surrogate and (self._eval_count % self.pop_size == 0):
            if self._surrogate_X:
                X_train = np.stack(self._surrogate_X, axis=0)
                y1 = np.asarray(self._surrogate_y1, dtype=np.float32)
                y2 = np.asarray(self._surrogate_y2, dtype=np.float32)
                finite_mask = (
                    np.isfinite(y1) & np.isfinite(y2) & np.isfinite(X_train).all(axis=1)
                )
                if not finite_mask.any():
                    logging.warning(
                        "No finite surrogate samples available; skipping surrogate fit."
                    )
                else:
                    X_train = X_train[finite_mask]
                    y1 = y1[finite_mask]
                    y2 = y2[finite_mask]
                    self._surrogate_obj1 = _fit_xgboost_regressor(X_train, y1)
                    self._surrogate_obj2 = _fit_xgboost_regressor(X_train, y2)

        if self.enforce_constraints and self.n_ieq_constr > 0:
            if self.drop_entire_block:
                pruned = sum(attn_mask)
                constraints = [
                    pruned - self.num_prune,
                    self.num_prune - pruned,
                ]
            else:
                total_pruned = sum(attn_mask) + sum(mlp_mask)
                constraints = [
                    total_pruned - self.num_prune,
                    self.num_prune - total_pruned,
                ]
            out["G"] = np.array(constraints, dtype=np.float64)

    def decode(self, vector: np.ndarray) -> Tuple[List[int], List[int]]:
        """Helper to expose mask decoding outside pymoo."""
        return self._vector_to_masks(vector)


@torch.no_grad()
def proposed_multiobjective(
    model,
    num_prune: int,
    tokenizer,
    dataset_c4,
    *,
    drop_entire_block: bool = False,
    calibration_tokens: int = 262144,
    calibration_sequence_length: Optional[int] = 2048,
    tokens_per_evaluation: Optional[int] = 32768,
    pop_size: int = 5,
    n_gen: int = 2,
    verbose: bool = True,
    return_pareto: bool = False,
    return_history: bool = False,
    save_pareto_path: Optional[str] = None,
    pareto_plot_path: Optional[str] = None,
    pareto_selection: str = "best",
    enforce_block_budget: bool = True,
    first_objective: str = "kl",
    second_objective: str = "params",
    latency_batch_size: int = 1,
    latency_max_seq_len: int = 64,
    latency_num_runs: int = 1,
    latency_num_warmup_runs: int = 0,
    latency_input_prompt: str = "The Leaning Tower of Pisa is known for",
    latency_top_p: float = 0.95,
    latency_top_k: float = 50,
    latency_temperature: float = 1.0,
    latency_seed: int = 1234,
    ppl_input_ids: Optional[torch.Tensor] = None,
    ppl_seq_len: int = 2048,
    debug_random_targets: bool = False,
    debug_random_fitness: bool = False,
    stage1_use_surrogate: bool = False,
    stage1_surrogate_true_fraction: float = 0.2,
    stage1_importance_scores: Optional[Dict[str, object]] = None,
    stage1_importance_penalty_weight: float = 0.0,
):
    """
    Run a multi-objective search over attention/MLP masks using NSGA-II.

    Each candidate evaluation samples approximately `tokens_per_evaluation` tokens
    from the calibration set (similar to Proposed minibatch evaluation). Set this
    parameter to ``None`` to evaluate on the full calibration corpus.

    Returns the Pareto-optimal set of masks along with their objectives.

    When ``enforce_block_budget`` is ``True`` the combined number of pruned
    attention and MLP submodules across all layers is kept equal to
    ``num_prune`` (or exactly ``num_prune`` blocks when pruning entire blocks).

    Args:
      pareto_selection: Controls which Pareto solutions are returned. Use
        ``"best"`` (default) to return the single best solution, or ``"all"`` to
        return the full Pareto front.
      first_objective: Whether to optimise for ``"kl"`` (calibration KL divergence)
        or ``"perplexity"`` (Wikitext2 perplexity). Lower is better for all
        objectives.
      second_objective: Whether to optimise for ``"params"`` (active parameters) or
        ``"latency"`` (average forward-pass latency in seconds). Lower is better for
        all objectives.
      stage1_importance_scores: Optional precomputed normalized per-layer importance
        scores for attention and MLP stage-1 blocks on the mixed prompt calibration
        set. When provided together with a positive penalty weight, candidates that
        prune higher-importance blocks receive an additive penalty on objective 1.
    """
    first_objective = first_objective.lower()
    if first_objective not in {"kl", "perplexity"}:
        raise ValueError("first_objective must be either 'kl' or 'perplexity'")
    second_objective = second_objective.lower()
    if second_objective not in {"params", "latency"}:
        raise ValueError("second_objective must be either 'params' or 'latency'")
    default_sequence_length = 8192
    max_positions = getattr(
        model.config, "max_position_embeddings", default_sequence_length
    )

    if calibration_sequence_length is None:
        calibration_sequence_length = default_sequence_length

    calibration_sequence_length = min(calibration_sequence_length, max_positions)

    if model.config.model_type in ("llama", "phi3", "phi4"):
        calibration_sequence_length = min(calibration_sequence_length, 4096)
    elif model.config.model_type == "qwen2":
        calibration_sequence_length = min(calibration_sequence_length, 2048)

    calibration_data: List[torch.Tensor] = []
    if first_objective == "kl":
        samples_required = max(
            1,
            (calibration_tokens + calibration_sequence_length - 1)
            // calibration_sequence_length,
        )
        calibration_data = collect_samples_with_join(
            dataset_c4,
            tokenizer,
            samples_required,
            calibration_sequence_length,
        )
        calibration_data = [sample.to(dtype=torch.long) for sample in calibration_data]

    device = next(model.parameters()).device
    if second_objective == "latency" and "cuda" not in str(device):
        raise ValueError("Latency objective requires a CUDA device.")

    baseline_latency = None
    latency_sample: Optional[torch.Tensor] = None
    if second_objective == "latency":
        original_pad_token = tokenizer.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        try:
            latency_inputs = tokenizer(
                [latency_input_prompt] * latency_batch_size,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=latency_max_seq_len,
            )
        finally:
            tokenizer.pad_token_id = original_pad_token
        latency_sample = latency_inputs["input_ids"].to(device)
        baseline_latency = evaluate_inference_time(model, latency_sample)
        print(f"[Proposed] Baseline latency (unpruned) = {baseline_latency:.6f} sec")
    if first_objective == "perplexity" and ppl_input_ids is None:
        raise ValueError("Perplexity objective requires ppl_input_ids tensor.")

    target_logits: List[torch.Tensor] = []
    if first_objective == "kl":
        target_logit_dtype = torch.float16
        vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
        if vocab_size > 0 and calibration_data:
            bytes_per_value = torch.finfo(target_logit_dtype).bits // 8
            est_storage_gib = (
                len(calibration_data)
                * calibration_sequence_length
                * vocab_size
                * bytes_per_value
            ) / float(1024**3)
            logging.info(
                "[Proposed] Stage-1 calibration uses %d samples x %d tokens; "
                "estimated target-logit storage %.2f GiB (%s).",
                len(calibration_data),
                calibration_sequence_length,
                est_storage_gib,
                str(target_logit_dtype).replace("torch.", ""),
            )
            if est_storage_gib > 4.0:
                logging.warning(
                    "[Proposed] Estimated target-logit storage is high. Reduce "
                    "stage1_calibration_tokens or stage1_calibration_sequence_length if this run crashes."
                )
        if debug_random_targets:
            vocab_size = 5
            seq_len = 2
            print(
                f"[Proposed][Debug] Using random target logits (vocab_size={vocab_size}) "
                "to bypass real forward passes."
            )
            for sample in calibration_data:
                seq_len = sample.shape[1]
                # Random logits tensor that matches expected shape for KL divergence.
                target_logits.append(
                    torch.randn(
                        (sample.shape[0], seq_len, vocab_size),
                        dtype=torch.bfloat16,
                    )
                )
        else:
            for i in tqdm(
                range(len(calibration_data)), desc="Computing target logits (calib)"
            ):
                input_ids = calibration_data[i].to(device, dtype=torch.long)
                logits = model(input_ids, use_cache=False).logits
                target_logits.append(logits.to(device="cpu", dtype=target_logit_dtype))
                del logits, input_ids
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    evaluation_history: List[Dict[str, object]] = []

    problem = ProposedMaskProblem(
        model=model,
        calibration_data=calibration_data,
        target_logits=target_logits,
        num_prune=num_prune,
        drop_entire_block=drop_entire_block,
        history=evaluation_history,
        enforce_constraints=enforce_block_budget,
        tokens_per_evaluation=tokens_per_evaluation,
        first_objective=first_objective,
        second_objective=second_objective,
        baseline_latency=baseline_latency,
        latency_sample=latency_sample,
        ppl_input_ids=ppl_input_ids,
        ppl_seq_len=ppl_seq_len,
        debug_random_fitness=debug_random_fitness,
        use_surrogate=stage1_use_surrogate,
        pop_size=pop_size,
        surrogate_true_fraction=stage1_surrogate_true_fraction,
        stage1_importance_scores=stage1_importance_scores,
        stage1_importance_penalty_weight=stage1_importance_penalty_weight,
    )

    sampling = BinaryRandomSampling()
    crossover = SBX(prob=1.0, eta=30)
    mutation = ProposedMaskMutation(
        num_layers=problem.num_layers,
        num_prune=num_prune,
        drop_entire_block=drop_entire_block,
        max_mutations=3,
        enforce_budget=enforce_block_budget,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    result = minimize(
        problem,
        algorithm,
        termination,
        verbose=verbose,
        save_history=False,
    )

    vectors = result.X
    objectives = result.F

    if vectors is None or objectives is None:
        # If NSGA-II fails to find feasible individuals fall back to the
        # last population so we can at least return the best candidate.
        population = algorithm.pop
        if population is not None and len(population) > 0:
            vectors = population.get("X")
            objectives = population.get("F")

    if vectors is None or objectives is None:
        raise RuntimeError(
            "Proposed multi-objective search produced no candidate solutions. "
            "Consider relaxing constraints or increasing the population size."
        )

    vectors = np.atleast_2d(vectors)
    objectives = np.atleast_2d(objectives)

    pareto_solutions: List[Dict[str, object]] = []
    for vector, objective_vals in zip(vectors, objectives):
        attn_mask, mlp_mask = problem.decode(vector)
        _, active_params = count_params_with_masks(model, attn_mask, mlp_mask)
        importance_penalty = problem._compute_stage1_importance_penalty(
            attn_mask, mlp_mask
        )
        record = {
            "attn": attn_mask,
            "mlp": mlp_mask,
            "fitness": float(objective_vals[0]),
            "active_params": float(active_params),
        }
        if problem.stage1_importance_penalty_weight > 0.0:
            record["stage1_importance_penalty"] = float(importance_penalty)
            record["stage1_importance_penalty_weight"] = float(
                problem.stage1_importance_penalty_weight
            )
        if second_objective == "latency":
            record["latency"] = float(objective_vals[1])
        pareto_solutions.append(record)

    if not pareto_solutions:
        raise RuntimeError(
            "Proposed multi-objective search returned an empty Pareto front."
        )

    # pareto_solutions.sort(key=lambda s: (s["fitness"], s["active_params"]))
    pareto_solutions.sort(key=lambda s: s["fitness"])

    selection_mode = pareto_selection.lower()
    if selection_mode not in {"best", "all"}:
        raise ValueError("pareto_selection must be either 'best' or 'all'")

    if save_pareto_path or pareto_plot_path:
        _export_pareto_front(
            pareto_solutions,
            csv_path=save_pareto_path,
            plot_path=pareto_plot_path,
            all_solutions=evaluation_history,
            second_objective=second_objective,
        )

    if return_pareto:
        return_payload: Dict[str, object] = {
            "solutions": pareto_solutions,
            "F": objectives,
            "X": vectors,
            "algorithm": "NSGA2",
        }
        if selection_mode == "best":
            return_payload["selected"] = pareto_solutions[0]
        else:
            return_payload["selected"] = pareto_solutions
        if return_history:
            return_payload["evaluations"] = evaluation_history
        return return_payload

    if selection_mode == "all":
        if return_history:
            return pareto_solutions, evaluation_history
        return pareto_solutions

    best_solution = pareto_solutions[0]
    if return_history:
        return best_solution["attn"], best_solution["mlp"], evaluation_history
    return best_solution["attn"], best_solution["mlp"]


def _export_pareto_front(
    pareto_solutions: List[Dict[str, object]],
    *,
    csv_path: Optional[str],
    plot_path: Optional[str],
    all_solutions: Optional[List[Dict[str, object]]] = None,
    second_objective: str = "params",
) -> None:
    """
    Persist Pareto solutions to disk and optionally create a scatter plot with
    all explored candidates highlighted separately from the Pareto front.
    """
    if not pareto_solutions:
        print("[Proposed] No Pareto solutions available to export.")
        return

    def _prepare_output_path(path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    csv_output_path = _prepare_output_path(csv_path)
    plot_output_path = _prepare_output_path(plot_path)

    if csv_output_path:
        fields = ["attn", "mlp", "fitness", "active_params"]
        if second_objective == "latency":
            fields.append("latency")
        if any(sol.get("first_objective") is not None for sol in pareto_solutions):
            fields.append("first_objective")
        if any(
            sol.get("stage1_importance_penalty") is not None for sol in pareto_solutions
        ):
            fields.append("stage1_importance_penalty")
        if any(
            sol.get("stage1_importance_penalty_weight") is not None
            for sol in pareto_solutions
        ):
            fields.append("stage1_importance_penalty_weight")
        records = []
        for sol in pareto_solutions:
            record = {key: sol.get(key) for key in fields}
            records.append(record)
        try:
            import pandas as pd
        except ImportError:
            import csv

            with csv_output_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fields,
                )
                writer.writeheader()
                writer.writerows(records)
            print(
                f"[Proposed] Saved Pareto front to {csv_output_path.resolve()} (via csv module)."
            )
        else:
            df = pd.DataFrame.from_records(records)
            df.to_csv(csv_output_path, index=False)
            print(f"[Proposed] Saved Pareto front to {csv_output_path.resolve()}.")

    # Save all evaluated solutions (used for plotting) to a companion CSV.
    base_for_history = csv_output_path if csv_output_path else plot_output_path
    if all_solutions and base_for_history:
        history_fields = ["attn", "mlp", "fitness", "active_params"]
        if second_objective == "latency":
            history_fields.append("latency")
        if any(sol.get("first_objective") is not None for sol in all_solutions):
            history_fields.append("first_objective")
        if any(
            sol.get("stage1_importance_penalty") is not None for sol in all_solutions
        ):
            history_fields.append("stage1_importance_penalty")
        if any(
            sol.get("stage1_importance_penalty_weight") is not None
            for sol in all_solutions
        ):
            history_fields.append("stage1_importance_penalty_weight")
        history_suffix = base_for_history.suffix.lower()
        history_suffix = history_suffix if history_suffix == ".csv" else ".csv"
        history_path = base_for_history.with_name(
            f"{base_for_history.stem}_all{history_suffix}"
        )
        history_records = [
            {key: sol.get(key) for key in history_fields} for sol in all_solutions
        ]
        try:
            import pandas as pd
        except ImportError:
            import csv

            with history_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=history_fields)
                writer.writeheader()
                writer.writerows(history_records)
            print(
                f"[Proposed] Saved all evaluated solutions to {history_path.resolve()} (via csv module)."
            )
        else:
            df_history = pd.DataFrame.from_records(history_records)
            df_history.to_csv(history_path, index=False)
            print(
                f"[Proposed] Saved all evaluated solutions to {history_path.resolve()}."
            )

    if plot_output_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Proposed] matplotlib not available; skipping Pareto plot export.")
            return

        x_key = "active_params" if second_objective == "params" else "latency"
        objective_labels = {
            str(sol.get("first_objective")).lower()
            for sol in pareto_solutions + (all_solutions or [])
            if sol.get("first_objective") is not None
        }
        if objective_labels == {"perplexity"}:
            y_label = "PPL"
        elif objective_labels == {"kl"}:
            y_label = "KL Divergence"
        else:
            y_label = "Fitness (lower is better)"
        pareto_scores = np.array(
            [
                (sol.get(x_key), sol.get("fitness"))
                for sol in pareto_solutions
                if sol.get(x_key) is not None and sol.get("fitness") is not None
            ],
            dtype=np.float64,
        )
        history_scores: Optional[np.ndarray] = None
        if all_solutions:
            history_scores = np.array(
                [
                    (sol.get(x_key), sol.get("fitness"))
                    for sol in all_solutions
                    if sol.get(x_key) is not None and sol.get("fitness") is not None
                ],
                dtype=np.float64,
            )

        plt.figure()
        if history_scores is not None and len(history_scores) > 0:
            plt.scatter(
                history_scores[:, 0],
                history_scores[:, 1],
                marker="x",
                color="dimgray",
                alpha=0.7,
                label="All Solutions",
            )
        if len(pareto_scores) > 0:
            pareto_sorted = pareto_scores[np.argsort(pareto_scores[:, 0])]
            plt.scatter(
                pareto_sorted[:, 0],
                pareto_sorted[:, 1],
                marker="x",
                color="red",
                label="Pareto Front",
            )
            if len(pareto_sorted) > 1:
                plt.plot(
                    pareto_sorted[:, 0],
                    pareto_sorted[:, 1],
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                )
        plt.xlabel("Active Parameters" if x_key == "active_params" else "Latency (s)")
        plt.ylabel(y_label)
        plt.title(
            "Proposed Pareto Front"
            if second_objective != "latency"
            else f"Proposed Pareto Front ({y_label} vs Latency)"
        )
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_output_path)
        plt.close()
        print(f"[Proposed] Saved Pareto plot to {plot_output_path.resolve()}.")
