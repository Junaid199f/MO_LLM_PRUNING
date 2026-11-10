"""BlockPruner comparison baseline."""

import logging
import torch
from .utilities import maskModel, unmaskModel
from .evaluation import evaluate_perplexity


@torch.no_grad()
def blockpruner(model, num_prune, calibration_sample):
    num_blocks = len(model.model.layers)
    attnMask = [0] * num_blocks
    mlpMask = [0] * num_blocks

    for _ in range(num_prune):

        # Find the best attention to prune
        best_to_prune = None
        best_ppl = float("inf")

        for to_prune in range(num_blocks):

            # Cannot prune a block twice
            if attnMask[to_prune] == 1:
                continue

            # mask the model
            attnMask[to_prune] = 1
            maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

            # Evaluate
            ppl = evaluate_perplexity(
                model, calibration_sample, seq_len=2048, enable_tqdm=False
            )

            logging.debug(f"[Attention] When pruning {to_prune} perplexity is {ppl}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_to_prune = to_prune

            # Unmask the model
            unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
            attnMask[to_prune] = 0

        logging.debug(f"[Attention] Best to prune: {best_to_prune} ({best_ppl})")
        attnMask[best_to_prune] = 1

        # Find the best MLP to prune
        best_to_prune = None
        best_ppl = float("inf")

        for to_prune in range(num_blocks):

            # Cannot prune a block twice
            if mlpMask[to_prune] == 1:
                continue

            # mask the model
            mlpMask[to_prune] = 1
            maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

            # Evaluate
            ppl = evaluate_perplexity(
                model, calibration_sample, seq_len=2048, enable_tqdm=False
            )

            logging.debug(f"[MLP] When pruning {to_prune} perplexity is {ppl}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_to_prune = to_prune

            # Unmask the model
            unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
            mlpMask[to_prune] = 0

        logging.debug(f"[MLP] Best to prune: {best_to_prune} ({best_ppl})")
        mlpMask[best_to_prune] = 1

    return attnMask, mlpMask
