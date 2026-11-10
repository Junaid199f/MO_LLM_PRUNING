"""Explicit search budgets, separate from model and hardware settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchBudget:
    population: int
    generations: int
    samples: int
    sequence_length: int
    calibration_tokens: int
    tokens_per_evaluation: int
    bo_trials: int
    bo_warmup: int
    importance_prompts: int


PROFILES = {
    "quick": SearchBudget(
        population=4,
        generations=2,
        samples=32,
        sequence_length=512,
        calibration_tokens=8192,
        tokens_per_evaluation=4096,
        bo_trials=4,
        bo_warmup=2,
        importance_prompts=16,
    ),
    "paper": SearchBudget(
        population=40,
        generations=40,
        samples=1024,
        sequence_length=2048,
        calibration_tokens=1024 * 2048,
        tokens_per_evaluation=32768,
        bo_trials=40,
        bo_warmup=20,
        importance_prompts=1024,
    ),
}
