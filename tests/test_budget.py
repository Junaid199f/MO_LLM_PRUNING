"""Numerical invariants of the actual layer-wise allocation functions."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_pruning.budget import _allocate_prune_budget, _build_monotone_prune_ratios


class BudgetTests(unittest.TestCase):
    def test_budget_is_conserved_with_uneven_weights_and_caps(self):
        rng = random.Random(13)
        for count in [1, 2, 8, 32]:
            for _ in range(20):
                caps = [rng.random() for _ in range(count)]
                target = rng.random() * sum(caps)
                result = _allocate_prune_budget(
                    [rng.random() for _ in caps], target, max_prune_by_layer=caps
                )
                self.assertAlmostEqual(sum(result), target, places=9)
                for ratio, cap in zip(result, caps):
                    self.assertGreaterEqual(ratio, 0)
                    self.assertLessEqual(ratio, cap)

    def test_infeasible_budget_is_limited_to_capacity(self):
        result = _allocate_prune_budget(
            [1, 5, 2], 10, max_prune_by_layer=[0.1, 0.3, 0.2]
        )
        self.assertAlmostEqual(sum(result), 0.6)

    def test_zero_weights_still_allocate_the_full_budget(self):
        result = _allocate_prune_budget(
            [0, 0, 0], 0.9, max_prune_by_layer=[0.1, 0.8, 0.8]
        )
        self.assertAlmostEqual(sum(result), 0.9)
        self.assertAlmostEqual(result[0], 0.1)

    def test_empty_zero_and_negative_budgets(self):
        self.assertEqual(_allocate_prune_budget([], 1, max_prune_by_layer=[]), [])
        self.assertEqual(
            _allocate_prune_budget([1, 2], -1, max_prune_by_layer=[1, 1]), [0, 0]
        )

    def test_caps_must_match_the_layer_count(self):
        with self.assertRaises(ValueError):
            _allocate_prune_budget([1, 2], 0.5, max_prune_by_layer=[1])

    def test_monotone_profile_preserves_budget_and_early_layers(self):
        for target in [0.1, 0.375, 0.5]:
            ratios = _build_monotone_prune_ratios(32, target, 5, 0.5)
            self.assertAlmostEqual(sum(ratios) / len(ratios), target, places=9)
            self.assertLessEqual(ratios[0], target * 0.75)
            self.assertLessEqual(ratios[1], target * 0.75)
            self.assertTrue(all(a <= b for a, b in zip(ratios, ratios[1:])))
            self.assertTrue(all(0 <= ratio <= 0.99 for ratio in ratios))
