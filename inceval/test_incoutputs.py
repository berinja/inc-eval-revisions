#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for incoutputs.py.
"""

import unittest

import numpy as np

from inceval.incoutputs import IncOutputs
from inceval.aux import GOLD

EMPTY = np.inf


class TestIncOutputs(unittest.TestCase):
    """Tests for methods in IncOutputs."""
    def setUp(self):
        recomputations = np.array([False, False, True, True, True,
                                   False, True, True, True, True])
        outputs = IncOutputs(10, recomputations=recomputations)
        outputs.add_prefix(0, [1])
        outputs.add_prefix(1, [1, 2])
        outputs.add_prefix(2, [1, 1, 3])
        outputs.add_prefix(3, [1, 1, 3, 2])
        outputs.add_prefix(4, [1, 2, 1, 1, 3])
        outputs.add_prefix(5, [1, 2, 1, 1, 3, 2])
        outputs.add_prefix(6, [1, 2, 2, 1, 3, 1, 1])
        outputs.add_prefix(7, [1, 2, 1, 1, 3, 1, 1, 3])
        outputs.add_prefix(8, [1, 2, 1, 1, 3, 1, 1, 2, 1])
        outputs.add_prefix(9, [1, 2, 1, 1, 3, 2, 2, 3, 1, 1])
        self.outputs_1 = outputs

        outputs = IncOutputs(5, [1, 2, 1, 2, 3], eval_mode=GOLD)
        outputs.add_all_prefixes(np.array([
            [1, np.inf, np.inf, np.inf, np.inf],
            [3, 2,      np.inf, np.inf, np.inf],
            [1, 2,      1,      np.inf, np.inf],
            [2, 2,      3,      1,      np.inf],
            [2, 2,      3,      1,      3],
        ]))
        self.outputs_2 = outputs

    def test_n_tokens(self):
        self.assertEqual(self.outputs_1.n_tokens, 10)
        self.assertEqual(self.outputs_2.n_tokens, 5)

    def test_revision_timesteps(self):
        self.assertEqual(self.outputs_1.revision_timesteps, [2, 4, 6, 7, 8, 9])
        self.assertEqual(self.outputs_2.revision_timesteps, [1, 2, 3])

    def test_write_timesteps(self):
        self.assertEqual(self.outputs_1.write_timesteps, [0, 1, 3, 5])
        self.assertEqual(self.outputs_2.write_timesteps, [0, 4])

    def test_correct_prefixes(self):
        self.assertEqual(self.outputs_1.correct_prefixes, [0, 1, 4, 5, 9])
        self.assertEqual(self.outputs_2.correct_prefixes, [0, 2])

    def test_incorrect_prefixes(self):
        self.assertEqual(self.outputs_1.incorrect_prefixes, [2, 3, 6, 7, 8])
        self.assertEqual(self.outputs_2.incorrect_prefixes, [1, 3, 4])

    def test_n_correct_prefixes(self):
        self.assertEqual(self.outputs_1.n_correct_prefixes, 5)
        self.assertEqual(self.outputs_2.n_correct_prefixes, 2)

    def test_n_incorrect_prefixes(self):
        self.assertEqual(self.outputs_1.n_incorrect_prefixes, 5)
        self.assertEqual(self.outputs_2.n_incorrect_prefixes, 3)

    def test_n_revisions(self):
        self.assertEqual(self.outputs_1.n_revisions, 6)
        self.assertEqual(self.outputs_2.n_revisions, 3)

    def test_n_recomputations(self):
        self.assertEqual(self.outputs_1.n_recomputations, 7)
        self.assertEqual(self.outputs_2.n_recomputations, None)

    def test_recomputation_timesteps(self):
        steps = [2, 3, 4, 6, 7, 8, 9]
        self.assertEqual(self.outputs_1.recomputation_timesteps, steps)
        self.assertEqual(self.outputs_2.recomputation_timesteps, None)

    def test_n_active_recomputations(self):
        self.assertEqual(self.outputs_1.n_active_recomputations, 6)
        self.assertEqual(self.outputs_2.n_active_recomputations, None)

    def test_n_writes(self):
        self.assertEqual(self.outputs_1.n_writes, 4)
        self.assertEqual(self.outputs_2.n_writes, 2)

    def test_n_revision_and_correct_prefix(self):
        self.assertEqual(self.outputs_1.n_revision_and_correct_prefix, 2)
        self.assertEqual(self.outputs_2.n_revision_and_correct_prefix, 2)

    def test_n_revision_and_incorrect_prefix(self):
        self.assertEqual(self.outputs_1.n_revision_and_incorrect_prefix, 4)
        self.assertEqual(self.outputs_2.n_revision_and_incorrect_prefix, 1)

    def test_n_write_and_correct_prefix(self):
        self.assertEqual(self.outputs_1.n_write_and_correct_prefix, 3)
        self.assertEqual(self.outputs_2.n_write_and_correct_prefix, 1)

    def test_n_write_and_incorrect_prefix(self):
        self.assertEqual(self.outputs_1.n_write_and_incorrect_prefix, 1)
        self.assertEqual(self.outputs_2.n_write_and_incorrect_prefix, 1)

    def test_edit_overhead(self):
        self.assertEqual(self.outputs_1.edit_overhead, 11 / (11 + 10))
        self.assertEqual(self.outputs_2.edit_overhead, 4 / (4 + 5))

    def test_relative_correctness(self):
        self.assertEqual(self.outputs_1.relative_correctness, 5 / 10)
        self.assertEqual(self.outputs_2.relative_correctness, 2 / 5)

    def test_n_total_edits(self):
        self.assertEqual(self.outputs_1.n_total_edits, 11)
        self.assertEqual(self.outputs_2.n_total_edits, 4)

    def test_n_edits_per_token(self):
        steps = [0, 2, 3, 1, 0, 2, 1, 2, 0, 0]
        self.assertEqual(self.outputs_1.n_edits_per_token.tolist(), steps)
        steps = [3, 0, 1, 0, 0]
        self.assertEqual(self.outputs_2.n_edits_per_token.tolist(), steps)

    def test_n_edits_per_timestep(self):
        steps = [0, 0, 1, 0, 3, 0, 2, 1, 1, 3]
        self.assertEqual(self.outputs_1.n_edits_per_timestep.tolist(), steps)
        steps = [0, 1, 1, 2, 0]
        self.assertEqual(self.outputs_2.n_edits_per_timestep.tolist(), steps)


if __name__ == '__main__':
    unittest.main()
