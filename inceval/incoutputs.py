#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to represent the incremental outputs chart for one sequence and to
compute the evaluation metrics.
"""

from itertools import groupby
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score as accuracy

from inceval.aux import build_empty_chart, EMPTY, Criterion, SILVER, GOLD
from inceval.edit import EditQualityChart, EditQualities
from inceval.revision import RevisionSeq

Label = Union[str, int, float]
Prefix = List[Label]


class IncOutputs:
    """A chart for incremental sequence labelling outputs and their metrics.

    The main chart is a lower triangular matrix in which each row represent
    one timestep. Cell (i, j) is the output label for token j at time i. The
    upper part is filled with the EMPTY constant as fillers.

    The edit chart contains 1 when a substitution/addition occured for a given
    label, else 0. It is a lower triangular matrix in which each row represent
    one timestep. The main diagonal is always filled with 1s, which are the
    additions. All other entries that are 1 are substitions. The upper part
    is filled with the EMPTY constant as fillers.
    """
    def __init__(self, n: int, gold: Optional[Prefix] = None, 
                 recomputations: Optional[np.array] = None,
                 eval_mode: Criterion = SILVER):
        self.eval_mode = eval_mode
        self.recomputations = recomputations
        self.chart = build_empty_chart(n)
        self.edits: Optional[np.array] = None
        self.edit_qualities: Optional[np.array] = None
        self.revision_qualities: Optional[np.array] = None
        self.silver: Optional[np.array] = None
        self.filled: bool = False
        self.standard: Optional[np.array] = None
        self.range_param: int = 2
        if gold is not None:
            assert len(gold) == n
            self.gold = self._build_inc_gold(gold)
            #self.gold = np.array(gold)
        if self.eval_mode is GOLD:
            assert gold is not None

    def add_all_prefixes(self, prefix_matrix: np.array) -> None:
        """Construct the chart from a complete output matrix."""
        assert prefix_matrix.shape[0] == self.n_tokens
        for t, row in enumerate(prefix_matrix):
            prefix = list(row[: t+1].astype(int))
            self.add_prefix(t, prefix)

    def add_prefix(self, time_step: int, prefix: Prefix) -> None:
        """For constructing chart row by row."""
        self._check_prefix_validity(time_step, prefix)
        self.chart[time_step][:time_step + 1] = prefix
        # upon addition of the last prefix, we create the edits and gold chart
        if time_step == self.n_tokens - 1:
            self._build_last_step()

    def _build_last_step(self) -> None:
        """Create edit charts, silver chart and set the evaluation standard."""
        self.edits = self._build_edits()
        self.silver = self._build_inc_silver()
        self._set_standard()
        self.filled = True
        self.edit_qualities = EditQualityChart(
            self.chart, self.edits, self.standard, self.range_param)
        self.revision_qualities = RevisionSeq(
            self.chart, self.edits, self.standard,
            self.revision_timesteps, self.range_param)

    def _check_prefix_validity(self, time_step: int, prefix: Prefix) -> None:
        """Ensure that prefixes are added in the right order."""
        if self.filled:
            raise ValueError('The chart has already been filled!')
        # time_step starts from 0 and is the position to be filled
        if self._n_filled != time_step:
            raise ValueError(f'Add time step {self._n_filled} first!')
        if len(prefix) != time_step + 1:
            raise ValueError('Prefix length does not match currect time step!')

    def _build_inc_gold(self, gold: Prefix) -> np.array:
        """Create incremental gold chart."""
        inc_gold = build_empty_chart(self.n_tokens)
        for t in range(self.n_tokens):
            inc_gold[t][: t+1] = gold[: t+1]
        return inc_gold

    def _build_inc_silver(self) -> np.array:
        """Create incremental silver chart, with final output as gold."""
        inc_silver = build_empty_chart(self.n_tokens)
        for t in range(self.n_tokens):
            inc_silver[t][: t+1] = self.final_output[: t+1]
        return inc_silver

    def _build_edits(self) -> np.array:
        """Extract edits from incremental outputs by comparison rowwise."""
        edit_chart = np.full(self.chart.shape, 0.)
        for t, prefix in enumerate(self.chart):
            if t == 0:
                # the first label is an addition (by definition, an edit)
                edit_chart[t][0] = 1.
            else:
                previous_prefix = self.chart[t-1][:t]
                edited_labels = (prefix[:t] != previous_prefix).astype(float)
                edit_chart[t][:t] = edited_labels
                # the last label is an addition (by definition, an edit)
                edit_chart[t][t] = 1.
        return edit_chart

    @property
    def n_tokens(self) -> int:
        """Length of the input and final output sequence."""
        return self.chart.shape[0]

    @property
    def _n_filled(self) -> int:
        """How many timesteps have already been filled with output prefixes."""
        return (self.chart[:, 0] != EMPTY).sum()

    @property
    def final_output(self) -> np.array:
        """The non-incremental output (i.e. final label sequence)."""
        return self.chart[-1]

    @property
    def final_accuracy(self) -> float:
        """Correctness of final output wrt gold."""
        return accuracy(self.standard, self.final_output)

    def get_accuracy_at_t(self, t: int) -> float:
        """Correctness of a prefix."""
        return accuracy(self.standard[:t+1], self.chart[t][:t+1])

    @property
    def accuracy_by_turn(self) -> np.array:
        """Correctness of a prefix."""
        accs = [self.get_accuracy_at_t(t) for t in range(self.n_tokens)]
        return np.array(accs)

    def _set_standard(self) -> None:
        """Sets the internal gold standard, using gold or silver labels."""
        if self.eval_mode == GOLD:
            self.standard = self.gold[-1]
        if self.eval_mode == SILVER:
            self.standard = self.final_output

    @property
    def revision_timesteps(self) -> List[int]:
        """Return a list of timesteps where revisions occurred."""
        return [t for t, row in enumerate(self.edits) if (row == 1).sum() > 1]

    @property
    def write_timesteps(self) -> List[int]:
        """Return a list of timesteps where revisions occurred."""
        return [t for t, row in enumerate(self.edits) if (row == 1).sum() == 1]

    @property
    def correct_prefixes(self) -> List[int]:
        """Which prefixes are correct with respect to the criterion."""
        return [t for t, row in enumerate(self.chart)
                if np.array_equal(row[:t+1], self.standard[:t+1])]

    @property
    def incorrect_prefixes(self) -> List[int]:
        """Which prefixes are correct with respect to the criterion."""
        return [t for t, row in enumerate(self.chart)
                if not np.array_equal(row[:t+1], self.standard[:t+1])]

    @property
    def n_correct_prefixes(self) -> int:
        """How many of the prefixes are correct."""
        return len(self.correct_prefixes)

    @property
    def perc_correct_prefixes(self) -> float:
        """% of the prefixes that are correct."""
        return 100 * self.n_correct_prefixes / self.n_tokens

    @property
    def n_incorrect_prefixes(self) -> int:
        """How many of the prefixes are incorrect."""
        return len(self.incorrect_prefixes)

    @property
    def perc_incorrect_prefixes(self) -> float:
        """% of the prefixes that are incorrect."""
        return 100 * self.n_incorrect_prefixes / self.n_tokens

    @property
    def n_revisions(self) -> int:
        """How many revisions occurred."""
        return len(self.revision_timesteps)

    @property
    def perc_revisions(self) -> float:
        """% of timesteps with revision."""
        return 100 * self.n_revisions / self.n_tokens

    @property
    def n_recomputations(self) -> int:
        """Total number of time steps where recomputations were performed."""
        if self.recomputations is not None:
            return self.recomputations.sum()

    @property
    def perc_recomputations(self) -> float:
        """% of timesteps with recomputations."""
        if self.recomputations is not None:
            return 100 * self.n_recomputations / self.n_tokens

    @property
    def recomputation_timesteps(self) -> List[int]:
        """Return a list of timesteps where recomputations occurred."""
        if self.recomputations is not None:
            return np.where(self.recomputations == True)[0].tolist()

    @property
    def n_active_recomputations(self) -> int:
        """Total number of recomputations that caused revisions."""
        if self.recomputations is not None:
            active = set(self.revision_timesteps) & set(self.recomputation_timesteps)
            return len(active)

    @property
    def perc_active_recomputations(self) -> float:
        """% of timesteps with recomputations that caused revisions."""
        if self.recomputations is not None:
            return 100 * self.n_active_recomputations / self.n_recomputations

    @property
    def n_inactive_recomputations(self) -> int:
        """Total number of recomputations that did not cause revisions."""
        return self.n_recomputations - self.n_active_recomputations

    @property
    def perc_inactive_recomputations(self) -> float:
        """% of timesteps with recomputations that caused revisions."""
        if self.recomputations is not None:
            return 100 * self.n_inactive_recomputations / self.n_recomputations

    @property
    def n_writes(self) -> int:
        """How many writes (not revisions) occurred."""
        return len(self.write_timesteps)

    @property
    def perc_writes(self) -> float:
        """% of timesteps without revision."""
        return 100 * self.n_writes / self.n_tokens

    @property
    def n_revision_and_correct_prefix(self) -> int:
        """How many revisions on correct prefixes."""
        # we need to compare whether a revision at time step t edited a prefix
        # that was correct at time step (t-1)
        shifted_steps = np.array(self.revision_timesteps) - 1
        intersect = set(shifted_steps) & set(self.correct_prefixes)
        return len(intersect)

    @property
    def n_revision_and_incorrect_prefix(self) -> int:
        """How many revisions on incorrect prefixes."""
        # we need to compare whether a revision at time step t edited a prefix
        # that was correct at time step (t-1)
        shifted_steps = np.array(self.revision_timesteps) - 1
        intersect = set(shifted_steps) & set(self.incorrect_prefixes)
        return len(intersect)

    @property
    def n_write_and_correct_prefix(self) -> int:
        """How many writes on correct prefixes."""
        # we need to compare whether a revision at time step t edited a prefix
        # that was correct at time step (t-1)
        shifted_steps = np.array(self.write_timesteps) - 1
        intersect = set(shifted_steps) & set(self.correct_prefixes)
        # FIXME: decide whether to add 1 here or not
        # by definition, we can set that the empty prefix is correct
        # and it will always be a write
        # either we do this, or we have to change the total number of writes
        # otherwise the total won't sum to the number of time steps
        return len(intersect) + 1

    @property
    def n_write_and_incorrect_prefix(self) -> int:
        """How many writes on incorrect prefixes."""
        # we need to compare whether a revision at time step t edited a prefix
        # that was correct at time step (t-1)
        shifted_steps = np.array(self.write_timesteps) - 1
        intersect = set(shifted_steps) & set(self.incorrect_prefixes)
        return len(intersect)

    @property
    def r_pertinence(self) -> float:
        """Revisions on incorrect prefixes divided by all revisions."""
        if self.n_revisions == 0:
            return 0
        return self.n_revision_and_incorrect_prefix / self.n_revisions

    @property
    def r_pertinence_complement(self) -> float:
        """Revisions on correct prefixes divided by all revisions."""
        if self.n_revisions == 0:
            return 0
        return self.n_revision_and_correct_prefix / self.n_revisions

    @property
    def a_pertinence(self) -> float:
        """Writes on correct prefixes divided by all writes."""
        if self.n_writes == 0:
            return 0
        return self.n_write_and_correct_prefix / self.n_writes

    @property
    def a_pertinence_complement(self) -> float:
        """Writes on incorrect prefixes divided by all writes."""
        if self.n_writes == 0:
            return 0
        return self.n_write_and_incorrect_prefix / self.n_writes

    @property
    def r_appropriateness(self) -> float:
        """Revisions on incorrect prefixes divided by all incorrect prefixes"""
        if self.n_incorrect_prefixes() == 0:
            return 0
        return self.n_revision_and_incorrect_prefix / self.n_incorrect_prefixes

    @property
    def r_appropriateness_complement(self) -> float:
        """Revisions on correct prefixes divided by all correct prefixes."""
        if self.n_correct_prefixes() == 0:
            return 0
        return self.n_revision_and_correct_prefix / self.n_correct_prefixes

    @property
    def a_appropriateness(self) -> float:
        """Writes on correct prefixes divided by all correct prefixes."""
        if self.n_correct_prefixes() == 0:
            return 0
        return self.n_write_and_correct_prefix / self.n_correct_prefixes

    @property
    def a_appropriateness_complement(self) -> float:
        """Writes on incorrect prefixes divided by all incorrect prefixes."""
        if self.n_incorrect_prefixes() == 0:
            return 0
        return self.n_write_and_incorrect_prefix / self.n_incorrect_prefixes()

    @property
    def edit_overhead(self):
        """Edit overhead metric on incremental chart."""
        necessary_edits = self.edits.diagonal().sum()
        unnecessary_edits = self.edits.sum() - necessary_edits
        return unnecessary_edits / (necessary_edits + unnecessary_edits)

    @property
    def relative_correctness(self) -> float:
        """Relative-correctness metric on incremental chart."""
        final = self.final_output
        n_correct = len([t for t, row in enumerate(self.chart)
                         if np.array_equal(row[:t+1], final[:t+1])])
        return n_correct / self.n_tokens

    @property
    def correction_time_score(self) -> float:
        """Correction time score on incremental chart."""
        pass

    @property
    def correction_time_per_token(self) -> np.array:
        """Correction time by label."""
        pass

    @property
    def n_total_edits(self) -> int:
        """Total number of edits."""
        return np.tril(self.edits, k=-1).sum()

    @property
    def perc_total_edits(self) -> float:
        """% of edits."""
        return 100 * self.n_total_edits / self.total_possible_edits

    @property
    def n_edits_per_token(self) -> np.array:
        """Total number of edits per token."""
        return np.tril(self.edits, k=-1).sum(axis=0)

    @property
    def perc_edits_per_token(self) -> np.array:
        """% of edits per token."""
        return 100 * self.n_edits_per_token / self.possible_edits_per_token

    @property
    def n_edits_per_timestep(self) -> np.array:
        """Number of edits per timestep."""
        return np.tril(self.edits, k=-1).sum(axis=1)

    @property
    def perc_edits_per_timestep(self) -> np.array:
        """% of edits per timestep."""
        return 100 * self.n_edits_per_timestep / self.possible_edits_per_timestep

    @property
    def n_edits_per_revision(self) -> np.array:
        """Number of edited labels for each time step where revision occurs."""
        return self.n_edits_per_timestep[self.revision_timesteps]

    @property
    def n_edit_groups_per_timestep(self) -> List[List[int]]:
        """Number of groups of edits for each time step."""
        n_edits = []
        for row in np.tril(self.edits, k=-1):
            n = [len(list(group)) for key, group in groupby(row) if key == 1]
            n_edits.append(n)
        return n_edits

    @property
    def n_edit_groups_per_revision(self) -> np.array:
        """Number of groups of edits for each time step with a revision."""
        edit_groups_per_timestep = self.n_edit_groups_per_revision()
        return edit_groups_per_timestep[self.revision_timesteps]

    @property
    def edit_distances(self) -> List[List]:
        """Distance of each edit to current step, for all steps."""
        distances = []
        for time_id, row in enumerate(np.tril(self.edits, k=-1)):
            row_dists = []
            for label_id in range(time_id):
                if row[label_id] == 1.:
                    row_dists.append(time_id - label_id)
            distances.append(row_dists)
        return distances

    @property
    def label_diversity_per_token(self) -> np.array:
        """Number of label types assigned to each token."""
        df = pd.DataFrame(self.chart).replace([EMPTY], np.nan)
        return df.nunique(dropna=True).to_list()

    def n_edits_with_quality_by_turn(self, quality: str):
        """Number of edits with a certain quality at each time step."""
        n_edits_quality = []
        for row in self.edit_qualities.chart:
            n = sum([1 for edit in row if self._has_quality(edit, quality)])
            n_edits_quality.append(n)
        return np.array(n_edits_quality)

    @staticmethod
    def _has_quality(var: Optional[EditQualities], quality: str) -> bool:
        """Check if a cell contains an edit that has a certain quality."""
        return var is not None and getattr(var, quality)

    @property
    def possible_edits_per_token(self) -> np.array:
        """Maximum number of edits that can happen for each token."""
        return np.arange(self.n_tokens - 1, -1, -1)

    @property
    def possible_edits_per_timestep(self) -> np.array:
        """Maximum number of edits that can happen at each time step."""
        return np.arange(0, self.n_tokens)

    @property
    def outputs_per_token(self) -> np.array:
        """Number of outputs for each token."""
        return np.arange(self.n_tokens, 0, -1)

    @property
    def total_possible_edits(self) -> np.array:
        """Maximum number of edits that can happen for a sequence."""
        return np.sum(np.arange(1, self.n_tokens))

    @property
    def total_possible_outputs(self) -> np.array:
        """Total output labels for a sequence."""
        return np.sum(np.arange(1, self.n_tokens + 1))

    def n_edits_with_quality_per_revision(self, quality: str) -> int:
        """Number of edits with a certain quality at revision steps."""
        edits = self.n_edits_with_quality_by_turn(quality)
        return edits[self.revision_timesteps]

    def total_edits_with_quality(self, quality: str) -> int:
        """Total number of edits with a certain quality in a chart."""
        return np.sum(self.n_edits_with_quality_by_turn(quality))

    def perc_edits_with_quality(self, quality: str) -> float:
        """% of all edits that have a certain quality."""
        return 100 * self.total_edits_with_quality(quality) / self.n_total_edits
        #which_total =  if mode == 'edits' else self.total_possible_edits
