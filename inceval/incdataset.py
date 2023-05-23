#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to represent the incremental outputs charts for all sentences in a
dataset, and retrieve dataset-level metrics.
"""

from typing import List
import numpy as np

class IncData:
    """
    Represent a complete dataset, where each item in self.instances
    is the incremental chart of one sentence in the dataset.
    """
    def __init__(self, output_dic: dict):
        self.instances = output_dic
        self.seqs = self.instances.values()

    def edits_with_quality(self, quality: str) -> List[int]:
        """Number of edits with a given quality per sentence."""
        return [seq.total_edits_with_quality(quality) for seq in self.seqs]

    def perc_edits_with_quality(self, quality: str) -> List[int]:
        """% edits with a given quality in the dataset."""
        n_edits_quality = np.sum(self.edits_with_quality(quality))
        return 100 * n_edits_quality / self.get_total('n_total_edits')

    def revisions_with_quality(self, quality: str) -> List[int]:
        """Number of revisions with a given quality per sentence."""
        return [seq.revision_qualities.n_revisions_with_quality(quality)
                for seq in self.seqs]

    def perc_revisions_with_quality(self, quality: str) -> float:
        """% revisions with a given quality in the dataset."""
        n_revisions_quality = np.sum(self.revisions_with_quality(quality))
        return 100 * n_revisions_quality / self.get_total('n_revisions')

    def get_dist(self, attr):
        """Distribution of number of a given attribute in a sentence."""
        return [getattr(seq, attr) for seq in self.seqs]

    def get_total(self, attr):
        """Total number of a given attribute in the dataset."""
        return np.sum(self.get_dist(attr))

    def get_perc(self, attr, denominator):
        """% number of a given attribute in the dataset.
        
        Denominator required because we may want to divide by total number of
        tokens, or total number of edits, or revisions etc.
        """
        return self.get_total(attr) / self.get_total(denominator)

    def get_mean(self, attr):
        """Mean of a given attribute in the dataset."""
        return np.mean(self.get_dist(attr))

    def get_std(self, attr):
        """Standard deviation of a given attribute in the dataset."""
        return np.std(self.get_dist(attr))

    @property
    def perc_revisions(self) -> float:
        """% of timesteps with revisions in the dataset."""
        return 100 * self.get_total('n_revisions') / self.get_total('n_tokens')

    @property
    def perc_recomputations(self) -> float:
        """% of timesteps with recomputations in the dataset."""
        return 100 * self.get_total('n_recomputations') / self.get_total('n_tokens')

    @property
    def perc_active_recomputations(self) -> float:
        """% of recomputations that caused revisions in the dataset."""
        return 100 * self.get_total('n_active_recomputations') / self.get_total('n_recomputations')

    @property
    def r_pertinence(self) -> float:
        """Dataset-level revision-pertinence."""
        numerator = self.get_total('n_revision_and_incorrect_prefix')
        denominator = self.get_total('n_revisions')
        return numerator / denominator

    @property
    def a_pertinence(self) -> float:
        """Dataset-level addition-pertinence."""
        numerator = self.get_total('n_write_and_correct_prefix')
        denominator = self.get_total('n_writes')
        return  numerator / denominator

    @property
    def r_appropriateness(self) -> float:
        """Dataset-level revision-appropriateness."""
        numerator = self.get_total('n_revision_and_incorrect_prefix')
        denominator = self.get_total('n_incorrect_prefixes')
        return numerator / denominator

    @property
    def a_appropriateness(self) -> float:
        """Dataset-level addition-appropriateness."""
        numerator = self.get_total('n_write_and_correct_prefix')
        denominator = self.get_total('n_correct_prefixes')
        return numerator / denominator
