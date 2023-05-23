#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two classes, one to represent the qualities of a single edit and one to
represent all edits in an incremental chart.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

from inceval.aux import build_empty_chart, EMPTY

array = np.array
Label = Union[str, int, float]


@dataclass
class EditQualities:
    """Characteristics of an edit on a label."""
    range_param: int
    # effectiveness
    effective: bool = False
    defective: bool = False
    ineffective: bool = False
    # convenience
    convenient: bool = False
    inconvenient: bool = False
    # novelty
    innovative: bool = False
    repetitive: bool = False
    # recurrence (vertical)
    recurrent: bool = False
    steady: bool = False
    # oscillation (vertical)
    oscillating: bool = False
    stable: bool = False
    # connectedness (horizontal)
    connected: bool = False
    disconnected: bool = False
    # company
    accompanied: bool = False
    isolated: bool = False
    # distance
    short: bool = False
    long: bool = False
    # definiteness
    temporary: bool = False
    definite: bool = False
    # time
    intermediate: bool = False
    final: bool = False

    @staticmethod
    def edited(previous_label, current_label):
        """Return True if labels differ."""
        return previous_label != current_label

    def set_effectiveness(self, previous_label: Label, current_label: Label,
                          gold_label: Label) -> None:
        """Set the effectiveness attribute of the edit."""
        assert self.edited(previous_label, current_label), "Label not edited."
        if previous_label == gold_label and current_label != gold_label:
            self.defective = True
        elif previous_label != gold_label and current_label != gold_label:
            self.ineffective = True
        elif previous_label != gold_label and current_label == gold_label:
            self.effective = True

    def set_convenience(self, previous_label: Label, current_label: Label,
                        gold_label: Label) -> None:
        """Set the convenience attribute of the edit."""
        assert self.edited(previous_label, current_label), "Label not edited."
        if previous_label == gold_label:
            self.inconvenient = True
        if previous_label != gold_label:
            self.convenient = True

    def set_novelty(self, previous_labels: array,
                    current_label: Label) -> None:
        """Set the novelty attribute of the edit."""
        if current_label in previous_labels:
            self.repetitive = True
        else:
            self.innovative = True

    def set_recurrence(self, label_id: int, time_id: int, vertical_seq: array,
                       n_tokens: int) -> None:
        """Set the recurrence attribute of the edit."""
        assert label_id != time_id, "This is an addition, not an edit."
        assert vertical_seq[time_id] == 1., "Label not edited!"
        if label_id < (time_id-1) and vertical_seq[time_id - 1] == 1.:
            # check previous timestep, except for cases where the previous
            # time step was the addition
            self.recurrent = True
        elif time_id != (n_tokens - 1) and vertical_seq[time_id + 1] == 1.:
            # check the next time step, except for cases where the next
            # time step does not exist
            self.recurrent = True
        else:
            self.steady = True

    def set_oscillation(self, vertical_seq: array):
        """Set the oscillation attribute of the edit."""
        edits = vertical_seq[vertical_seq != EMPTY]
        assert not edits.sum() < 2, "Label not edited!"
        if edits.sum() == 2:
            self.stable = True
        elif edits.sum() > 2:
            self.oscillating = True

    def set_connectedness(self, label_id: int, time_id: int,
                          current_edits: array) -> None:
        """Set the conectedness attribute of the edit."""
        assert current_edits[label_id] == 1., "Label not edited!"
        if label_id != 0 and current_edits[label_id - 1] == 1.:
            self.connected = True
        elif label_id < (time_id - 1) and current_edits[label_id + 1] == 1.:
            # main diagonal always an addition
            self.connected = True
        else:
            self.disconnected = True

    def set_company(self, edit_prefix: array) -> None:
        """Set the company attribute of the edit."""
        assert edit_prefix.sum() > 0, "No edits!"
        if edit_prefix.sum() > 1.:
            self.accompanied = True
        else:
            self.isolated = True

    def set_distance(self, time_id: int, label_id: int) -> None:
        """Set the distance attribute of the edit."""
        if label_id < time_id - self.range_param:
            self.long = True
        else:
            self.short = True

    def set_definiteness(self, time_id: int, current_seq: array,
                         n_tokens: int) -> None:
        """Set the definiteness attribute of the edit."""
        if time_id == n_tokens or current_seq[time_id + 1:].sum() == 0.:
            self.definite = True
        else:
            self.temporary = True

    def set_time(self, time_id: int, n_tokens: int) -> None:
        """Set the time attribute of the edit."""
        if time_id == n_tokens - 1:
            self.final = True
        else:
            self.intermediate = True

    def set_qualities(self, current_label: Label, previous_label: Label,
                      gold_label: Label, previous_labels: array,
                      current_edits: array, vertical_edits: array,
                      time_id: int, label_id: int, n_tokens: int) -> None:
        """Set all edit's qualities."""
        # initialise and fill in the qualities of the edit
        self.set_effectiveness(previous_label, current_label, gold_label)
        self.set_convenience(previous_label, current_label, gold_label)
        self.set_novelty(previous_labels, current_label)
        self.set_recurrence(label_id, time_id, vertical_edits, n_tokens)
        self.set_oscillation(vertical_edits)
        self.set_connectedness(label_id, time_id, current_edits)
        self.set_company(current_edits)
        self.set_distance(time_id, label_id)
        self.set_definiteness(time_id, vertical_edits, n_tokens)
        self.set_time(time_id, n_tokens)


class EditQualityChart:
    """Represent the incremental chart with the characteristics of edits."""
    def __init__(self, outputs: array, edits: array, gold: array,
                 range_param: int):
        self._check_input(outputs, edits, gold)
        self.n_tokens = outputs.shape[0]
        self.range_param = range_param
        self.chart = self._fill_chart(outputs, edits, gold)

    @staticmethod
    def _check_input(outputs: array, edits: array, gold: array) -> None:
        """Check that sizes match."""
        assert outputs.shape == edits.shape
        assert gold.shape[0] == outputs.shape[0]

    def _fill_chart(self, outputs: array, edits: array, gold: array) -> array:
        """Extract and log qualities of each edit in the chart."""
        chart = build_empty_chart(self.n_tokens, filler=None)
        for time_id, label_id in zip(*np.tril_indices_from(edits, k=-1)):
            # looping over all elements in the lower portion of the matrix;
            # the main diagonal is ignored because it only contains additions
            if edits[time_id, label_id] == 0.:
                # label was not edited
                continue
            assert edits[time_id, label_id] == 1.
            edit = self._build_edit(time_id, label_id, outputs, edits, gold)
            chart[time_id, label_id] = edit
        return chart

    def _build_edit(self, time_id: int, label_id: int, outputs: array,
                    edits: array, gold: array) -> EditQualities:
        """Extract all subcomponents and build an edit object."""
        # get some specific components
        current_label = outputs[time_id, label_id]
        previous_label = outputs[time_id - 1, label_id]
        gold_label = gold[label_id]
        previous_labels = outputs[: time_id, label_id]
        current_edits = edits[time_id, : time_id]
        vertical_edits = edits[:, label_id]
        assert previous_label != current_label
        # build the edit
        edit = EditQualities(range_param=self.range_param)
        edit.set_qualities(
            current_label, previous_label, gold_label, previous_labels,
            current_edits, vertical_edits, time_id, label_id, self.n_tokens)
        return edit
