#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions
"""

import enum
from typing import Optional

import numpy as np

EMPTY = np.inf

# We can compare prefixes to the gold labels or to the final output.
Criterion = enum.Enum('Criterion', ['GOLD', 'SILVER'])
SILVER = Criterion.SILVER
GOLD = Criterion.GOLD

def build_empty_chart(n: int, filler: Optional[float] = EMPTY) -> np.array:
    """Initialize a chart with a given shape, filled with EMPTY default."""
    return np.full([n, n], filler, dtype='O')

def accuracy(x: np.array, y: np.array) -> float:
    """Fraction of labels that match in x and y."""
    return np.mean(x == y)
