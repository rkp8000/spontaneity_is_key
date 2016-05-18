"""
Various standard analysis metrics.
"""
from __future__ import division
import numpy as np


def inter_spike_intervals(spike_train, dt):
    """
    Get the inter-spike intervals for a spike train.
    :param spike_train:
    :return: inter-spike intervals
    """

    return np.diff(spike_train.nonzero()[0]) * dt