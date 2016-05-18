"""
Functions for munging of data.
"""
from __future__ import division
import numpy as np


def down_sample_spike_probabilities(spike_probs, down_sample_factor):
    """
    Downsample time-series of spike probabilities by a certain factor.
    At each time point for each cell, calculate the probability that the cell spiked at least once.

    :param data: data array (rows are cells, cols are timepoints)
    :param down_sample_factor: how many frames to use for each new frame
    return down-sampled data array, original start times and end times for each down-sampled time point
    """

    n_cells = spike_probs.shape[0]
    n_time_points = spike_probs.shape[1]

    n_time_points_down_sampled = n_time_points // down_sample_factor

    spike_probs_down_sampled = np.nan * np.zeros((n_cells, n_time_points_down_sampled), dtype=float)

    t_starts = np.arange(0, n_time_points_down_sampled * down_sample_factor, down_sample_factor)
    t_ends = t_starts + down_sample_factor

    for t_down_sampled, t_start in enumerate(t_starts):

        spike_probs_at_t = 1 - np.prod(1 - spike_probs[:, t_start:t_start + down_sample_factor], axis=1)

        spike_probs_down_sampled[:, t_down_sampled] = spike_probs_at_t

    return spike_probs_down_sampled, t_starts, t_ends