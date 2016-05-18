"""
Figures for analyzing single-neuron timescales.
"""
from __future__ import division, print_function
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

import data_io
import munging
from plot import set_fontsize


def down_sampled_interval_variation_vs_chance(
        SEED,
        DATA_DIRECTORY, DATA_FILENAME,
        DT_ORIGINAL, DOWN_SAMPLE_TIMESCALE,
        SPIKE_THRESHOLD,
        N_SHUFFLES,
        EXAMPLE_CELL_PROBABILITIES, EXAMPLE_CELL_PROBABILITIES_N_SECONDS,
        SPIKE_TRAIN_CELLS, SPIKE_TRAINS_N_SECONDS,
        FIG_SIZE, FONT_SIZE):
    """
    This figure asks whether slow timescales are present in spike probabilities extracted from calcium imaging data.
    """

    ## PERFORM ANALYSIS

    # load data

    data_loader = data_io.DataLoader(DATA_DIRECTORY)

    spike_probs = data_loader.foopsi(DATA_FILENAME)
    spike_trains = spike_probs >= SPIKE_THRESHOLD

    # calculate down-sample factor (number of time steps)

    down_sample_factor = int(np.round(DOWN_SAMPLE_TIMESCALE / DT_ORIGINAL))

    spike_probs_down_sampled, t_start_idxs, t_end_idxs = munging.down_sample_spike_probabilities(
        spike_probs, down_sample_factor)
    spike_trains_down_sampled = spike_probs_down_sampled >= SPIKE_THRESHOLD

    # make time vectors

    ts = np.arange(spike_probs.shape[1]) * DT_ORIGINAL
    ts_down_sampled = ts[np.round((t_start_idxs + t_end_idxs) / 2).astype(int)]


    ## MAKE PLOTS

    fig = plt.figure(facecolor='white', figsize=FIG_SIZE, tight_layout=True)
    gs = gridspec.GridSpec(6, 1)
    axs = []

    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1:]))

    # plot example spike probability and its down-sampled version

    # plot original probabilities and spikes

    ts_currents = [ts, ts_down_sampled]
    spike_probs_currents = [spike_probs, spike_probs_down_sampled]
    spike_trains_currents = [spike_trains, spike_trains_down_sampled]
    bar_widths = [DT_ORIGINAL, DT_ORIGINAL*down_sample_factor]
    colors = ['b', 'k']

    for ctr, color in enumerate(colors):

        ts_current = ts_currents[ctr]
        spike_probs_current = spike_probs_currents[ctr]
        spike_trains_current = spike_trains_currents[ctr]
        bar_width = bar_widths[ctr]

        example_idxs = ts_current < EXAMPLE_CELL_PROBABILITIES_N_SECONDS
        y_offset = ctr * 2

        axs[0].bar(
            ts_current[example_idxs],
            spike_probs_current[EXAMPLE_CELL_PROBABILITIES, example_idxs],
            width=bar_width, align='center', bottom=y_offset, color=color, lw=0)

        spike_train_example = spike_trains_current[EXAMPLE_CELL_PROBABILITIES, example_idxs]

        spike_times = ts_current[spike_train_example.nonzero()[0]]
        spike_times_y_coord = (y_offset + 1.5) * np.ones(spike_times.shape)

        axs[0].scatter(spike_times, spike_times_y_coord, s=200, c=color, marker='|', lw=2)

    axs[0].set_xlim(0, EXAMPLE_CELL_PROBABILITIES_N_SECONDS)
    axs[0].set_ylim(0, 4)

    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('spike probabilities/\nspikes')

    # plot spike trains for other cells

    spike_train_plot_t_idxs = ts_down_sampled < SPIKE_TRAINS_N_SECONDS

    for ctr, cell_idx in enumerate(SPIKE_TRAIN_CELLS):

        y_offset = ctr

        spike_train = spike_trains_down_sampled[cell_idx, spike_train_plot_t_idxs]
        spike_times = ts_down_sampled[spike_train.nonzero()[0]]
        spike_times_y_coord = (y_offset + 0.5) * np.ones(spike_times.shape)

        axs[1].scatter(spike_times, spike_times_y_coord, s=200, c='k', marker='|', lw=2)

    axs[1].set_xlim(0, SPIKE_TRAINS_N_SECONDS)
    axs[1].set_ylim(0, len(SPIKE_TRAIN_CELLS))

    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('cell')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)