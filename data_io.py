"""
Functions and classes for loading data from .mat files.
"""
import numpy as np
import os
from scipy import io


class DataLoader(object):

    def __init__(self, data_dir):

        self.data_dir = data_dir

    def rois(self, filename):
        """
        :return: List of rois, each given as 2 column array of xy positions.
        """

        return io.loadmat(os.path.join(self.data_dir, filename))['CONTS'][0]

    def coords(self, filename):
        """
        :return: 2 column array of cell center coordinates
        """

        return np.loadtxt(os.path.join(self.data_dir, filename))

    def dfof(self, filename):
        """
        :return: DF/F array; rows are cells, cols are time points
        """

        return io.loadmat(os.path.join(self.data_dir, filename))['dfof']

    def foopsi(self, filename):
        """
        :return: Array of spike probabilities calculated using Foopsi algo; rows are cells, cols are time points
        """

        spike_probs = io.loadmat(os.path.join(self.data_dir, filename))

        if 'Traces_spon1_2_registered_1' in spike_probs.keys():

            return spike_probs['Traces_spon1_2_registered_1']

        elif 'foopsi' in spike_probs.keys():

            return spike_probs['foopsi']