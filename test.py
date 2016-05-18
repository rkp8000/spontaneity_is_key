from __future__ import division
import numpy as np
import unittest


import munging


class MungingTestCase(unittest.TestCase):

    def test_down_sample_spike_probabilities(self):

        spike_probs_original = np.array([
            [ 0,  0,  0, .1,  0,  0, .2, .3,  0,  0, .9,  0, .1],
            [.3, .3,  0, .4, .1,  0, .2, .1, .7, .9, .1,  0,  0],
            [.1, .9, .9, .9, .2, .5,  0,  0, .1, .4, .2, .1,  0],
        ])

        down_sample_factor = 4

        spike_probs_down_sampled_correct = np.array([
            [0.09999999999999998, 0.44000000000000006,    0.9],
            [ 0.7060000000000001, 0.35199999999999987,  0.973],
            [             0.9991,                 0.6, 0.6112],
        ])

        t_starts_correct = np.array([0, 4, 8])
        t_ends_correct = np.array([4, 8, 12])

        spike_probs_down_sampled, t_starts, t_ends = munging.down_sample_spike_probabilities(
            spike_probs_original, down_sample_factor)

        np.testing.assert_array_almost_equal(spike_probs_down_sampled, spike_probs_down_sampled_correct)
        np.testing.assert_array_equal(t_starts, t_starts_correct)
        np.testing.assert_array_equal(t_ends, t_ends_correct)

if __name__ == '__main__':

    unittest.main()