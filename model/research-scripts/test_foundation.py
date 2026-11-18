"""Check sustained-event evaluation boundary and causal smoothing."""
import unittest
from importlib import import_module
import numpy as np

f = import_module('06_foundation')


class EventEvaluationTests(unittest.TestCase):
    def test_six_windows_do_not_meet_three_seconds(self):
        self.assertEqual(f.event_score(np.ones(6)), 0)

    def test_seven_windows_meet_three_seconds(self):
        self.assertEqual(f.event_score(np.ones(7)), 1)

    def test_smoothing_does_not_use_future(self):
        np.testing.assert_allclose(f.smooth(np.array([0., 0., 0., 1.])), [0., 0., 0., .25])

    def test_single_spike_is_not_sustained(self):
        self.assertEqual(f.event_score(np.r_[np.zeros(10), 1., np.zeros(10)]), 0)


if __name__ == '__main__':
    unittest.main()
