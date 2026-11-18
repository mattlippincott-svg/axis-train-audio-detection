import unittest
import numpy as np
from rail_recurrence import recurrence

class RecurrenceTests(unittest.TestCase):
    def test_one_short_burst_cannot_qualify(self):
        x=np.zeros(30);x[3:7]=1
        self.assertLess(recurrence(x).max(),.9)

    def test_two_separated_observations_qualify_then_expire(self):
        x=np.zeros(30);x[3]=.95;x[8]=.99
        r=recurrence(x)
        self.assertEqual(r[8],.95);self.assertEqual(r[14],.95);self.assertEqual(r[15],0)

    def test_future_changes_cannot_change_past(self):
        x=np.random.default_rng(1).normal(size=50);changed=x.copy();changed[25:]=100
        np.testing.assert_array_equal(recurrence(x)[:25],recurrence(changed)[:25])

    def test_strong_continuous_evidence_and_quiet_values(self):
        r=recurrence(np.full(20,.99));self.assertEqual(r[4],.99)
        self.assertTrue(np.isneginf(recurrence(np.full(20,-np.inf))).all())

if __name__=='__main__':unittest.main()
