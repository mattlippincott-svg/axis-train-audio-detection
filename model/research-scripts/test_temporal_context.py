import unittest
import numpy as np
from temporal_context import context_indices

class ContextTests(unittest.TestCase):
    def test_context_never_contains_future_of_target(self):
        groups=[('a',np.arange(5)),('b',np.arange(5,10))]
        x=context_indices(10,groups,groups,context=4)
        for family,ids in groups:
            for j,index in enumerate(ids):
                self.assertEqual(x[index,-1],index)
                self.assertFalse(set(x[index]).intersection(ids[j+1:]))
        np.testing.assert_array_equal(x[0],[7,8,9,0])
        np.testing.assert_array_equal(x[5],[2,3,4,5])

    def test_same_family_other_version_cannot_supply_prefix(self):
        groups=[('same',np.arange(3)),('same',np.arange(3,6))]
        with self.assertRaises(ValueError):context_indices(6,groups,groups,context=3)

    def test_short_prefix_repeats_only_available_past(self):
        x=context_indices(4,[('a',np.arange(3))],[('b',np.array([3]))],context=4)
        np.testing.assert_array_equal(x[0],[3,3,3,0])
        np.testing.assert_array_equal(x[2],[3,0,1,2])

    def test_cold_start_uses_first_frame_without_future(self):
        x=context_indices(3,[('a',np.arange(3))],[],context=3,continuous=False)
        np.testing.assert_array_equal(x,[[0,0,0],[0,0,1],[0,1,2]])

if __name__=='__main__':unittest.main()
