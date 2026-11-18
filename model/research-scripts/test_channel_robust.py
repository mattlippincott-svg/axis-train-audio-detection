"""Conditioning invariants that protect signal handling at the model boundary."""
import unittest
from importlib import import_module
import numpy as np

condition=import_module('13_channel_robust').condition


class ConditioningTests(unittest.TestCase):
    def test_silence_stays_silent_and_finite(self):
        out,level=condition(np.zeros((1,15600),np.float32))
        self.assertTrue(np.isfinite(out).all() and np.isfinite(level).all())
        self.assertFalse(out.any())

    def test_boost_does_not_exceed_limit(self):
        wave=np.full((1,15600),1e-5,np.float32)
        out,_=condition(wave)
        self.assertLessEqual(float(out.max()/wave.max()),10**1.8+.001)

    def test_normal_level_waveform_is_preserved(self):
        wave=np.linspace(-.5,.5,15600,dtype=np.float32)[None,:]
        out,_=condition(wave)
        np.testing.assert_array_equal(out,wave)

    def test_peak_limit_with_large_dynamic_range(self):
        wave=np.full((1,15600),1e-4,np.float32);wave[0,500]=.9
        out,_=condition(wave)
        self.assertLessEqual(float(np.max(np.abs(out))),.980001)
        self.assertGreaterEqual(float(out[0,500]),.9)


if __name__=='__main__':unittest.main()
