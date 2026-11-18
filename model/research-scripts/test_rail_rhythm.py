import unittest
import numpy as np
from rail_rhythm import rhythm_features,LAGS

class RhythmTests(unittest.TestCase):
    def test_future_audio_cannot_change_past_features(self):
        rng=np.random.default_rng(7);w=rng.normal(0,.01,96000).astype(np.float32)
        changed=w.copy();changed[64000:]=1
        np.testing.assert_array_equal(rhythm_features(w,[64000]),rhythm_features(changed,[64000]))

    def test_gain_does_not_change_rhythm(self):
        w=np.zeros(64000,np.float32);w[::4000]=.1
        np.testing.assert_allclose(rhythm_features(w,[64000]),rhythm_features(w*.01,[64000]),atol=1e-5)

    def test_silence_and_short_context_are_finite(self):
        a=rhythm_features(np.zeros(20000),[1,15600,20000])
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a.shape,(3,4*(2*len(LAGS)+8)))

    def test_periodic_impacts_have_expected_lag(self):
        w=np.zeros(64000,np.float32);w[::4000]=.1
        a=rhythm_features(w,[64000])[0]
        self.assertGreater(a[:len(LAGS)].max(),.2)

    def test_cached_spectra_match_each_exact_context(self):
        rng=np.random.default_rng(11);w=rng.normal(0,.01,96000).astype(np.float32)
        ends=[15600,23280,31000,64000,71680,90000]
        batch=rhythm_features(w,ends)
        for i,end in enumerate(ends):
            segment=w[max(0,end-64000):end]
            segment=np.pad(segment,(64000-len(segment),0))
            expected=rhythm_features(segment,[64000])[0]
            np.testing.assert_allclose(batch[i],expected,atol=1e-7)

if __name__=='__main__':unittest.main()
