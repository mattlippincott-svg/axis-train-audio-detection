import unittest
import numpy as np
from rail_fusion import rail_scores

class RailPolicyTests(unittest.TestCase):
    def score(self,base=.1,clatter=-20,squeal=20,tags=None,context=False):
        tags=np.zeros((1,521)) if tags is None else tags
        return rail_scores(np.array([base]),np.array([clatter]),np.array([squeal]),tags,0,context)

    def test_squeal_alone_cannot_add_support(self):
        tags=np.zeros((1,521));tags[0,327]=1
        score,clatter,support=self.score(tags=tags)
        self.assertEqual(float(support[0]),0)
        self.assertAlmostEqual(float(score[0]),.1)

    def test_engine_and_alarm_do_not_qualify(self):
        tags=np.zeros((1,521));tags[0,[337,382,392]]=1
        self.assertEqual(float(self.score(tags=tags)[2][0]),0)

    def test_bell_allows_squeal_support(self):
        tags=np.zeros((1,521));tags[0,195]=.8
        score,clatter,support=self.score(tags=tags)
        self.assertGreater(float(support[0]),.119)
        self.assertGreater(float(score[0]),.219)

    def test_clatter_can_trigger_without_bell_horn_or_squeal(self):
        score,clatter,support=self.score(clatter=5,squeal=-20)
        self.assertGreater(float(score[0]),.869)
        self.assertGreater(float(clatter[0]),.869)

    def test_context_gate_blocks_unrelated_impacts(self):
        score,clatter,support=self.score(clatter=20,context=True)
        self.assertEqual(float(clatter[0]),0)
        self.assertAlmostEqual(float(score[0]),.1)

    def test_baseline_is_preserved(self):
        self.assertAlmostEqual(float(self.score(base=.95,context=True)[0][0]),.95)

if __name__=='__main__':unittest.main()
