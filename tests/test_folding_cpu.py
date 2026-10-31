"""Small deterministic folding tests; no CUDA, circuit shots or private data."""
import unittest
import numpy as np
from shor_qem.folding import build_shotwise_folding_masks


class FoldingTests(unittest.TestCase):
    def test_cardinality_seed_and_endpoint_invariants(self):
        # The public historical helper has a positional scientific contract.
        configs=[(0.0001,1.0),(0.0001,1.4),(0.0001,3.0)]
        first=build_shotwise_folding_masks(configs,9,0,3,50,global_seed=42,N=119,a=13)
        second=build_shotwise_folding_masks(configs,9,0,3,50,global_seed=42,N=119,a=13)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first.sum(axis=0),np.repeat([0,10,50],9))


if __name__=='__main__':
    unittest.main()
