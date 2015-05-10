from pg_ella.pgpe import PGPE
import math
from nose.tools import assert_greater, assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal

class TestPGPE(object):

    def test_pgpe(self):
        optimaltheta = [1, -1, 1, -1]
        diff = lambda param, optimal: abs(param - optimal)
        evaluator = lambda params: 5.0 * sum(map(diff, params, optimaltheta))
        pgpe = PGPE(evaluator)
        for x in range(0, 10000):
            pgpe.learn()

        assert assert_array_almost_equal(optimaltheta, pgpe.current)
        # Should learn to make params 0
