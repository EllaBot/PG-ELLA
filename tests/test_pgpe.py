from pg_ella.pgpe import PGPE
from numpy.testing import assert_allclose

class TestPGPE(object):

    def test_pgpe(self):
        optimaltheta = [-1, 1, 1, -1]
        diff = lambda param, optimal: abs(param - optimal)
        evaluator = lambda params: 5.0 * sum(map(diff, params, optimaltheta))
        pgpe = PGPE(evaluator, 4)
        for x in range(0, 10000):
            pgpe.learn()

        assert_allclose(optimaltheta, pgpe.theta, 0.0001)
        # Should learn to make params 0
