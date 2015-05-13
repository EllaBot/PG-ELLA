from pg_ella.pgpe import PGPE
from numpy.testing import assert_allclose

class TestPGPE(object):

    def test_pgpe(self):
        optimaltheta = [-1, 0, 3, 0]
        pgpe = PGPE(4, epsilon=1.0)
        for x in range(0, 10000):
            thetas = pgpe.getperturbedthetas()
            reward1 = evaluator(thetas[0], optimaltheta)
            reward2 = evaluator(thetas[1], optimaltheta)
            print reward1
            print reward2
            pgpe.learn(reward1, reward2)

        assert_allclose(optimaltheta, pgpe.theta, atol=.01)
        # Should learn to make params


def evaluator( params, optimaltheta):
    diff = lambda param, optimal: abs(param - optimal)
    return -5.0 * sum(map(diff, params, optimaltheta))