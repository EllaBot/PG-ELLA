import random
import numpy as np
from gradientdescent import GradientDescent


class PGPE():
    def __init__(self, evaluator, alphasigma=0.1, alphatheta=0.2):
        """

        :param evaluator: lambda that evaluates a policy
        """
        self.problemsize = 4

        self.epsilon = 2.0
        self.bestevaluation = -1000

        self.current = np.zeros(self.problemsize)
        self.gd = GradientDescent()
        self.gd.alpha = alphatheta
        self.gd.init(self.current)

        self.sigmalist = np.ones(self.problemsize, 'f') * self.epsilon
        self.gdsigma = GradientDescent()
        self.gdsigma.alpha = alphasigma
        self.gdsigma.init(self.sigmalist)

        self.deltas = np.zeros(self.problemsize, 'f')
        self.wdecay = 0.0
        self.evaluate = evaluator
        self.meanreward = 0
        self.baseline = 0

    def learn(self):
        deltas = self.perturbation()

        # Conduct two rollouts
        reward1 = self.evaluate(self.current + deltas)
        reward2 = self.evaluate(self.current - deltas)

        self.meanreward = (reward1 + reward2) / 2.
        if reward1 != reward2:
            fakt = (reward1 - reward2) / (
                2. * self.bestevaluation - reward1 - reward2)
        else:
            fakt = 0.0

        # normalized sigma gradient with moving average baseline
        norm = self.bestevaluation - self.baseline
        if norm != 0.0:
            fakt2 = (self.meanreward - self.baseline) / (
                self.bestevaluation - self.baseline)
        else:
            fakt2 = 0.0
        # update baseline
        self.baseline = 0.9 * self.baseline + 0.1 * self.meanreward
        # update parameters and sigmas
        self.current = self.gd(
            fakt * deltas - self.current * self.sigmalist * self.wdecay)
        # for sigma adaption alg. follows only positive gradients
        if fakt2 > 0.:
            # apply sigma update globally
            self.sigmalist = self.gdsigma(
                fakt2 * ((self.deltas ** 2).sum() - (self.sigmalist ** 2).sum())
                / (self.sigmalist * float(self.problemsize)))

    def perturbation(self):
        """ Generate a difference vector with the given standard deviations """
        return np.random.normal(0., self.sigmalist)