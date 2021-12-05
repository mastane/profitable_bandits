# -*- coding: utf-8 -*-
'''The Bayes-UCB policy.
 Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012]'''

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.7 $"

from random import choice
import numpy as np

from posterior import Posterior
from IndexPolicy import IndexPolicy

class BayesUCB(IndexPolicy):
    """The Bayes-UCB.
      Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def __init__(self, nbArms, posterior, thresholds, amplitude=1., lower=0.):
        self.nbArms = nbArms
        self.amplitude = amplitude
        self.lower = lower
        self.posterior = dict()
        for arm in range(self.nbArms):
            self.posterior[arm] = posterior()
        self.nbDraws = dict()
        self.cumReward = dict()
        self.thresholds = thresholds

    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.posterior[arm].reset()
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0

    def getReward(self, arms, rewards):
        for i in range(len(arms)):
            c = len(rewards[i])
            self.nbDraws[arms[i]] += c
            self.cumReward[arms[i]] += np.sum([(r - self.lower) / self.amplitude for r in rewards[i]])
            for j in range(c):
                self.posterior[arms[i]].update(rewards[i][j])
        self.t += 1

    def computeIndex(self, arm):
        return self.posterior[arm].quantile(1-1./self.t)
