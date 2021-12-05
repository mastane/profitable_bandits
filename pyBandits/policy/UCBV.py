# -*- coding: utf-8 -*-
'''The UCB-V policy for bounded bandits.
  Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"


from math import sqrt, log
import numpy as np

from IndexPolicy import IndexPolicy

class UCBV(IndexPolicy):
    """Class that implements the UCB-V policy.
    """

    def __init__(self, nbArms, thresholds, amplitude=1., lower=0.):
        self.nbArms = nbArms
        self.amplitude = amplitude
        self.lower = lower
        self.nbDraws = dict()
        self.cumReward = dict()
        self.cumReward2 = dict()
        self.thresholds = thresholds

    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0
            self.cumReward2[arm] = 0.0

    def computeIndex(self, arm):
        if self.nbDraws[arm] < 2:
            return float('+infinity')
        else:
            m = self.cumReward[arm]/self.nbDraws[arm]
            v = self.cumReward2[arm]/self.nbDraws[arm] - m*m
            return m + sqrt(2*log(self.t) * v / self.nbDraws[arm]) + 3*self.amplitude*log(self.t)/self.nbDraws[arm]

    def getReward(self, arms, rewards):
        for i in range(len(arms)):
            c = len(rewards[i])
            self.nbDraws[arms[i]] += c
            self.cumReward[arms[i]] += np.sum(rewards[i])
            self.cumReward2[arms[i]] += np.sum(r**2 for r in rewards[i])
        self.t += 1
