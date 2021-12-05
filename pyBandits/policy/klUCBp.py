# -*- coding: utf-8 -*-
'''The generic kl-UCB policy for one-parameter exponential distributions.
  Reference: [Garivier & cappé - COLT, 2011].'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.15 $"

from math import log
import numpy as np

import kullback
from IndexPolicy import IndexPolicy

class klUCBp(IndexPolicy):
    """kl-UCB+
      """
    def __init__(self, nbArms, thresholds, amplitude=1., lower=0., klucb=kullback.klucbBern):
        self.c = 1.
        self.nbArms = nbArms
        self.amplitude = amplitude
        self.lower = lower
        self.nbDraws = dict()
        self.cumReward = dict()
        self.klucb = klucb
        self.thresholds = thresholds

    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0

    def computeIndex(self, arm):
        if self.nbDraws[arm] == 0:
            return float('+infinity')
        else:
            C = 10  # arbitrary number to avoid negative log
            return self.klucb(self.cumReward[arm] / self.nbDraws[arm], self.c * log(C*self.t/self.nbDraws[arm]) / self.nbDraws[arm], 1e-4) # Could adapt tolerance to the value of self.t

    def getReward(self, arms, rewards):
        for i in range(len(arms)):
            c = len(rewards[i])
            self.nbDraws[arms[i]] += c
            self.cumReward[arms[i]] += np.sum([(r - self.lower) / self.amplitude for r in rewards[i]])
        self.t += 1
    # Debugging code
    #print "arm " + str(arm) + " receives " + str(reward)
    #print str(self.nbDraws[arm]) + " " + str(self.cumReward[arm])
