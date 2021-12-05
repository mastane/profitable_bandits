# -*- coding: utf-8 -*-
'''The Thompson (Bayesian) index policy.
Reference: [Thompson - Biometrika, 1933].
'''

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.9 $"


from random import choice
import numpy as np
from scipy import stats

from posterior.GammaForPoisson import GammaForPoisson
from posterior.GammaForExp import GammaForExp
from IndexPolicy import IndexPolicy

class Thompson(IndexPolicy):
  """The Thompson (Bayesian) index policy.
  Reference: [Thompson - Biometrika, 1933].
  """

  def __init__(self, nbArms, posterior, thresholds):
    self.nbArms = nbArms
    self.posterior = dict()
    for arm in range(self.nbArms):
        self.posterior[arm] = posterior()
    self.thresholds = thresholds

  def startGame(self):
    self.t = 1;
    for arm in range(self.nbArms):
        self.posterior[arm].reset()

  def getReward(self, arms, rewards):
    for i in range(len(arms)):
        c = len(rewards[i])
        for j in range(c):
            self.posterior[arms[i]].update(rewards[i][j])
    self.t += 1

  def computeIndex(self, arm):
      return self.posterior[arm].sample()
