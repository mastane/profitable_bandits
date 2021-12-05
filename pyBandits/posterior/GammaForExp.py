# -*- coding: utf-8 -*-
'''Gamma posterior for Exponential bandits (cf. Jeffreys prior in Korda 2013)'''

__author__ = "Mastane Achab"
__version__ = "$Revision: 1 $"

from Posterior import Posterior

from random import gammavariate
from scipy.stats import gamma, invgamma

class GammaForExp(Posterior):
    """Manipulate posteriors of Bernoulli/Beta experiments.
    """
    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b

    def reset(self, a=0, b=0):
        if a==0:
            a = self.a
        if b==0:
            b = self.b
        self.N = [a, b]

    def update(self, obs):
        self.N[0] += 1
        self.N[1] += obs

    def sample(self):
        # take inverse of lambda for the mean of Exponential distribution
        return 1./gamma.rvs(self.N[0], scale=1./self.N[1])

    def quantile(self, p):
        # take quantile of the inverse Gamma distribution for the mean of Exponential distribution
        return invgamma.ppf(p, self.N[0], scale=self.N[1])
