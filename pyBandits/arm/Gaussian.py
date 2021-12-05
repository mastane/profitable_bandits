# -*- coding: utf-8 -*-
'''Gaussian distributed arm.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.4 $"

import random as rand
from Arm import Arm
from scipy.stats import poisson

class Gaussian(Arm):
    """Gaussian distributed arm."""
    def __init__(self, mu, sigma, lambda_poisson=5):
        self.sigma = sigma
        self.mu=mu
        self.expectation = mu
        self.lambda_poisson = lambda_poisson

    def draw(self):
        c = poisson.rvs(lambda_poisson-1)+1
		return [self.mu+self.sigma*rand.gauss(0,1) for i in range(c)]
