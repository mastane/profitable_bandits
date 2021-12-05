# -*- coding: utf-8 -*-
'''Bernoulli distributed arm.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.6 $"

from random import random
from Arm import Arm
from scipy.stats import poisson

class Bernoulli(Arm):
    """Bernoulli distributed arm."""

    def __init__(self, p, lambda_poisson=5):
        self.p = p
        self.expectation = p
        self.lambda_poisson = lambda_poisson

    def draw(self):
        c = poisson.rvs(self.lambda_poisson-1)+1
        return [float(random()<self.p) for i in range(c)]
