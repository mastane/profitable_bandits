# -*- coding: utf-8 -*-
'''Generic index policy.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.5 $"


from random import choice
from scipy.stats import poisson
import numpy as np

from Policy import *

class IndexPolicy(Policy):
    """Class that implements a generic index policy."""

#  def __init__(self):

#  def computeIndex(self, arm):

    def choice(self):
        """In an index policy, choose at random an arm with maximal index."""
        index = dict()
        for arm in range(self.nbArms):
            index[arm] = self.computeIndex(arm)
        chosen_set = [arm for arm in range(self.nbArms) if index[arm]>self.thresholds[arm]]
        return chosen_set
