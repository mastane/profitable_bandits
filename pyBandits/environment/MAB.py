# -*- coding: utf-8 -*-
'''
Environement for a Multi-armed bandit problem
with arms given in the 'arms' list
'''

__author__ = "Olivier Cappé,Aurélien Garivier"
__version__ = "$Revision: 1.5 $"

from Result import *
from Environment import Environment

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, arms, thresholds):
        self.arms = arms
        self.nbArms = len(arms)
        self.thresholds = thresholds
        # supposed to have a property nbArms

    def play(self, policy, horizon):
        policy.startGame()
        result = Result(self.nbArms, horizon, self.thresholds)
        # initialization t=0: pull all arms
        chosen_set = range(self.nbArms)
        rewards = [self.arms[arm].draw() for arm in chosen_set]
        policy.getReward(chosen_set, rewards)
        result.store(0, chosen_set, rewards)
        # run algo for t>0
        for t in range(1, horizon):
            chosen_set = policy.choice()
            rewards = [self.arms[arm].draw() for arm in chosen_set]
            policy.getReward(chosen_set, rewards)
            result.store(t, chosen_set, rewards)
        return result
