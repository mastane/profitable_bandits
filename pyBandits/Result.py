# -*- coding: utf-8 -*-
'''Utility class for handling the results of a Multi-armed Bandits experiment.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"


import numpy as np

class Result:
    """The Result class for analyzing the output of bandit experiments."""
    def __init__(self, nbArms, horizon, thresholds):
        self.nbArms = nbArms
        self.choices = [None]*horizon  # np.zeros(horizon)
        self.rewards = [None]*horizon  # np.zeros(horizon)
        self.profits = [None]*horizon  # np.zeros(horizon)
        self.thresholds = thresholds
        self.cumRewardPerRound = np.zeros(horizon)
        self.cumProfitPerRound = np.zeros(horizon)

    def store(self, t, chosen_set, rewards):
        self.choices[t] = chosen_set
        self.rewards[t] = rewards
        self.cumRewardPerRound[t] = np.sum(np.sum(rewards))
        self.cumProfitPerRound[t] = np.sum(np.sum(rewards)) - np.sum([self.thresholds[arm]*len(rewards[i]) for i, arm in enumerate(chosen_set)])

    def getNbObs(self):
        if (self.nbArms==float('inf')):
            self.nbObs=np.array([])
            pass
        else:
            nbObs = np.zeros(self.nbArms)
            for i, chosen_set in enumerate(self.choices):
                for arm in chosen_set:
                    nbObs[arm] += len(self.rewards[i])
            return nbObs

    def getRegret(self, optimal_arms, true_means):
        regret = 0
        nbObs = self.getNbObs()
        for arm in optimal_arms:
            regret += (true_means[arm]-self.thresholds[arm])*nbObs[arm]
        for t in range(horizon):
            for i in range(len(self.choices[t])):
                regret -= (np.sum(self.rewards[t][i])-self.thresholds[self.choices[t][i]]*len(self.rewards[t][i]))
        return regret
