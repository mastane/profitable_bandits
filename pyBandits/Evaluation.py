# -*- coding: utf-8 -*-
'''A utility class for evaluating the performance of a policy in multi-armed bandit problems.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.10 $"


import numpy as np
#from translate.misc.progressbar import ProgressBar

class Evaluation:

    def __init__(self, env, pol, nbRepetitions, horizon, tsav=[]):
        if len(tsav)>0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
        self.env = env
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.horizon = horizon
        self.nbArms = env.nbArms
        self.nbObs = np.zeros((self.nbRepetitions, self.nbArms))
        self.cumReward = np.zeros((self.nbRepetitions, len(tsav)))
        self.cumProfit = np.zeros((self.nbRepetitions, len(tsav)))
        self.thresholds = env.thresholds

        # progress = ProgressBar()
        for k in range(nbRepetitions): # progress(range(nbRepetitions)):
            #if nbRepetitions < 10 or k % (nbRepetitions/10)==0:
            #    print k
            result = env.play(pol, horizon)
            self.nbObs[k,:] = result.getNbObs()
            self.cumReward[k,:] = np.cumsum(result.cumRewardPerRound)[tsav]
            self.cumProfit[k,:] = np.cumsum(result.cumProfitPerRound)[tsav]
        # progress.finish()

    def meanReward(self):
        return np.sum(self.cumReward[:,-1])/self.nbRepetitions

    def meanProfit(self):
        return np.sum(self.cumProfit[:,-1])/self.nbRepetitions

    def meanNbDraws(self):
        return np.mean(self.nbObs ,0)

    def meanRegret(self):
        optimal_arms = [arm for arm in range(self.nbArms) if self.env.arms[arm].expectation > self.thresholds[arm]]
        expectedProfitPerRound = [(self.env.arms[arm].expectation - self.thresholds[arm])*self.env.arms[arm].lambda_poisson for arm in optimal_arms]
        #return (1+self.tsav)*np.sum([self.env.arms[arm].expectation for arm in optimal_arms]) - np.mean(self.cumProfit, 0)
        return (1+self.tsav)*np.sum(expectedProfitPerRound) - np.mean(self.cumProfit, 0)
