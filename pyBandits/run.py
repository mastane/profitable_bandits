from Evaluation import *
import numpy as np
import time

def f_run(args):
    begin = time.time()
    nb, env, policies, policy_name, tsav, nbRep, nbRepPerRun, horizon, seed0 = args
    # specify seed, otherwise same simulations on all cores
    np.random.seed(nb+seed0)
    regrets = [None]*len(policies)
    for i, policy in enumerate(policies):
        ev = Evaluation(env, policy, nbRepPerRun, horizon, tsav)
        #print policy_name[i]
        #print 'mean Reward', ev.meanReward()
        #print 'mean Profit', ev.meanProfit()
        #print 'mean number draws', ev.meanNbDraws()
        meanRegret = ev.meanRegret()
        regrets[i] = meanRegret
    print 'done:', nb+1, '/', nbRep/nbRepPerRun, 'in:', time.time()-begin, 'seconds'
    return regrets
