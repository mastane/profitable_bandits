# -*- coding: utf-8 -*-
'''Demonstration file for the pyBandits package'''

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.6 $"


from environment.MAB import MAB
from arm.Bernoulli import Bernoulli
from arm.Poisson import Poisson
from arm.Exponential import Exponential
from policy.UCB import UCB
from numpy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from policy.UCBV import UCBV
from policy.klUCB import klUCB
from policy.klUCBp import klUCBp
from policy.KLempUCB import KLempUCB
from policy.Thompson import Thompson
from policy.BayesUCB import BayesUCB
from Evaluation import *
from kullback import *
from posterior.Beta import Beta
from posterior.GammaForPoisson import GammaForPoisson
from posterior.GammaForExp import GammaForExp
import seaborn as sns
import sys
#from multiprocessing import Pool
from multiprocess import Pool
from run import f_run

current_palette = sns.color_palette()
sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc("lines", linewidth=1, markersize=10)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \boldmath"]
#colors = current_palette[0:4]+current_palette[5:6]
markers = ['o', '^', 's', '*', 'P', 'X', 'D', 'v', '.']
colors = ['red', 'blue', 'green', 'magenta', 'cyan', 'grey', 'chocolate', 'orange', 'yellow']
graphic = 'yes'
scenario = 2
nbRep = 100
nbRepPerRun = 10
horizon = 100
num_processes = 4#10  # parallelization
seed0 = 94  # offset for the seed

if scenario == 0:
    # First scenario (default): Bernoulli experiment
    lambdas = range(3, 8)
    arms_bernoulli = [Bernoulli(p, lambdas[i]) for i, p in enumerate([0.1, 0.3, 0.5, 0.5, 0.7])]
    K = len(arms_bernoulli)
    #thresholds = [0.999, 0.001, 0.999, 0.001, 0.999]
    thresholds = [0.2, 0.2, 0.4, 0.6, 0.8]
    env = MAB(arms_bernoulli, thresholds)
    policies = [klUCB(K, thresholds), BayesUCB(K, Beta, thresholds), Thompson(K, Beta, thresholds), klUCBp(K, thresholds), KLempUCB(K, thresholds), UCB(K, thresholds), UCBV(K, thresholds)]
    policy_name = ['kl-UCB-4P', 'Bayes-UCB-4P', 'TS-4P', 'kl-UCB+-4P', 'KL-Emp-UCB-4P', 'kl-Gaussian-UCB-4P', 'UCB-V-4P']
    #policies = [klUCB(K, thresholds), BayesUCB(K, Beta, thresholds), Thompson(K, Beta, thresholds), klUCBp(K, thresholds), KLempUCB(K, thresholds)]
    #policy_name = ['kl-UCB-4P', 'Bayes-UCB-4P', 'TS-4P', 'kl-UCB+-4P', 'KL-Emp-UCB-4P']
elif scenario == 1:
    # Second scenario: Truncated Poissson distrubtions
    trunc = 100
    lambdas = range(3, 8)
    arms_poisson = [Poisson(a, trunc, lambdas[a-1]) for a in range(1,6)]
    K = len(arms_poisson)
    delta = [1, -1, 1, -1, 1]
    thresholds = np.array([arms_poisson[i].expectation + delta[i] for i in range(K)])
    env = MAB(arms_poisson, thresholds)
    policies = [klUCB(K, thresholds, klucb=klucbPoisson), BayesUCB(K, GammaForPoisson, thresholds), Thompson(K, GammaForPoisson, thresholds), klUCBp(K, thresholds, klucb=klucbPoisson), KLempUCB(K, thresholds, maxReward=trunc), klUCB(K, thresholds/trunc, amplitude=trunc), klUCBp(K, thresholds/trunc, amplitude=trunc), UCBV(K, thresholds, trunc)]
    policy_name = ['kl-UCB-4P', 'Bayes-UCB-4P', 'TS-4P', 'kl-UCB+-4P', 'KL-Emp-UCB-4P', 'kl-Bernoulli-UCB-4P', 'kl-Bernoulli-UCB+-4P', 'UCB-V-4P']
    #policies = [klUCB(K, thresholds, klucb=klucbPoisson), BayesUCB(K, GammaForPoisson, thresholds), Thompson(K, GammaForPoisson, thresholds), klUCBp(K, thresholds, klucb=klucbPoisson), KLempUCB(K, thresholds, maxReward=trunc), klUCB(K, thresholds/trunc, amplitude=trunc), UCB(K, thresholds/trunc, amplitude=trunc), klUCBp(K, thresholds/trunc, amplitude=trunc), UCBV(K, thresholds, trunc)]
    #policy_name = ['kl-UCB-4P', 'Bayes-UCB-4P', 'TS-4P', 'kl-UCB+-4P', 'KL-Emp-UCB-4P', 'kl-Bernoulli-UCB-4P', 'kl-Gaussian-UCB-4P', 'kl-Bernoulli-UCB+-4P', 'UCB-V-4P']
else:
    # Third scenario: Truncated exponential distributions
    trunc = 100
    lambdas = range(3, 8)
    arms_exp = [Exponential(1./p, trunc, lambdas[p-1]) for p in range(1, 6)]
    K = len(arms_exp)
    delta = [1, -1, 1, -1, 1]
    thresholds = np.array([arms_exp[i].expectation + delta[i] for i in range(5)])
    env = MAB(arms_exp, thresholds)
    policies = [klUCB(K, thresholds, klucb=klucbExp), BayesUCB(K, GammaForExp, thresholds), Thompson(K, GammaForExp, thresholds), klUCBp(K, thresholds, klucb=klucbExp), KLempUCB(K, thresholds, maxReward=trunc), klUCB(K, thresholds/trunc, amplitude=trunc), klUCBp(K, thresholds/trunc, amplitude=trunc), UCBV(K, thresholds, trunc)]
    policy_name = ['kl-UCB-4P', 'Bayes-UCB-4P', 'TS-4P', 'kl-UCB+-4P', 'KL-Emp-UCB-4P', 'kl-Bernoulli-UCB-4P', 'kl-Bernoulli-UCB+-4P', 'UCB-V-4P']
    #policies = [klUCB(K, thresholds, klucb=klucbExp), BayesUCB(K, GammaForExp, thresholds), Thompson(K, GammaForExp, thresholds), klUCBp(K, thresholds, klucb=klucbExp), KLempUCB(K, thresholds, maxReward=trunc), klUCB(K, thresholds/trunc, amplitude=trunc), UCB(K, thresholds/trunc, amplitude=trunc), klUCBp(K, thresholds/trunc, amplitude=trunc), UCBV(K, thresholds, trunc)]
    #policy_name = ['kl-UCB-4P', 'Bayes-UCB-4P', 'TS-4P', 'kl-UCB+-4P', 'KL-Emp-UCB-4P', 'kl-Bernoulli-UCB-4P', 'kl-Gaussian-UCB-4P', 'kl-Bernoulli-UCB+-4P', 'UCB-V-4P']

expectations = [a.expectation for a in env.arms]
for i in range(K):
    print 'arm', i+1, ': expectation =', expectations[i], 'and threshold =', thresholds[i]

tsav = int_(linspace(0,horizon-1,200))

# parallelizing
pool = Pool(processes=num_processes)
args = [env, policies, policy_name, tsav, nbRep, nbRepPerRun, horizon, seed0]
all_regrets = pool.map(f_run, [[i]+args for i in range(int(nbRep/nbRepPerRun))])

# mean regret
mean_regret = np.mean(all_regrets, axis=0)
np.save('data/mean_regret_scenar{}_horiz{}_rep{}_seed{}_markers'.format(scenario, horizon, nbRep, seed0), mean_regret)
mean_regret = np.load('data/mean_regret_scenar{}_horiz{}_rep{}_seed{}_markers.npy'.format(scenario, horizon, nbRep, seed0))

if graphic == 'yes':
    plt.figure(1)
    for i, policy in enumerate(policies):
        plt.semilogx(1+tsav, mean_regret[i, :], linestyle='dashed', color=colors[i], marker=markers[i], fillstyle='none', markeredgecolor=colors[i], markeredgewidth=1, markevery=0.2, label=r"\textsc{%s}"%policy_name[i])
        plt.xlabel('Time (log scale)', fontsize=20)
        plt.ylabel('Regret', fontsize=20)
        #legend([policy.__class__.__name__ for policy in policies], loc=2)
        #plt.legend([pol for pol in policy_name], loc=2, fontsize=20).draw_frame(True)
        plt.legend(loc=2, fontsize=12).draw_frame(True)
        #plt.title('Average regret for various policies', fontsize=20)
    plt.savefig('fig/scenar{}_horiz{}_rep{}_seed{}_markers.pdf'.format(scenario, horizon, nbRep, seed0), bbox_inches='tight')
    #_delta0.1_bestPolicies
    #plt.show()
