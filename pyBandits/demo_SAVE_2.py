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
nbRep = 10000
nbRepPerRun = 100
horizon = 10000
num_processes = 10  # parallelization
seed0 = 94  # offset for the seed

policy_name = ['kl-UCB-4P', 'Bayes-UCB-4P', 'TS-4P', 'kl-UCB+-4P', 'KL-Emp-UCB-4P', 'kl-Bernoulli-UCB-4P', 'kl-Bernoulli-UCB+-4P', 'UCB-V-4P']

tsav = int_(linspace(0,horizon-1,200))

mean_regret = np.load('data/mean_regret_scenar{}_horiz{}_rep{}_seed{}_markers.npy'.format(scenario, horizon, nbRep, seed0))

if graphic == 'yes':
    plt.figure(1)
    for i in range(len(policy_name)):  #, policy in enumerate(policies):
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
