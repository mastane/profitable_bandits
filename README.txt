    py/maBandits: matlab and python packages for multi-armed bandits
    
      authors: Olivier Cappé, Aurélien Garivier, Emilie Kaufmann
     ............................................................



# About

This package contains a python and a matlab implementation of the most widely
used algorithms for multi-armed bandit problems. The purpose of this package
is to provide simple environments for comparison and numerical evaluation of
policies. Part of the code proposed here was used to produce the Figures
included in our bandit papers (referenced below).

The python code is provided with some C extensions that make it faster, but
configuration-dependent. Some (basic) compilation work is required to use
it. However, a plain python version is also included so that these extensions
are by no way necessary to run the experiments.

Each version contains a demonstration file that shows how to run experiments.


# The Environments

Currently, the following arm distributions are provided:

* Bernoulli distribution
* Poisson distribution (possibly truncated)
* Exponential distribution (possibly truncated)

# The Policies

The packages presently include the following policies:

* Gittin's Bayesian optimal strategy for binary rewards, see [1], in the matlab package only
* The classical UCB policy, see [2]
* The UCB-V policy of [3]
* The KL-UCB policy of [4], with variants tuned for Exponential and Poisson distributions
* The Clopper-Pearson policy for binary rewards, see [4], in the matlab package only
* The MOSS policy of [5], in the matlab package only
* The DMED policy (implemented for binary rewards only) of [6], in the matlab package only
* The Emipirical Likelihood UCB of [7]
* The Bayes-UCB policy of [8]
* The Thompson sampling policy, see [9]


# Difference between the python and matlab versions

As noted above, a few policies are available only in the matlab
implementation. This being said, the python version was developed more
recently and is better designed. For instance, arms with different
distributions can be used in the same environment. The handling of bounds on
the support of the arm distributions is also different: for more details see
the readme files in each subdirectory.


# References

[1] Bandit Processes and Dynamic Allocation Indices J. C. Gittins. Journal of
the Royal Statistical Society. Series B (Methodological) Vol. 41, No. 2. 1979
pp. 148–177

[2] Finite-time analysis of the multiarmed bandit problem Peter Auer, Nicolò
Cesa-Bianchi and Paul Fischer. Machine Learning 47 2002 pp.235-256

[3] Exploration-exploitation trade-off using variance estimates in multi-armed
bandits J.-Y. Audibert, R. Munos, Cs. Szepesvári.  Theoretical Computer
Science Volume 410 Issue 19 Apr. 2009 pp. 1876-1902

[4] The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond
A. Garivier, O. Cappé JMLR Workshop and Conference Proceedings Volume 19: COLT
2011 Conference on Learning Theory pp. 359-376 Jul. 2011

[5] Minimax Policies for Adversarial and Stochastic Bandits J-Y. Audibert and
S. Bubeck. Proceedings of the 22nd Annual Conference on Learning Theory 2009

[6] An asymptotically optimal policy for finite support models in the
multiarmed bandit problem J. Honda, A. Takemura. Machine Learning 85(3) 2011
pp. 361-391

[7] UCB policies based on Kullback-Leibler divergence O. Cappé, A. Garivier,
O-A. Maillard, R. Munos. in preparation

[8] On Bayesian Upper Confidence Bounds for Bandit Problems E. Kaufmann,
O. Cappé, A. Garivier.  Fifteenth International Conference on Artificial
Intelligence and Statistics (AISTAT) Apr. 2012

[9] Thompson Sampling: an optimal analysis E. Kaufmann, N. Korda et R.
Munos. preprint

[10] On Upper-Confidence Bound Policies for Non-stationary Bandit Problems
A. Garivier, E. Moulines. Algorithmic Learning Theory (ALT) pp. 174-188
Oct. 2011

[11] Optimism in Reinforcement Learning and Kullback-Leibler Divergence
S. Filippi, O. Cappé, A. Garivier. 48th Annual Allerton Conference on
Communication, Control, and Computing Sep. 2010

[12] Parametric Bandits: The Generalized Linear Case S. Filippi, O. Cappé,
A. Garivier, C. Szepesvari. Neural Information Processing Systems (NIPS)
Dec. 2010


 [1]:  http://www.jstor.org/stable/2985029
 [2]:  http://www.springerlink.com/content/l7v1647363415h1t/?MUD=MP
 [3]:  http://dl.acm.org/citation.cfm?id=1519712
 [4]:  http://jmlr.csail.mit.edu/proceedings/papers/v19/
 [5]:  http://www.cs.mcgill.ca/~colt2009/papers/022.pdf#page=1
 [6]:  http://www.springerlink.com/content/0885-6125/85/3/
 [7]:  http://perso.telecom-paristech.fr/~garivier/bandits/
 [8]:  http://jmlr.csail.mit.edu/proceedings/papers/v22/
 [9]:  http://perso.telecom-paristech.fr/~kaufmann/1205.4217v1.pdf
 [10]: http://www.springerlink.com/content/v06722603413/
 [11]: http://arxiv.org/abs/1004.5229
 [12]: http://books.nips.cc/papers/files/nips23/NIPS2010_0828.pdf
