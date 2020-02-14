"""
author: Julien SEZNEC
Plot utility to reproduce Figure 1 of [Seznec et al.,  2019a]
Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)
"""
from matplotlib import pyplot as plt
from SMPyBandits.Policies import EFF_RAWUCB, EFF_FEWA, GaussianGLR_IndexPolicy, Exp3S, klUCBloglog_forGLR
import os
import numpy as np
import pandas as pd
from math import sqrt, log

plt.style.use('seaborn-colorblind') # not the prettiest but accessible
plt.style.use('style.mplstyle')
MARKERS = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']


def fig1(data, name='fig1.pdf'):
    # --------------  PLOT  --------------
    fig,ax  = plt.subplots(figsize=(12, 10))
    for i, policy in enumerate(data):
        X = range(data[policy]["mean"].shape[0])
        ax.plot(X, data[policy]["mean"], label=policy, linewidth=3)
        color = ax.get_lines()[-1].get_c()
        ax.plot(X, data[policy]["uppq"], label=None, linestyle='--', color=color,
                linewidth=1)
        ax.plot(X, data[policy]["lowq"], label=None, linestyle='--', color=color,
                linewidth=1)
        plt.fill_between(X, data[policy]["uppq"], data[policy]["lowq"], alpha=.05,
                         color=color)
    max_value = np.max([np.max(data[key]['uppq']) for key in data])
    plt.ylim(0, 1.1 * max_value)
    plt.xlim(0, data[policy]["mean"].shape[0])
    plt.legend(prop={'variant': 'small-caps'})
    plt.xlabel('Round ($t$)')
    plt.ylabel('Average regret $R_t$')
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    # -------------- SAVE --------------
    plt.savefig(name)


if __name__ == "__main__":
    DAY = 2
    data_file = 'data/Reward/reward_data_day_%s.csv' % (DAY)
    DRAWS = 10
    df = pd.read_csv(data_file, index_col=0).transpose().reset_index(drop=True)
    K = len(df.columns)
    HORIZON = T = len(df)  # Horizon T
    SIGMA = (0.03 * 0.97 * DRAWS) ** .5
    policies = [
        [EFF_RAWUCB, {'alpha': 1.4, 'subgaussian': SIGMA, 'delta': K, 'm': 1.1}],
        [EFF_RAWUCB, {'alpha': 1.4, 'subgaussian': SIGMA, 'delta': K, 'm': 2}],
        [EFF_RAWUCB, {'alpha': 4, 'subgaussian': SIGMA, 'delta': K, 'm': 1.1}],
        [EFF_FEWA, {'alpha': .06, 'subgaussian': SIGMA, 'delta': K, 'm': 1.1}],
        [EFF_FEWA, {'alpha': 4, 'subgaussian': SIGMA, 'delta': K, 'm': 1.1}],
        [GaussianGLR_IndexPolicy,
         {'policy': klUCBloglog_forGLR, 'delta': sqrt(1 / T), 'alpha0': 0, 'per_arm_restart': True,
          'sig2': SIGMA ** 2}],
        [Exp3S, {'alpha': 1 / T, 'gamma': min(1, sqrt(K * log(K * T) / T))}],
    ]
    data = {}
    for policy in policies:
        policy_name = str(policy[0](nbArms=2, **policy[1]))
        policy_name_nospace = policy_name.replace(' ', '_')
        policy_data = [
            np.load(os.path.join('./data', file)) for file in os.listdir('./data') if
            file.startswith("DAY_%s_REGRET_%s_"%(DAY, policy_name_nospace))
        ]
        if not policy_data:
            continue
        policy_data_array = np.concatenate(policy_data, axis=0)
        print(len(policy_data), policy_data_array.shape)
        print(data)
        data[policy_name] = {
            "mean": policy_data_array.mean(axis=0),
            "uppq": np.quantile(policy_data_array, 0.9, axis=0),
            "lowq": np.quantile(policy_data_array, 0.1, axis=0)
        }
    fig1(data)

