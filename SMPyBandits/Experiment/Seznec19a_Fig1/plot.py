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
from SMPyBandits.Policies import wSWA, FEWA, EFF_FEWA, RAWUCB, GaussianGLR_IndexPolicy, Exp3S, klUCBloglog_forGLR, EFF_RAWUCB, EFF_RAWUCB_pp
import os
import numpy as np

plt.style.use('seaborn-colorblind')  # not the prettiest but accessible
plt.style.use('style.mplstyle')
MARKERS = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']


def fig1A(data, L, save=True, name="fig1A.pdf", ylim=400):
  # --------------  PLOT  --------------
  fig, ax = plt.subplots(figsize=(12, 10))
  for i, policy in enumerate(data):
    ax.semilogx(L, data[policy]["mean"][:, -1], label=policy,
                marker=MARKERS[i % len(MARKERS)], linewidth=3, markersize=6,
            color='gray' if i == 6 else None)
    color = ax.get_lines()[-1].get_c()
    ax.semilogx(L, data[policy]["uppq"][:, -1], label=None, linestyle='--', color=color,
                linewidth=1)
    ax.semilogx(L, data[policy]["lowq"][:, -1], label=None, linestyle='--', color=color,
                linewidth=1)
    plt.fill_between(L, data[policy]["uppq"][:, -1], data[policy]["lowq"][:, -1], alpha=.05, color=color)
  plt.ylim(0, ylim)
  plt.legend(prop={'variant': 'small-caps'}, edgecolor = 'k')
  plt.xlabel('$L$')
  plt.ylabel('Average regret at $T = 10^4$')
  ax.xaxis.set_label_coords(0.5, -0.08)
  ax.yaxis.set_label_coords(-0.09, 0.5)
  ax.grid(False)
  # -------------- SAVE --------------
  if save:
    plt.savefig(name)


def fig1BC(data, mus, mu_index=11, name='fig1B.pdf', ylim=300, freq=50):
  # --------------  PLOT  --------------
  L = mus[mu_index]
  fig, ax = plt.subplots(figsize=(12, 10))
  for i, policy in enumerate(data):
    X = sorted(list(range(0,data[policy]["mean"].shape[1], freq)) + [2500, 2501, 2502, 2503, 2504,2505, 2506, 2538, 2765])
    ax.plot(X, data[policy]["mean"][mu_index, X], label=policy, linewidth=3,
            color='gray' if i == 6 else None)
    color = ax.get_lines()[-1].get_c()
    ax.plot(X, data[policy]["uppq"][mu_index, X], label=None, linestyle='--', color=color,
            linewidth=1)
    ax.plot(X, data[policy]["lowq"][mu_index, X], label=None, linestyle='--', color=color,
            linewidth=1)
    plt.fill_between(X, data[policy]["uppq"][mu_index, X], data[policy]["lowq"][mu_index, X],
                     alpha=.05,
                     color=color)
  plt.ylim(0, ylim)
  plt.legend(prop={'variant': 'small-caps'}, edgecolor = 'k')
  plt.xlabel('Round ($t$)')
  plt.ylabel('Average regret $R_t$')
  ax.xaxis.set_label_coords(0.5, -0.08)
  ax.yaxis.set_label_coords(-0.09, 0.5)
  ax.grid(False)
  plt.title('$L = {:.3g}$'.format(L), y=1.04)
  # -------------- SAVE --------------
  plt.savefig(name)

def plot_all_fig(policies, ylimA=900, ylimB=350, ylimC=350, name=''):
  L = [0.02 * 1.25 ** (i) for i in range(30)]
  data = {}
  for policy in policies:
    policy_name = str(policy[0](nbArms=2, **policy[1]))
    policy_name_nospace = policy_name.replace(' ', '_')
    policy_data = [
      np.load(os.path.join('./data', file)) for file in os.listdir('./data') if
      file.startswith("REGRET_" + policy_name_nospace)
    ]
    if not policy_data:
      continue
    policy_data_array = np.concatenate(policy_data, axis=1)
    print(len(policy_data), policy_data_array.shape)
    data[policy_name] = {
      "mean": policy_data_array.mean(axis=1),
      "uppq": np.quantile(policy_data_array, 0.9, axis=1),
      "lowq": np.quantile(policy_data_array, 0.1, axis=1)
    }
    fig1A(data, L, name='fig1A_%s.pdf'%name, ylim=ylimA)
    fig1BC(data, L, mu_index=11, name='fig1B_%s.pdf'%name, ylim=ylimB)
    fig1BC(data, L, mu_index=24, name='fig1C_%s.pdf'%name, ylim=ylimC)


if __name__ == "__main__":
  HORIZON = T = 10000  # Horizon T
  sigma = 1  # Gaussian noise std
  K = 2
  # policies = [
  #   [wSWA, {'alpha': 0.002}],  # 4
  #   [wSWA, {'alpha': 0.02}],  # 5
  #   [wSWA, {'alpha': 0.2}],  # 6
  # ]
  # plot_all_fig(policies, name="SWA")
  # policies = [
  #   [RAWUCB, {'alpha': 1.4}],  # 7
  #   [RAWUCB, {'alpha': 4}],  # 8
  #   [FEWA, {'alpha': .06, 'delta': 1}],  # 1
  #   [FEWA, {'alpha': 4}],  # 9
  #   [wSWA, {'alpha': 0.002}],  # 4
  #   [wSWA, {'alpha': 0.02}],  # 5
  #   [wSWA, {'alpha': 0.2}],  # 6
  # ]
  # plot_all_fig(policies, name="main")
  policies = [
    [RAWUCB, {'alpha': 1.4}],  # 7
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.1}],  # 12
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 2}],  # 13
  ]
  plot_all_fig(policies, name="eff", ylimA=250, ylimB=200, ylimC=30)

  # policies = [
  #   [RAWUCB, {'alpha': 1.4}],  # 7
  #   [EFF_RAWUCB_pp, {'alpha': 1.4, 'm': 1.01}],  # 14
  # ]
  # plot_all_fig(policies, name="pp", ylimA=300, ylimB=300, ylimC=350)
  # policies =[
  # [GaussianGLR_IndexPolicy, {'policy': klUCBloglog_forGLR, 'delta': np.sqrt(1 / T), 'alpha0': 0,
  #                         'per_arm_restart': True, 'sig2': sigma ** 2, 'use_localization': False}],  # 10
  # [Exp3S, {'alpha': 1 / T, 'gamma': min(1, np.sqrt(K * np.log(K * T) / T))}],  # 11
  # [EFF_RAWUCB_pp2, {'alpha': 1.4, 'm': 1.01}],  # 12
  #   ]


