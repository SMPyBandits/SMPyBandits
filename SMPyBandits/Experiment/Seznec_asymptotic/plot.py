"""
author: Julien SEZNEC
Plot utility to reproduce RAWUCB++ algorithms experiment figure (thesis)
"""
from matplotlib import pyplot as plt
from SMPyBandits.Policies import MOSSAnytime, EFF_RAWUCB, EFF_RAWUCB_pp, UCB
import os
import numpy as np

plt.style.use('seaborn-colorblind')  # not the prettiest but accessible
plt.style.use('style.mplstyle')
MARKERS = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']


def fig_asymp(data,  name='fig_asy.pdf'):
  # --------------  PLOT  --------------
  fig, ax = plt.subplots(figsize=(12, 10))
  for i, policy in enumerate(data):
    X = range(data[policy]["mean"].shape[0])
    ax.plot(X, data[policy]["mean"], label=policy, linewidth=3)
    color = ax.get_lines()[-1].get_c()
    if "uppq" in data[policy]:
      ax.plot(X, data[policy]["uppq"], label=None, linestyle='--', color=color,
              linewidth=1)
      ax.plot(X, data[policy]["lowq"], label=None, linestyle='--', color=color,
              linewidth=1)
      plt.fill_between(X, data[policy]["uppq"], data[policy]["lowq"][:], alpha=.05,
                       color=color)
  max_value = np.max([np.max(data[key]['uppq'] if 'uppq' in data[key] else data[key]['mean'])for key in data])
  plt.ylim(0, 1.2 * max_value)
  plt.legend(prop={'variant': 'small-caps'})
  plt.xlabel('Round ($t$)')
  plt.ylabel('Average regret $R_t$')
  ax.grid(False)
  ax.xaxis.set_label_coords(0.5, -0.08)
  ax.yaxis.set_label_coords(-0.09, 0.5)
  ax.grid(False)
  # -------------- SAVE --------------
  plt.savefig(name)



if __name__ == "__main__":
  mus = [0.01, 1]
  data = {}
  for mu in mus :
    data[mu] = {}
  policies = [
    [MOSSAnytime, {'alpha': 3}],  # 0
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.01}],  # 1
    [EFF_RAWUCB_pp, {'beta': 0, 'm': 1.01}],  # 2
    [EFF_RAWUCB_pp, {'beta': 1, 'm': 1.01}],  # 3
    [EFF_RAWUCB_pp, {'beta': 2, 'm': 1.01}],  # 4
    [EFF_RAWUCB_pp, {'beta': 3, 'm': 1.01}],  # 5
    [UCB, {'beta': 3, 'm': 1.01}],  # 5

  ]
  for policy in policies:
    quantile = False
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
    del policy_data
    for i, mu in enumerate(mus):
      if quantile :
        data[mu][policy_name] = {
          "mean": policy_data_array[i,:,:].mean(axis=0),
          "uppq": np.quantile(policy_data_array[i,:,:], 0.9, axis=0),
          "lowq": np.quantile(policy_data_array[i,:,:], 0.1, axis=0)
        }
      else:
        data[mu][policy_name] = {
          "mean": policy_data_array[i,:,:].mean(axis=0),
        }
    del policy_data_array
    # policy_data_time = [
    #   np.load(os.path.join('./data', file))
    #   for file in os.listdir('./data') if
    #   file.startswith("TIME_" + policy_name_nospace)
    # ]
    # time_array = np.concatenate(policy_data, axis=1)[0, :, :]
    # data[policy_name]["time_mean"] = time_array.mean(axis=0)
  for mu in data:
    fig_asymp(data[mu], name='try%s.pdf'%mu)