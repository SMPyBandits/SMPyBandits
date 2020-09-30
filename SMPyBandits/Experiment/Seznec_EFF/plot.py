"""
author: Julien SEZNEC
Plot utility to reproduce Efficient algorithms experiment figure (thesis)
"""
from matplotlib import pyplot as plt
from SMPyBandits.Policies import wSWA, FEWA, EFF_FEWA, RAWUCB, EFF_RAWUCB
import os
import numpy as np

plt.style.use('seaborn-colorblind')  # not the prettiest but accessible
plt.style.use('style.mplstyle')
MARKERS = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']


def fig_eff(data,  name='fig_eff.pdf'):
  # --------------  PLOT  --------------
  fig, ax = plt.subplots(figsize=(12, 10))
  for i, policy in enumerate(data):
    X = range(data[policy]["mean"].shape[0])
    ax.plot(X, data[policy]["mean"], label=policy, linewidth=3, color= 'gray' if i==6 else None)
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
  plt.legend(prop={'variant': 'small-caps'}, edgecolor = 'k')
  plt.xlabel('Round ($t$)')
  plt.ylabel('Average regret $R_t$')
  ax.grid(False)
  ax.xaxis.set_label_coords(0.5, -0.08)
  ax.yaxis.set_label_coords(-0.09, 0.5)
  ax.grid(False)
  # -------------- SAVE --------------
  plt.savefig(name)



if __name__ == "__main__":
  policies = [
    [RAWUCB, {'alpha': 1.4}],  # 0
    [wSWA, {'alpha': 0.002}],  # 2
    [wSWA, {'alpha': 0.02}],  # 3
    [wSWA, {'alpha': 0.2}],  # 4
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.01}],  # 5
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.1}],  # 6
    #[EFF_RAWUCB, {'alpha': 1.4, 'm': 1.2}],  # 7
    #[EFF_RAWUCB, {'alpha': 1.4, 'm': 1.3}],  # 8
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.5}],  # 9
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.9}],  # 10
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 2}],  # 1
    [EFF_RAWUCB, {'alpha': 1.4, 'm': 2.1}],  # 11
    #[EFF_RAWUCB, {'alpha': 1.4, 'm': 3}],  # 12
    #[EFF_RAWUCB, {'alpha': 1.4, 'm': 10}],  # 13
  ]
  data = {}
  for policy in policies:
    quantile = True
    policy_name = str(policy[0](nbArms=2, **policy[1]))
    policy_name_nospace = policy_name.replace(' ', '_')
    policy_data = [
      np.load(os.path.join('./data', file)) for file in os.listdir('./data') if
      file.startswith("REGRET_" + policy_name_nospace)
    ]
    policy_data_time = [
      np.load(os.path.join('./data', file)) for file in os.listdir('./data') if
      file.startswith("TIME_" + policy_name_nospace)
    ]
    if not policy_data:
      continue
    policy_data_array = np.concatenate(policy_data, axis=1)[0,:,:]
    policy_data_time_array = np.concatenate(policy_data_time, axis=1)[0,:,:]
    print(len(policy_data), policy_data_array.shape)
    del policy_data
    if quantile :
      data[policy_name] = {
        "mean": policy_data_array.mean(axis=0),
        "uppq": np.quantile(policy_data_array, 0.9, axis=0),
        "lowq": np.quantile(policy_data_array, 0.1, axis=0),
        "time_mean": policy_data_time_array.mean(axis=0)
      }
    else:
      data[policy_name] = {
        "mean": policy_data_array.mean(axis=0),
      }
    del policy_data_array
  fig_eff(data, name='try.pdf')