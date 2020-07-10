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
  # -------------- SAVE --------------
  plt.savefig(name)


def delay_data(T, m=2):
  res= []
  sigma = 1
  policy = EFF_FEWA(1, subgaussian=sigma, alpha=0.06, delay=True, m=m)
  for t in range(T):
    policy.getReward(0, 1)
    res.append(policy.delay.copy())
  res[-1] = res[-1][~np.isnan(res[-1])]
  final_length = len(res[-1])
  for i, r in enumerate(res):
    res[i] = np.append(r[:final_length], np.nan * np.ones(final_length - len(r[:final_length])))
  return np.arange(1, T+2 ,1), policy.windows[:final_length], np.array(res) / policy.windows[:final_length]

def delay_plot(x,w,Z,T,m):
  fig, ax = plt.subplots(figsize=(12, 7))
  ax.set_xticks(np.arange(len(x)), minor=True)
  ax.set_yticks(np.arange(len(w)), minor=True)
  ax.set_xlabel("$N_i$")
  ax.set_ylabel("$j$", rotation=0, labelpad = 18)
  title = "$m=%s$"%m
  ax.set_title(title, pad = 12)
  y = np.arange(0, len(w)+1,1)
  X,Y = np.meshgrid(x,y)
  c = ax.pcolormesh(X-0.5, Y-0.5, Z.transpose(), cmap='RdYlGn_r', vmin= 0, vmax=1, rasterized =True)
  fig.colorbar(c, ax=ax)
  plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
  name = 'T=%s_m=%s' % (T, m)
  plt.savefig(name + '.pdf', dpi=1200)



if __name__ == "__main__":
  M = [10]
  for m in M:
    T = 5000
    x,w,z = delay_data(T, m)
    delay_plot(x, w, z, T, m )

  # policies = [
  #   [RAWUCB, {'alpha': 1.4}],  # 0
  #   [EFF_RAWUCB, {'alpha': 1.4, 'm': 2}],  # 1
  #   [wSWA, {'alpha': 0.002}],  # 2
  #   [wSWA, {'alpha': 0.02}],  # 3
  #   [wSWA, {'alpha': 0.2}],  # 4
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 1.01}],  # 5
  #   [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.1}],  # 6
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 1.2}],  # 7
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 1.3}],  # 8
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 1.5}],  # 9
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 1.9}],  # 10
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 2.1}],  # 11
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 3}],  # 12
  #   #[EFF_RAWUCB, {'alpha': 1.4, 'm': 10}],  # 13
  # ]
  #
  #
  # data = {}
  # for policy in policies:
  #   quantile = False
  #   policy_name = str(policy[0](nbArms=2, **policy[1]))
  #   policy_name_nospace = policy_name.replace(' ', '_')
  #   policy_data = [
  #     np.load(os.path.join('./data', file)) for file in os.listdir('./data') if
  #     file.startswith("REGRET_" + policy_name_nospace)
  #   ]
  #   if not policy_data:
  #     continue
  #   policy_data_array = np.concatenate(policy_data, axis=0)
  #   print(len(policy_data), policy_data_array.shape)
  #   del policy_data
  #   if quantile :
  #     data[policy_name] = {
  #       "mean": policy_data_array.mean(axis=0),
  #       "uppq": np.quantile(policy_data_array, 0.9, axis=0),
  #       "lowq": np.quantile(policy_data_array, 0.1, axis=0)
  #     }
  #   else:
  #     data[policy_name] = {
  #       "mean": policy_data_array.mean(axis=0),
  #     }
  #   del policy_data_array

