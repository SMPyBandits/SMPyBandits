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

def lai_and_robbins_lb(X, delta, sigma = 1):
  return 2*sigma**2*np.log(X)/delta

def fig_asymp(data, delta,  name='fig_asy.pdf', freq=1000):
  # --------------  PLOT  --------------
  fig, ax = plt.subplots(figsize=(12, 10))
  for i, policy in enumerate(data):
    T=data[policy]["mean"].shape[0]
    X = np.array([np.int(np.ceil(1.05**n)) for n in range(int(np.log(T)/np.log(1.05)))]+ [T-1])
    ax.semilogx(X, data[policy]["mean"][X], label=policy, linewidth=3)
    color = ax.get_lines()[-1].get_c()
    if "uppq" in data[policy]:
      ax.semilogx(X, data[policy]["uppq"][X], label=None, linestyle='--', color=color,
              linewidth=1)
      ax.semilogx(X, data[policy]["lowq"][X], label=None, linestyle='--', color=color,
              linewidth=1)
      plt.fill_between(X, data[policy]["uppq"][X], data[policy]["lowq"][X], alpha=.05,
                       color=color)
  ax.semilogx(X, lai_and_robbins_lb(X,delta), label="Lai and Robbins' lower bound", linewidth=5, color = 'k')
  max_value = np.max([np.max(data[key]['uppq'] if 'uppq' in data[key] else data[key]['mean'])for key in data])
  plt.ylim(0, 1.2 * max_value)
  plt.legend(prop={'variant': 'small-caps'}, loc = 2, edgecolor = 'k')
  plt.xlabel('Round ($t$)')
  plt.ylabel('Average regret $R_t$')
  plt.title('$\Delta = {:.3g}$'.format(delta), y=1.04)
  ax.grid(False)
  ax.xaxis.set_label_coords(0.5, -0.08)
  ax.yaxis.set_label_coords(-0.09, 0.5)
  ax.grid(False)
  # -------------- SAVE --------------
  plt.savefig(name, rasterized=True, dpi=1200)



# if __name__ == "__main__":
#   mus = [0.01, 1]
#   data = {}
#   for mu in mus :
#     data[mu] = {}
#   policies = [
#    [MOSSAnytime, {'alpha':3}], #0
#    [UCB, {}], #6
#    #  [EFF_RAWUCB, {'alpha': 1.4, 'm': 1.01}],  # 1
#    #  [EFF_RAWUCB_pp2, {'alpha': 1.4, 'm': 1.01}],  # 10
#    # # [EFF_RAWUCB_pp, {'beta': 3.5, 'm': 1.01}],  # 8
#   # [EFF_RAWUCB_pp, {'alpha': 1.3, 'm': 1.01}],  # 9
#   # [EFF_RAWUCB_pp, {'alpha': 1.4, 'm': 1.01}],  # 10
#   # [EFF_RAWUCB_pp, {'alpha': 1.6, 'm': 1.01}],  # 11
#   # [EFF_RAWUCB_pp, {'alpha': 1.7, 'm': 1.01}],  # 12
#   [EFF_RAWUCB_pp, {'alpha': 1.8, 'm': 1.01}],  # 11
#   [EFF_RAWUCB_pp, {'alpha': 1.9, 'm': 1.01}],  # 12
#   [EFF_RAWUCB_pp, {'alpha': 2.0, 'm': 1.01}],  # 12
#   [EFF_RAWUCB_pp, {'alpha': 2.1, 'm': 1.01}],  # 12
#   ]
#   for policy in policies:
#     quantile = True
#     print(str(policy[0](nbArms=2, **policy[1])))
#     policy_name = str(policy[0](nbArms=2, **policy[1]))
#     policy_name_nospace = policy_name.replace(' ', '_')
#     policy_data = [
#       np.load(os.path.join('./data', file)) for file in os.listdir('./data') if
#       file.startswith("REGRET_" + policy_name_nospace)
#     ]
#     if not policy_data:
#       print('no data')
#       continue
#     print(len(policy_data), policy_data)
#     policy_data_array = np.concatenate(policy_data, axis=1)
#     print(len(policy_data), policy_data_array.shape)
#     del policy_data
#     for i, mu in enumerate(mus):
#       if quantile :
#         data[mu][policy_name] = {
#           "mean": policy_data_array[i,:,:].mean(axis=0),
#           "uppq": np.quantile(policy_data_array[i,:,:], 0.9, axis=0),
#           "lowq": np.quantile(policy_data_array[i,:,:], 0.1, axis=0)
#         }
#       else:
#         data[mu][policy_name] = {
#           "mean": policy_data_array[i,:,:].mean(axis=0),
#         }
#     del policy_data_array
#   for mu in data:
#     fig_asymp(data[mu],mu, name='try%s.pdf'%mu)

# tmp = [38.63, 25.92, 21.86, 19.73, 18.45, 17.34, 16.57, 15.99, 15.27, 13.81, 14.05, 14.54, 13.13, 13.60, 11.58, 11.23, 11.05, 12.01, 11.44, 12.29, 11.88, 9.68, 11.10, 9.18, 10.27, 10.74, 9.47, 10.68, 10.49]
# fig, ax = plt.subplots(figsize=(12, 10))
# ax.plot(range(1, 1+len(tmp)), tmp, marker = 'o', color = 'k')
# plt.ylim(0,40)
# plt.xlabel('Number of connections')
# plt.ylabel('Churn rate (%)')
# plt.savefig('churn.pdf')

X= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Exercises = [1109062, 1018218, 1252420, 1123453, 1301960, 4813583, 456254, 514661, 440526, 584011, 839030,739434]
Users = [12738, 11862, 13834, 12557, 14855, 27017, 4679, 4801, 6955, 8434, 10377, 9875]

fig, ax1 = plt.subplots(figsize=(12, 10))

ax2 = ax1.twinx()
ax1.plot(X, Exercises)
color1 = ax1.get_lines()[-1].get_c()
ax1.set_ylim(0,5000000)
ax2.plot(X, Users, linewidth=0)
ax2.plot(X, Users )
color2 = ax2.get_lines()[-1].get_c()
ax2.set_ylim(0,30000)

ax1.set_xlabel('Month')
ax1.set_ylabel('Exercises', color= color1)
ax2.set_ylabel('Students', color = color2)
plt.tight_layout()
plt.savefig('student-exercice.pdf')