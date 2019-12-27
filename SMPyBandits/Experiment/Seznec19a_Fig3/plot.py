"""
author: Julien SEZNEC
Plot utility to reproduce Figure 3 of [Seznec et al.,  2019a]
Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)
"""
from matplotlib import pyplot as plt
from SMPyBandits.Policies import FEWA, UCB
import os
import numpy as np

plt.style.use('seaborn-colorblind')
plt.style.use('style.mplstyle')



def fig3(data, delta , name='fig3A.pdf',  ylim=300):
    # --------------  PLOT  --------------
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, policy in enumerate(data):
        print(data[policy]["mean"])
        X = range(data[policy]["mean"].shape[0])
        ax.plot(X, data[policy]["mean"], label=policy, linewidth=3)
        color = ax.get_lines()[-1].get_c()
        ax.plot(X, data[policy]["uppq"], label=None, linestyle='--', color=color, linewidth=1)
        ax.plot(X, data[policy]["lowq"], label=None, linestyle='--', color=color, linewidth=1)
        plt.fill_between(X, data[policy]["uppq"], data[policy]["lowq"], alpha=.05, color=color)
    plt.xlim(0,5000)
    plt.ylim(0, ylim)
    plt.legend(prop={'variant': 'small-caps'})
    plt.xlabel('Round ($t$)')
    plt.ylabel('Average regret $R_t$')
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    plt.title('$\Delta = {:.3g}$'.format(delta), y=1.04)
    # -------------- SAVE --------------
    plt.savefig(name)


if __name__ == "__main__":
    for game in range(1,3):
        policies = [
            [FEWA, {'alpha': .01, 'delta': 1, 'subgaussian': 1}],
            [FEWA, {'alpha': .06, 'delta': 1, 'subgaussian': 1}],
            [FEWA, {'alpha': 0.25, 'delta': 1, 'subgaussian': 1}],
            [UCB, {}]
        ]
        data = {}
        for policy in policies:
            policy_name = str(policy[0](nbArms=2, **policy[1]))
            policy_name_nospace = policy_name.replace(' ', '_')
            policy_data = [
                np.load(os.path.join('./data', file)) for file in os.listdir('./data') if
                file.startswith("REGRET%s_"%game + policy_name_nospace)
            ]
            if not policy_data:
                continue
            policy_data_array = np.concatenate(policy_data, axis=0)
            print(len(policy_data), policy_data_array.shape)
            data[policy_name] = {
                "mean": policy_data_array.mean(axis=0),
                "uppq": np.quantile(policy_data_array, 0.9, axis=0),
                "lowq": np.quantile(policy_data_array, 0.1, axis=0)
            }

        fig3(data, delta=0.14 if game == 1 else 1, name='fig3%s.pdf'%game)
