"""
author: Julien SEZNEC
Plot utility to reproduce Figure 2 of [Seznec et al.,  2019a]
Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)
"""
from SMPyBandits.Policies import wSWA, FEWA, EFF_FEWA
import os
import numpy as np
from numpy import format_float_scientific
from matplotlib import pyplot as plt
plt.style.use('seaborn-colorblind')
plt.style.use('style.mplstyle')


def fig2A(data, name='fig2.pdf', ylim=2400, ylim2=500):
    # --------------  PLOT  --------------
    legend_size = 0.45
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, legend_size]})
    N_arms = 9
    ind = np.arange(N_arms)  # the x locations for the groups
    width = 0.7  # the width of the bars
    L = np.array([0.001 * np.sqrt(10) ** (i) for i in range(9)])
    for i, policy in enumerate(data):
        X = range(data[policy]["mean"].shape[0])
        ax1.plot(X, data[policy]["mean"], linewidth=3)
        color = ax1.get_lines()[-1].get_c()
        ax1.plot(X, data[policy]["uppq"], linestyle='--', color= color, linewidth=1)
        ax1.plot(X, data[policy]["lowq"], linestyle='--', color= color, linewidth=1)
        ax1.fill_between(X, data[policy]["uppq"], data[policy]["lowq"], alpha=.05, color= color)

        height = data[policy]["pull"][1:] * L
        x_pos = ind - width / 2 + (i + 2) * width / len(data)
        width_bar = width / len(data)
        ax2.bar(x_pos, height, width_bar, bottom=0, label=policy, color=color)
        for j in np.argwhere(height > ylim2):
            ax2.text(x_pos[j], ylim2 * 1.01, int(height[j]), ha='center', va='bottom', rotation='vertical',
                     fontsize=18, color=color)
    ax1.set_ylim(0, ylim)
    ax1.set_xlabel('Round ($t$)', fontsize=30)
    ax1.set_ylabel('Average regret $R_t$', fontsize=30)
    ax1.xaxis.set_label_coords(0.5, -0.08)
    ax2.set_xticks(ind + width / len(data))
    xticks = [format_float_scientific(mu, exp_digits=1, precision=0) for mu in L]
    xticks = [float(xtick) if j % 2 == 0 else '' for j, xtick in enumerate(xticks)]
    ax2.set_ylim(0, ylim2)
    ax2.set_xticklabels(xticks)
    ax2.set_ylabel('Average regret per arm $R_T^i$ at $T = 25000$', fontsize=30)
    ax2.set_xlabel("Arm's $\Delta_i$", fontsize=30)
    ax2.xaxis.set_label_coords(0.5, -0.08)
    ax2.yaxis.set_label_coords(-0.08, 0.5)
    handles, labels = ax2.get_legend_handles_labels()
    pos = ax3.get_position()
    fig.legend(handles, labels, loc=[0.9 * pos.x0 + 0.1 * pos.x1, (pos.y1 - pos.y0) / 2])
    ax3.grid(False)
    ax3.axis('off')
    # Hide axes ticks
    ax3.set_xticks([])
    ax3.set_yticks([])
        # -------------- SAVE --------------
    fig.set_size_inches(30, 10)
    fig.tight_layout()
    fig.savefig(name)

if __name__ == "__main__":
    policies = [
        [FEWA, {'alpha': .03, 'delta': 1}],
        [FEWA, {'alpha': .06, 'delta': 1}],
        [FEWA, {'alpha': .1, 'delta': 1}],
        [EFF_FEWA, {'alpha': .06, 'delta': 1, 'm': 2}],
        [wSWA, {'alpha': 0.002}],
        [wSWA, {'alpha': 0.02}],
        [wSWA, {'alpha': 0.2}]
    ]
    data = {}
    for policy in policies:
        policy_name = str(policy[0](nbArms=2, **policy[1]))
        policy_name_nospace = policy_name.replace(' ', '_')
        policy_data_regret = [
        np.load(os.path.join('./data', file))
        for file in os.listdir('./data') if
            file.startswith("REGRET_" + policy_name_nospace)
        ]
        policy_data_pull = [
        np.load(os.path.join('./data', file))
        for file in os.listdir('./data') if
            file.startswith("DIFFPULL_" + policy_name_nospace)
        ]
        if not policy_data_regret:
            continue
        regret_data_array = np.concatenate(policy_data_regret, axis=0)
        pull_data_array = np.concatenate(policy_data_pull, axis=0)
        data[policy_name] = {
            "mean": regret_data_array.mean(axis=0),
            "uppq": np.quantile(regret_data_array, 0.9, axis=0),
            "lowq": np.quantile(regret_data_array, 0.1, axis=0),
            "pull": pull_data_array.mean(axis=0)
        }
    fig2A(data)
