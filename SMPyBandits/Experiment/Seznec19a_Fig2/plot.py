from matplotlib import pyplot as plt
plt.style.use('style.mplstyle')
from SMPyBandits.Policies import wSWA, FEWA, EFF_FEWA
import os
import numpy as np

def fig1A(data ,L, save = True ):
    # --------------  PLOT  --------------
    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot()
    for policy in data:
        ax.semilogx(L, data[policy]["mean"][:,-1], label=policy)
        plt.fill_between(L, data[policy]["uppq"][:,-1], data[policy]["lowq"][:,-1], alpha=.1)

    plt.ylim(0,400)
    plt.legend(prop={'variant': 'small-caps'})
    plt.xlabel('$L$')
    plt.ylabel('Average regret at $T = 10^4$')
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    #-------------- SAVE --------------
    if save:
        plt.savefig("fig1A.pdf")


def fig1BC(regret, policies, mus, mu_index = 11, name= 'fig1B.pdf', ylim=300):
    #--------------  PLOT  --------------
    L = mus[mu_index]
    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot()
    for i in range(len(policies)):
        ax.plot(range(regret.shape[2]),regret[mu_index,i,:], label=policies[i][0](2, **policies[i][1]))
    plt.ylim(0,ylim)
    plt.legend(prop={'variant': 'small-caps'})
    plt.xlabel('Round ($t$)')
    plt.ylabel('Average regret $R_t$')
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.title('$L = {:.3g}$'.format(L), y=1.04)

    #-------------- SAVE --------------
    plt.savefig(name)

if __name__ == "__main__":
    policies = [
        [FEWA, {'alpha': .03, 'delta': 1}],
        [FEWA, {'alpha': .06, 'delta': 1}],
        [FEWA, {'alpha': .1, 'delta': 1}],
        [EFF_FEWA, {'alpha': .06,  'delta':1}],
        [wSWA, {'alpha': 0.002}],
        [wSWA, {'alpha': 0.02}],
        [wSWA, {'alpha': 0.2}]
    ]
    L = [0.02 * 1.25 ** (i) for i in range(30)]
    data = {}
    for policy in policies:
        policy_name = str(policy[0](nbArms=2, **policy[1]))
        policy_name_nospace = policy_name.replace(' ', '_')
        policy_data = [
            np.load(os.path.join('./data', file)) for file in os.listdir('./data') if file.startswith("REGRET_"+policy_name_nospace)
        ]
        if not policy_data:
            continue
        policy_data_array =  np.concatenate(policy_data, axis = 1)
        print(len(policy_data), policy_data_array.shape)
        data[policy_name] = {
            "mean" :  policy_data_array.mean(axis=1),
            "uppq" : np.quantile(policy_data_array, 0.9, axis =1),
            "lowq": np.quantile(policy_data_array, 0.9, axis=1)
        }
        for m in data[policy_name]:
            print(data[policy_name][m].shape)
    fig1A(data, L)


