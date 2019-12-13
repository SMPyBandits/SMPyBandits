from matplotlib import pyplot as plt
plt.style.use('style.mplstyle')
from Arms import *
from Policies import *
import os
import numpy as np



def fig1A(regret,L, policies, save = True ):
    # --------------  PLOT  --------------
    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot()
    for i in range(regret.shape[1]):
        ax.semilogx(L,regret[:,i,-1], label=policies[i][0](2, **policies[i][1]))

    plt.ylim(0,400)
    plt.legend(prop={'variant': 'small-caps'})
    plt.xlabel('$L$')
    plt.ylabel('Average regret at $T = 10^4$')
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    #-------------- SAVE --------------
    if save:
        plt.savefig("fig1A.pdf")


def fig1A2(regret,low_regret,up_regret, L, policies, save=True):
    # --------------  PLOT  --------------
    fig = plt.figure(figsize=(24, 20))
    ax = plt.subplot()
    #ax.text(0.8, 0.8, 'We show theoretical and empirical parameter value for FEWA and XUCB with 90% confidence band. We see that theoretical FEWA is unpractical (4 times more regret, as predicted by our theory). However, FEWA has a practical tuning far outside its theory limit which have good empirical result. Nevertheless, XUCB has a tuning which slightly outperform FEWA but also enjoys better concentration around its mean. ', color='black',
    #        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    print(regret.shape, policies.shape)
    for i in range(regret.shape[1]):
        ax.semilogx(L, regret[:, i, -1], label=policies[i][0](2, **policies[i][1]))
        ax.fill_between(L, up_regret[:, i, -1], low_regret[:, i, -1], alpha=.5)

    plt.ylim(0, 800)
    plt.legend(prop={'variant': 'small-caps'})
    plt.xlabel('$L$')
    plt.ylabel('Average regret at $T = 10^4$')
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    # -------------- SAVE --------------
    if save:
        plt.savefig("fig1A.pdf")

if __name__ == "__main__":
    # --------------  CONFIG --------------
    policies = [
        #[FEWA, {'alpha': .03, 'delta': 1}],
        #[FEWA, {'alpha': .06, 'delta': 1}],
        #[FEWA, {'alpha': .1, 'delta': 1}],
        #[EFF_FEWA, {'alpha': .06, 'subgaussian': 1, 'delta':1}],
        #[wSWA, {'alpha': 0.006}],
        [wSWA, {'alpha': 0.002}],
        [wSWA, {'alpha': 0.02}],
        [wSWA, {'alpha': 0.2}]
    ]
    L = [0.02 * 1.25 ** (i) for i in range(30)]

    # --------------  LOAD --------------
    print(os.listdir('../../data'))
    #current = '../../data/Fig1/regret291_2.npy'
    #file =  '../../data/2019-03-01_19-37-52_fig1/regret.npy'
    #file2 =  '../../data/2019-03-11_13-50-30_fig1/regret.npy'
    file3 = '../../data/2019-12-04_16-14-27_fig1/regret.npy'



    data = np.load(file3)
    #data1= np.load(file)
    #data2= np.load(file2)
    #data = np.concatenate([data,  data2], axis=1)
    #fig1A(data[:,[1,4,5,6,7,8,9, 10],:], L, policies)
    print(data.shape, len(policies))
    fig1A(data.mean(axis=2), L, np.array(policies))


