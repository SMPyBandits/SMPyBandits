# -*- coding: utf-8 -*-
""" The Aggregated bandit algorithm
Reference: FIXME write it!
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    print("joblib not found. Install it from pypi ('pip install joblib') or conda.")
    USE_JOBLIB = False

import numpy as np
import numpy.random as rn


# Default values for the parameters
update_all_children = False
one_job_by_children = True


class Aggr:
    """ The Aggregated bandit algorithm
    Reference: FIXME write it!
    """

    def __init__(self, nbArms, learningRate, children,
                 decreaseRate=None,
                 update_all_children=update_all_children, prior='uniform',
                 one_job_by_children=one_job_by_children, n_jobs=1, verbosity=5):
        self.nbArms = nbArms
        self.learningRate = learningRate
        self.decreaseRate = decreaseRate
        self.update_all_children = update_all_children
        self.nbChildren = len(children)
        # Parameters for internal use of joblib.Parallel ?
        self.n_jobs = n_jobs
        # Create all child children
        if one_job_by_children:
            # XXX I am not sure if it speeds up to use joblib with more jobs than CPU cores... ?
            self.n_jobs = self.nbChildren
            print("Forcing parallelization for Aggr.Aggr: one job by child algorithm ({}).".format(self.n_jobs))
        self.USE_JOBLIB = USE_JOBLIB
        if self.n_jobs == 1:
            self.USE_JOBLIB = False
        self.verbosity = verbosity
        self.children = []
        for i in range(self.nbChildren):
            self.children.append(children[i]['archtype'](nbArms, **children[i]['params']))
        # Initialize the arrays
        if prior is not None and prior != 'uniform':
            assert len(prior) == self.nbChildren, "Error: the 'prior' argument given to Aggr.Aggr has to be an array of the good size ({}).".format(self.nbChildren)
            self.trusts = prior
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbChildren) / self.nbChildren
        # self.rewards = np.zeros(self.nbArms)
        # self.pulls = np.zeros(self.nbArms)
        self.params = "nb:" + repr(self.nbChildren) + ", rate:" + repr(self.learningRate)
        self.startGame()
        self.choices = (-1) * np.ones(self.nbChildren, dtype=int)

    def __str__(self):
        return "Aggr ({})".format(self.params)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof)
    def startGame(self):
        self.t = 0
        # self.rewards = np.zeros(self.nbArms)
        # self.pulls = np.zeros(self.nbArms)
        # Start all child children
        if self.USE_JOBLIB:
            # FIXME the parallelization here was not improving anything
            joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
                joblib.delayed(delayed_startGame)(self, i)
                for i in range(self.nbChildren)
            )
        else:
            for i in range(self.nbChildren):
                self.children[i].startGame()
        self.choices = (-1) * np.ones(self.nbChildren, dtype=int)

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof)
    def getReward(self, arm, reward):
        self.t += 1
        # self.rewards[arm] += reward
        # self.pulls[arm] += 1
        # FIXME I am trying to reduce the learning rate (geometrically) when t increase...
        if self.decreaseRate is not None:
            learningRate = self.learningRate * np.exp(- self.t / self.decreaseRate)
        else:
            learningRate = self.learningRate
        if self.USE_JOBLIB:
            # FIXME the parallelization here was not improving anything
            joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
                joblib.delayed(delayed_getReward)(self, arm, reward, i)
                for i in range(self.nbChildren)
            )
        else:
            # Give reward to all child children
            for i in range(self.nbChildren):
                self.children[i].getReward(arm, reward)
        # FIXED do this with numpy arrays instead ! FIXME try it !
        scalingConstant = np.exp(reward * learningRate)
        self.trusts[self.choices == arm] *= scalingConstant
        # for i in range(self.nbChildren):
        #     if self.choices[i] == arm:  # this child's choice was chosen
        #         # 3. increase self.trusts for the children who were true
        #         self.trusts[i] *= np.exp(reward * learningRate)
        # FIXED do this with numpy arrays instead ! FIXME try it !
        # DONE test both, by changing the option self.update_all_children
        if self.update_all_children:
            self.trusts[self.choices != arm] /= scalingConstant
            # for i in range(self.nbChildren):
            #     if self.choices[i] != arm:  # this child's choice was not chosen
            #         # 3. XXX decrease self.trusts for the children who were wrong
            #         self.trusts[i] *= np.exp(- reward * learningRate)
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        # print("  The most trusted child policy is the {}th with confidence {}.".format(1 + np.argmax(self.trusts), np.max(self.trusts)))  # DEBUG
        self.trusts = self.trusts / np.sum(self.trusts)
        # print("self.trusts =", self.trusts)  # DEBUG

    # @profile  # DEBUG with kernprof (cf. https://github.com/rkern/line_profiler#kernprof)
    def choice(self):
        # 1. make vote every child children
        if self.USE_JOBLIB:
            # FIXME the parallelization here was not improving anything
            joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
                joblib.delayed(delayed_choice)(self, i)
                for i in range(self.nbChildren)
            )
        else:
            for i in range(self.nbChildren):
                self.choices[i] = self.children[i].choice()
            # ? we could be faster here, first sample according to self.trusts, then make it decide
            # XXX in fact, no we need to vector self.choices to update the self.trusts probabilities!
        # print("self.choices =", self.choices)  # DEBUG
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)


# Helper functions for the parallelization
def delayed_choice(self, j):
    self.choices[j] = self.children[j].choice()


def delayed_getReward(self, arm, reward, j):
    self.children[j].getReward(arm, reward)


def delayed_startGame(self, j):
    self.children[j].startGame()
