# -*- coding: utf-8 -*-
""" The Aggregated bandit algorithm
Reference: FIXME write it!
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.1"

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
                 update_all_children=update_all_children, prior='uniform',
                 one_job_by_children=one_job_by_children, n_jobs=1, verbosity=5):
        self.nbArms = nbArms
        self.learningRate = learningRate
        self.update_all_children = update_all_children
        self.nbChildren = len(children)
        # Parameters for joblib
        self.n_jobs = n_jobs
        # Create all child children
        if one_job_by_children:
            # XXX I am not sure if it speeds up to use joblib with more jobs than CPU cores... ?
            self.n_jobs = self.nbChildren
            print("Forcing parallelization for Aggr.Aggr: one job by child algorithm ({}).".format(self.n_jobs))
        self.verbosity = verbosity
        self.children = []
        for i in range(self.nbChildren):
            self.children.append(children[i]['archtype'](nbArms, **children[i]['params']))
        # Initialize the arrays
        if prior is not None and prior != 'uniform':
            assert len(prior) == self.nbChildren, "Error: the 'prior' argument given to Aggr.Aggr has to be an array of the good size ({}).".format(self.nbChildren)
            self.trusts = prior
        else:   # Assume uniform prior if not given or if = 'uniform'
            self.trusts = np.ones(self.nbChildren) / float(self.nbChildren)
        self.rewards = np.zeros(self.nbArms)
        self.pulls = np.zeros(self.nbArms)
        self.params = "nbChildren:" + repr(self.nbChildren)
        self.startGame()
        self.choices = None

    def __str__(self):
        return "Aggr ({})".format(self.params)

    def startGame(self):
        self.rewards[:] = 0
        self.pulls[:] = 0
        self.t = 1
        # Start all child children
        if USE_JOBLIB:
            # FIXME test and debug parallelization here!
            joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
                joblib.delayed(delayed_startGame)(self, i)
                for i in range(self.nbChildren)
            )
        else:
            for i in range(self.nbChildren):
                self.children[i].startGame()

    def getReward(self, arm, reward):
        self.rewards[arm] += reward
        self.pulls[arm] += 1
        self.t += 1
        if USE_JOBLIB:
            # FIXME test and debug parallelization here!
            joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
                joblib.delayed(delayed_getReward)(self, arm, reward, i)
                for i in range(self.nbChildren)
            )
        else:
            # Give reward to all child children
            for i in range(self.nbChildren):
                self.children[i].getReward(arm, reward)
        for i in range(self.nbChildren):
            if self.choices[i] == arm:  # this child's choice was chosen
                # 3. increase self.trusts for the children who were true
                self.trusts[i] *= np.exp(reward * self.learningRate)
            # DONE test both, by changing the option self.update_all_children
            elif self.update_all_children:
                # 3. XXX decrease self.trusts for the children who were wrong
                self.trusts[i] *= np.exp(- reward * self.learningRate)
        # 4. renormalize self.trusts to make it a proba dist
        # In practice, it also decreases the self.trusts for the children who were wrong
        # print("  The most trusted child policy is the {}th with confidence {}.".format(1 + np.argmax(self.trusts), np.max(self.trusts)))  # DEBUG
        self.trusts = self.trusts / float(np.sum(self.trusts))
        # print("self.trusts =", self.trusts)  # DEBUG

    def choice(self):
        # 1. make vote every child children
        self.choices = [-1] * self.nbChildren
        if USE_JOBLIB:
            # FIXME test and debug parallelization here!
            self.choices = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
                joblib.delayed(delayed_choice)(self, i)
                for i in range(self.nbChildren)
            )
        else:
            for i in range(self.nbChildren):
                self.choices[i] = self.children[i].choice()
        # print("self.choices =", self.choices)  # DEBUG
        # 2. select the vote to trust, randomly
        return rn.choice(self.choices, p=self.trusts)


# Helper functions for the parallelization
def delayed_choice(self, j):
    return self.children[j].choice()


def delayed_getReward(self, arm, reward, j):
    return self.children[j].getReward(arm, reward)


def delayed_startGame(self, j):
    return self.children[j].startGame()
