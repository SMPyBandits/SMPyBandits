# -*- coding: utf-8 -*-
r""" An experimental policy, using algorithms from Unsupervised Learning.

.. warning:: This is still highly experimental!
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from sklearn.neighbors.kde import KernelDensity


T0 = 10          #: Default value for the parameter T_0
FIT_EVERY = 100  #: Default value for the parameter fit_every
MEAN_OF = 50     #: Default value for the parameter meanOf


class UnsupervisedLearning(object):
    r""" Generic policy using an Unsupervised Learning algorithm, for instance from scikit-learn.

    - By default, it uses a [KernelDensity](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity) estimator.

    .. warning:: This is still highly experimental!


    .. info:: The algorithm I designed is not obvious, but here are some explanations:

    - Initially : create :math:`K` Unsupervised Learning algorithms :math:`\mathcal{U}_k(0)`, :math:`k\in\{1,\dots,K\}`, for instance `KernelDensity` estimators.

    - For the first :math:`K \times T_0` time steps, each arm :math:`k \in \{1, \dots, K\}` is sampled exactly :math:`T_0` times, to get a lot of initial observations for each arm.

    - With these first :math:`T_0` (e.g., :math:`50`) observations, train a first version of the Unsupervised Learning algorithms :math:`\mathcal{U}_k(t)`, :math:`k\in\{1,\dots,K\}`.

    - Then, for the following time steps, :math:`t \geq T_0 + 1` :

      + Once in a while (every :math:`T_1 =` `fit_every` steps, e.g., :math:`100`), retrain all the Unsupervised Learning algorithms :
        * For each arm :math:`k\in\{1,\dots,K\}`, use all the previous observations of that arm
          to train the model :math:`\mathcal{U}_k(t)`.

      + Otherwise, use the previously trained model to chose the arm :math:`A(t) \in \{1,\dots,K\}` to play next (see :meth:`choice` below).
    """

    def __init__(self, nbArms, estimator=KernelDensity,
                 T_0=T0, fit_every=FIT_EVERY, meanOf=MEAN_OF,
                 lower=0., amplitude=1.,  # not used, but needed for my framework
                 *args, **kwargs):
        """ Create a new UnsupervisedLearning policy."""
        self.nbArms = nbArms  #: Number of arms of the MAB problem.
        self.t = -1  #: Current time.
        T_0 = int(T_0)
        self.T_0 = int(max(1, T_0))  #: Number of random exploration of each arm at the beginning.
        self.fit_every = int(fit_every)  #: Frequency of refit of the unsupervised models.
        self.meanOf = int(meanOf)  #: Number of samples used to estimate the best arm.
        # Unsupervised Learning algorithm
        self._was_fitted = False
        self._estimator = estimator  #: The class to use to create the estimator.
        self._args = args  #: Other non-kwargs args given to the estimator.
        self._kwargs = kwargs  #: Other kwargs given to the estimator.
        self.ests = [self._estimator(*self._args, **self._kwargs) for _ in range(nbArms)]  #: List of estimators (i.e., an object with a `fit` and `sample` method).
        # Store all the observations
        self.observations = [[] for _ in range(nbArms)]  #: List of observations for each arm. This is the main weakness of this policy: it uses a **linear** storage space, in the number of observations so far (i.e., the time index t), in other words it has a **linear memory complexity** : that's really bad!
        self.lower = lower  #: Known lower bounds on the rewards.
        self.amplitude = amplitude  #: Known lower amplitude of the rewards.

    # --- Easy methods

    def __str__(self):
        return "UnsupervisedLearning({.__name__}, :math:`T_0={:.3g}`, :math:`T_1={:.3g}`, :math:`M={:.3g}`)".format(self._estimator, self.T_0, self.fit_every, self.meanOf)

    def startGame(self):
        """ Reinitialize the estimators."""
        self.t = -1
        self.ests = [self._estimator(*self._args, **self._kwargs) for _ in range(self.nbArms)]

    def getReward(self, armId, reward):
        """ Store this observation `reward` for that arm `armId`."""
        # print("   - At time {}, we saw {} from arm {} ...".format(self.t, reward, armId))  # DEBUG
        self.observations[armId].append(reward)

    # --- The main part of the algorithm

    def choice(self):
        r""" Choose an arm, according to this algorithm:

        * Get a random sample, :math:`s_k(t)` from the :math:`K` Unsupervised Learning algorithms :math:`\mathcal{U}_k(t)`, :math:`k\in\{1,\dots,K\}` :

          .. math: \forall k\in\{1,\dots,K\}, \;\; s_k(t) \sim \mathcal{U}_k(t).

        * Chose the arm :math:`A(t)` with *highest* sample :

          .. math: A(t) \in \arg\max_{k\in\{1,\dots,K\}} s_k(t).

        * Play that arm :math:`A(t)`, receive a reward :math:`r_{A(t)}(t)` from its (unknown) distribution, and store it.


        .. note:: A more robust approach ?

           A more robust (and so more correct) variant could be to use a bunch of samples, and use their mean to give :math:`s_k(t)` :

           * Get a bunch of :math:`M` random samples (e.g., :math:`50`), :math:`s_k^i(t)` from the :math:`K` Unsupervised Learning algorithms :math:`\mathcal{U}_k(t)`, :math:`k\in\{1,\dots,K\}` :

             .. math: \forall k\in\{1,\dots,K\}, \;\; \forall i\in\{1,\dots,M\}, \;\; s_k^i(t) \sim \mathcal{U}_k(t).

           * Average them to get :math:`\hat{s_k}(t)` :

             .. math: \forall k\in\{1,\dots,K\}, \;\; \hat{s_k}(t) := \frac{1}{M} \sum_{i=1}^{M} s_k^i(t).

           * Chose the arm :math:`A(t)` with *highest* mean sample :

             .. math: A(t) \in \arg\max_{k\in\{1,\dots,K\}} \hat{s_k}(t).
        """
        self.t += 1
        # Start by sampling each arm a certain number of times
        if self.t < self.nbArms * self.T_0:
            # print("- First phase: exploring arm {} at time {} ...".format(self.t % self.nbArms, self.t))  # DEBUG
            return self.t % self.nbArms
        else:
            # print("- Second phase: at time {} ...".format(self.t))  # DEBUG
            # 1. Fit the Unsupervised Learning on *all* the data observed so far
            #    but do it once in a while only
            if not self._was_fitted:
                # print("   - Need to first fit the model of each arm with the first {} observations, now of shape {} ...".format(self.fit_every, np.shape(self.observations)))  # DEBUG
                self.fit(self.observations)
                self._was_fitted = True
            elif self.t % self.fit_every == 0:
                # print("   - Need to refit the model of each arm with {} more observations, now of shape {} ...".format(self.fit_every, np.shape(self.observations)))  # DEBUG
                self.fit(self.observations)
            # 2. Sample a random prediction for next output of the arms
            prediction = self.sample_with_mean()
            # exp_score = np.exp(self.score(prediction))
            # Project to the simplex Delta_K, if needed
            # score = exp_score / np.sum(exp_score)
            # print("   - Got a prediction = {} and score {} ...".format(prediction, score))  # DEBUG
            # 3. Use this sample to select next arm to play
            best_arm_predicted = np.argmax(prediction)
            # print("   - So the best arm seems to be = {} ...".format(best_arm_predicted))  # DEBUG
            return best_arm_predicted
            # best_arm_predicted2 = np.argmax(prediction * score)
            # print("   - So the best arm seems to be = {} ...".format(best_arm_predicted2))  # DEBUG
            # # return best_arm_predicted2
            # sampled_arm = np.random.choice(self.nbArms, p=score)
            # print("   - And a random sample from the score was drawn as = {} ...".format(sampled_arm))  # DEBUG
            # return sampled_arm

    # --- Shortcut methods

    def fit(self, data):
        """ Fit each of the K models, with the data accumulated up-to now."""
        for armId in range(self.nbArms):
            # print(" - Fitting the #{} model, with observations of shape {} ...".format(armId + 1, np.shape(self.observations[armId])))  # DEBUG
            est = self.ests[armId]
            est.fit(np.asarray(data[armId]).reshape(-1, 1))
            self.ests[armId] = est

    def sample(self):
        """ Return a vector of random sample from each of the K models."""
        return [float(est.sample()) for est in self.ests]

    def sample_with_mean(self, meanOf=None):
        """ Return a vector of random sample from each of the K models, by averaging a lot of samples (reduce variance)."""
        if meanOf is None:
            meanOf = self.meanOf
        return [float(np.mean(est.sample(meanOf))) for est in self.ests]

    def score(self, obs):
        """ Return a vector of scores, for each of the K models on its observation."""
        return [float(est.score(o)) for est, o in zip(self.ests, obs)]

    # --- Other method

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
        return np.argsort(self.sample_with_mean())
