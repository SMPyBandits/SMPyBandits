# -*- coding: utf-8 -*-
"""
author: Julien Seznec

Filtering on Expanding Window Algorithm for rotting bandits.

Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)

Reference : [Seznec et al.,  2019b]
A single algorithm for both rested and restless rotting bandits (WIP)
Julien Seznec, Pierre Ménard, Alessandro Lazaric, Michal Valko
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Julien Seznec"
__version__ = "0.1"

import numpy as np
import time
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
  from .BasePolicy import BasePolicy
except ImportError:
  from BasePolicy import BasePolicy


class EFF_FEWA(BasePolicy):
  """
  Efficient Filtering on Expanding Window Average
  Efficient trick described in [Seznec et al.,  2019a, https://arxiv.org/abs/1811.11043] (m=2)
  and [Seznec et al.,  2019b, WIP] (m<=2)
  We use the confidence level :math:`\delta_t = \frac{1}{t^\alpha}`.
   """

  def __init__(self, nbArms, alpha=0.06, subgaussian=1, m=None, delta=None, delay = False):
    super(EFF_FEWA, self).__init__(nbArms)
    self.alpha = alpha
    self.nbArms = nbArms
    self.subgaussian = subgaussian
    self.delta = delta
    self.inlogconst = 1 / delta ** (1 / alpha) if delta is not None else 1
    self.armSet = np.arange(nbArms)
    self.display_m = m is not None
    self.grid = m if m is not None else 2
    self.statistics = np.ones(shape=(3, self.nbArms, 2)) * np.nan
    # [0,:,:] : current statistics, [1,:,:]: pending statistics, [2,:,:]: number of sample in the pending statistics
    self.windows = np.array([1, int(np.ceil(self.grid))])
    self.outlogconst = self._append_thresholds(self.windows)
    self.delay = np.array([0, np.nan]) if delay else []
    self.idx_nan = np.ones(nbArms)

  def __str__(self):
    if self.delta != None:
      if self.display_m:
        return r"EFF_FEWA($\alpha={:.3g}, \, \delta={:.3g}, \, m={:.3g}$)".format(self.alpha, self.delta, self.grid)
      else:
        return r"EFF_FEWA($\alpha={:.3g}, \, \delta={:.3g}$)".format(self.alpha, self.delta)
    else:
      if self.display_m:
        return r"EFF_FEWA($\alpha={:.3g}, \, m={:.3g}$)".format(self.alpha, self.grid)
      else:
        return r"EFF_FEWA($\alpha={:.3g}$)".format(self.alpha)

  def getReward(self, arm, reward):
    super(EFF_FEWA, self).getReward(arm, reward)
    if not np.all(np.isnan(self.statistics[2, :, -1])):
      add_size = (1 if self.grid > 1.005 else 3) * self.statistics.shape[2]
      self.statistics = np.append(self.statistics, np.nan * np.ones([3, self.nbArms, add_size]), axis=2)
      self.windows = np.append(self.windows, np.array(self._compute_windows(self.windows[-1], add_size), dtype=np.double))
      self.outlogconst = self._append_thresholds(self.windows)
      if len(self.delay):
        self.delay = np.append(self.delay, np.nan * np.ones(add_size))
    self.statistics[1, arm, 0] = reward
    self.statistics[2, arm, 0] = 1
    self.statistics[1, arm, 1:] += reward
    self.statistics[2, arm, 1:] += 1
    idx = np.where((self.statistics[2, arm, :] == self.windows))[0]
    if len(self.delay):
      self.delay += 1
      self.delay[idx] = 0
    self.statistics[0, arm, idx] = self.statistics[1, arm, idx]
    if int(self.idx_nan[arm] -1) in idx:
      idx = np.append(idx, int(self.idx_nan[arm]))
      self.idx_nan[arm] += 1
    self.statistics[1:, arm, idx[idx != 0]] = self.statistics[1:, arm, idx[idx != 0] - 1]

  def choice(self):
    remainingArms = self.armSet.copy()
    i = 0
    selected = remainingArms[np.isnan(self.statistics[0, :, i])] if len(
        remainingArms) != 1 else remainingArms
    sqrtlogt = np.sqrt(np.log(self._inlog()))
    while len(selected) == 0:
      thresh = np.max(self.statistics[0, remainingArms, i]) - sqrtlogt * self.outlogconst[i]
      remainingArms = remainingArms[self.statistics[0, remainingArms, i] >= thresh]
      i += 1
      selected = remainingArms[np.isnan(self.statistics[0, remainingArms, i])] if len(
        remainingArms) != 1 else remainingArms
    return selected[np.argmin(self.pulls[selected])]

  def _append_thresholds(self, w):
    return np.sqrt(8 * w * self.alpha * self.subgaussian ** 2)

  def _compute_windows(self, first_window, add_size):
      last_window = first_window
      res = []
      for i in range(add_size):
        new_window = int(np.ceil(last_window * self.grid))
        res.append(new_window)
        last_window = new_window
      return res


  def _inlog(self):
    return max(self.inlogconst * self.t, 1)

  def startGame(self):
    super(EFF_FEWA, self).startGame()
    self.statistics = np.ones(shape=(3, self.nbArms, 2)) * np.nan
    self.windows = np.array([1, int(np.ceil(self.grid))])
    self.outlogconst = self._append_thresholds(self.windows)


class FEWA(EFF_FEWA):
  """ Filtering on Expanding Window Average.
  Reference: [Seznec et al.,  2019a, https://arxiv.org/abs/1811.11043].
  FEWA is equivalent to EFF_FEWA for :math:`m < 1+1/T` [Seznec et al.,  2019b, WIP].
  This implementation is valid for $:math:`T < 10^{15}`.
  For :math:`T>10^{15}`, FEWA will have time and memory issues as its time and space complexity is O(KT) per round.
  """

  def __init__(self, nbArms, subgaussian=1, alpha=4, delta=None):
    super(FEWA, self).__init__(nbArms, subgaussian=subgaussian, alpha=alpha, delta=delta, m=1 + 10 ** (-15))

  def __str__(self):
    if self.delta != None:
      return r"FEWA($\alpha={:.3g}, \, \delta ={:.3g}$)".format(self.alpha, self.delta)
    else:
      return r"FEWA($\alpha={:.3g}$)".format(self.alpha)




if __name__ == "__main__":
  # Code for debugging purposes.
  start = time.time()
  HORIZON = 100
  sigma = 1
  policy = FEWA(5, subgaussian=sigma, alpha=0.06)
  reward = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}
  res = []
  for t in range(HORIZON):
    choice = policy.choice()
    policy.getReward(choice, reward[choice])
    res.append(policy.delay.copy())

  print(time.time() -start)
  print(policy.statistics.shape)
  print(policy.idx_nan)
  print(policy.pulls)
