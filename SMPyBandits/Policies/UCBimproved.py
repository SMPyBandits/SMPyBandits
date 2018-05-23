# -*- coding: utf-8 -*-
""" The UCB-Improved policy for bounded bandits, with knowing the horizon, as an example of successive elimination algorithm.

- Reference: [[Auer et al, 2010](https://link.springer.com/content/pdf/10.1007/s10998-010-3055-6.pdf)].
"""

__author__ = "Lilian Besson"
__version__ = "0.9"

from copy import copy
from numpy import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .SuccessiveElimination import SuccessiveElimination
except ImportError:
    from SuccessiveElimination import SuccessiveElimination

#: Default value for parameter :math:`\alpha`.
ALPHA = 0.5


# --- Utility functions

def n_m(horizon, delta_m):
    r""" Function :math:`\lceil \frac{2 \log(T \Delta_m^2)}{\Delta_m^2} \rceil`."""
    return int(np.ceil((2 * np.log(horizon * delta_m**2)) / delta_m**2))


# --- The class

class UCBimproved(SuccessiveElimination):
    """ The UCB-Improved policy for bounded bandits, with knowing the horizon, as an example of successive elimination algorithm.

    - Reference: [[Auer et al, 2010](https://link.springer.com/content/pdf/10.1007/s10998-010-3055-6.pdf)].
    """

    def __init__(self, nbArms, horizon=None, alpha=ALPHA, lower=0., amplitude=1.):
        super(UCBimproved, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert horizon is not None, "Error: UCBimproved requires to know the horizon T."  # DEBUG
        self.horizon = int(horizon)  #: Parameter :math:`T` = known horizon of the experiment.
        self.alpha = alpha  #: Parameter alpha
        #: Set of active arms
        self.activeArms = list(np.arange(self.nbArms))
        #: Current estimate of the gap :math:`\Delta_0`
        self.estimate_delta = 1
        #: Keep in memory the :math:`n_m` quantity, using :func:`n_m`
        self.max_nb_of_exploration = 1
        #: Current round m
        self.current_m = 0
        #: Bound :math:`m = \lfloor \frac{1}{2} \log_2(\frac{T}{e}) \rfloor`
        self.max_m = int(np.floor(0.5 * np.log2(horizon / np.exp(1))))
        #: Also keep in memory when the arm was kicked out of the ``activeArms`` sets, so fake index can be given, if we ask to order the arms for instance.
        self.when_did_it_leave = np.zeros(self.nbArms, dtype=int) + float('-inf')

    def __str__(self):
        return r"UCB-Improved($T={}$, $\alpha={:.3g}$)".format(self.horizon, self.alpha)

    def update_activeArms(self):
        """ Update the set ``activeArms`` of active arms."""
        # first compute UCB and LCB
        assert np.all(self.pulls >= 1), "Error: UCBimproved.update_activeArms() should not be called with min(pulls) = {} but >= 1...".format(np.min(self.pulls))  # DEBUG

        means = self.rewards / self.pulls
        exploration_bias = np.sqrt(self.alpha * (np.log(self.horizon * self.estimate_delta**2)) / self.max_nb_of_exploration)

        LCB = means - exploration_bias
        UCB = means + exploration_bias
        max_LCB = np.max(LCB)

        # now it's time to update activeArms, m, and estimate_delta
        new_active_arms = [
            arm
            for arm in self.activeArms
            if UCB[arm] >= max_LCB
        ]
        # maybe we eliminated some arms, so we need to keep in memory when the left
        if len(new_active_arms) < len(self.activeArms):
            for arm in self.activeArms:
                if arm not in new_active_arms:
                    # self.when_did_it_leave[arm] = self.current_m  # XXX use current_m ?
                    self.when_did_it_leave[arm] = self.t
        self.activeArms = new_active_arms

        # then update the rest
        self.estimate_delta /= 2.0
        self.max_nb_of_exploration = n_m(self.horizon, self.estimate_delta)
        # and finally m
        self.current_m += 1
        assert self.current_m <= self.max_m, "Error: the current m in UCBimproved policy, = {}, should always be smaller than max m = {}...".format(self.current_m, self.max_m)  # DEBUG

    def choice(self, recursive=False):
        r""" In policy based on successive elimination, choosing an arm is the same as choosing an arm from the set of active arms (``self.activeArms``) with method ``choiceFromSubSet``.
        """
        if len(self.activeArms) == 1:
            return self.activeArms[0]
        else:
            arms_not_explored_enough = [
                arm
                for arm in self.activeArms
                if self.pulls[arm] < self.max_nb_of_exploration
            ]
            if len(arms_not_explored_enough) == 0:
                self.update_activeArms()
                if not recursive:
                    return self.choice(recursive=True)
                else:
                    print("Warning: something is wrong for UCBimproved.choice(), in a recursive call it could find any arm not explored enough. Please DEBUG me!")  # DEBUG
                    return np.random.choice(self.nbArms)
            # here we have some arms that are not explored enough
            return np.random.choice(arms_not_explored_enough)

    # --- boring methods

    def computeIndex(self, arm):
        """ Nothing to do, just copy from ``when_did_it_leave``."""
        self.index[arm] = self.when_did_it_leave[arm]
