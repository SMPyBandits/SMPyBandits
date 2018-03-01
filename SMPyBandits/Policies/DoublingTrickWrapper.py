# -*- coding: utf-8 -*-
r""" A policy that acts as a wrapper on another policy `P`, assumed to be *horizon dependent* (has to known :math:`T`), by implementing a "doubling trick":

- starts to assume that :math:`T=T_0=1000`, and run the policy :math:`P(T_0)`, from :math:`t=1` to :math:`t=T_0`,
- if :math:`t > T_0`, then the "doubling trick" is performed, by either reinitializing or just changing the parameter `horizon` of the policy P, for instance with :math:`T_2 = 10 \times T_0`,
- and keep doing this until :math:`t = T`.

.. note::

   This is implemented in a very generic way, with simply a function `next_horizon(horizon)` that gives the next horizon to try when crossing the current guess.
   It can be a simple linear function (`next_horizon(horizon) = horizon + 100`), a geometric growth to have the "real" doubling trick (`next_horizon(horizon) = horizon * 10`), or even functions growing exponentially fast (`next_horizon(horizon) = horizon ** 1.1`, `next_horizon(horizon) = horizon ** 1.5`, `next_horizon(horizon) = horizon ** 2`).

.. note::

   My guess is that this "doubling trick" wrapping policy can only be efficient if:

   - the underlying policy `P` is a very efficient horizon-dependent algorithm, e.g., the :class:`Policies.DoublingTrickWrapper`,
   - the growth function `next_horizon` is growing faster than any geometric rate, so that the number of refresh is :math:`o(\log T)` and not :math:`O(\log T)`.

.. seealso::

   Reference: [[What the Doubling Trick Can or Can't Do for Multi-Armed Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX), to be presented soon.

.. warning::

   Interface: If `FULL_RESTART=False` (default), the underlying algorithm is recreated at every breakpoint,
   instead its attribute `horizon` or `_horizon` is updated. Be sure that this is enough to really
   change the internal value used by the policy. Some policy use T only once to compute others parameters,
   which should be updated as well. A manual implementation of the `__setattr__` method can help.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"


import numpy as np
from .BasePolicy import BasePolicy

from .UCBH import UCBH

try:
    from .usenumba import jit  # Import numba.jit or a dummy jit(f)=f
except (ValueError, SystemError):
    from usenumba import jit  # Import numba.jit or a dummy jit(f)=f


#: Default horizon-dependent policy
default_horizonDependent_policy = UCBH

#: Default constant to know what to do when restarting the underlying policy with a new horizon parameter.
#:
#: - `True` means that a new policy, initialized from scratch, will be created at every breakpoint.
#: - `False` means that the same policy object is used but just its attribute `horizon` is updated (default).
FULL_RESTART = True
FULL_RESTART = False



#: Default horizon, used for the first step.
DEFAULT_FIRST_HORIZON = 200


#: Default stepsize for the arithmetic horizon progression.
ARITHMETIC_STEP = 10 * DEFAULT_FIRST_HORIZON
ARITHMETIC_STEP = 1 * DEFAULT_FIRST_HORIZON


@jit
def next_horizon__arithmetic(i, horizon):
    r""" The arithmetic horizon progression function:

    .. math::

        T &\mapsto T + 100,\\
        T_i &:= T_0 + 100 \times i.
    """
    return horizon + ARITHMETIC_STEP

next_horizon__arithmetic.__latex_name__ = "arithm"
next_horizon__arithmetic.__latex_name__ = r"$T_i = {} + {} \times i$".format(DEFAULT_FIRST_HORIZON, ARITHMETIC_STEP)


#: Default multiplicative constant for the geometric horizon progression.
GEOMETRIC_STEP = 2


@jit
def next_horizon__geometric(i, horizon):
    r""" The geometric horizon progression function:

    .. math::

        T &\mapsto T \times 2,\\
        T_i &:= T_0 2^i.
    """
    return horizon * GEOMETRIC_STEP

next_horizon__geometric.__latex_name__ = "geom"
next_horizon__geometric.__latex_name__ = r"$T_i = {} \times {}^i$".format(DEFAULT_FIRST_HORIZON, GEOMETRIC_STEP)


#: Default exponential constant for the exponential horizon progression.
EXPONENTIAL_STEP = 1.5


@jit
def next_horizon__exponential(i, horizon):
    r""" The exponential horizon progression function:

    .. math::

        T &\mapsto \left\lfloor T^{1.5} \right\rfloor,\\
        T_i &:= \left\lfloor T_0^{1.5^i} \right\rfloor.
    """
    return int(np.floor(horizon ** EXPONENTIAL_STEP))

next_horizon__exponential.__latex_name__ = "exp"
next_horizon__exponential.__latex_name__ = r"$T_i = {}^{}$".format(DEFAULT_FIRST_HORIZON, r"{%.3g^i}" % EXPONENTIAL_STEP)


#: Default exponential constant for the slow exponential horizon progression.
SLOW_EXPONENTIAL_STEP = 1.1


@jit
def next_horizon__exponential_slow(i, horizon):
    r""" The exponential horizon progression function:

    .. math::

        T &\mapsto \left\lfloor T^{1.1} \right\rfloor,\\
        T_i &:= \left\lfloor T_0^{1.1^i} \right\rfloor.
    """
    return int(np.floor(horizon ** SLOW_EXPONENTIAL_STEP))

next_horizon__exponential_slow.__latex_name__ = "slow exp"
next_horizon__exponential_slow.__latex_name__ = r"$T_i = {}^{}$".format(DEFAULT_FIRST_HORIZON, r"{%.3g^i}" % SLOW_EXPONENTIAL_STEP)


#: Default exponential constant for the fast exponential horizon progression.
FAST_EXPONENTIAL_STEP = 2


@jit
def next_horizon__exponential_fast(i, horizon):
    r""" The exponential horizon progression function:

    .. math::

        T &\mapsto \lfloor T^{2} \rfloor,\\
        T_i &:= \lfloor T_0^{2^i} \rfloor.
    """
    return int(np.floor(horizon ** 2))

next_horizon__exponential_fast.__latex_name__ = "fast exp"
next_horizon__exponential_fast.__latex_name__ = r"$T_i = {}^{}$".format(DEFAULT_FIRST_HORIZON, r"{%.3g^i}" % FAST_EXPONENTIAL_STEP)


#: Default constant :math:`\alpha` for the generic exponential sequence.
ALPHA = 2
#: Default constant :math:`\beta` for the generic exponential sequence.
BETA = 2

def next_horizon__exponential_generic(i, horizon):
    r""" The generic exponential horizon progression function:

    .. math:: T_i := \left\lfloor \frac{T_0}{a} a^{b^i} \right\rfloor.
    """
    return int((DEFAULT_FIRST_HORIZON / ALPHA) * ALPHA ** (BETA ** i))
    # return int(ALPHA * np.floor(horizon ** BETA))

next_horizon__exponential_generic.__latex_name__ = r"exp $a={:.3g}$, $b={:.3g}$".format(ALPHA, BETA)
next_horizon__exponential_generic.__latex_name__ = r"$T_i = ({}/{}) {}^{}$".format(DEFAULT_FIRST_HORIZON, ALPHA, ALPHA, r"{%.3g^i}" % BETA)


#: Chose the default horizon growth function.
# default_next_horizon = next_horizon__arithmetic
# default_next_horizon = next_horizon__geometric
# default_next_horizon = next_horizon__geometric
# default_next_horizon = next_horizon__exponential_fast
default_next_horizon = next_horizon__exponential_slow


# --- Utility function

def breakpoints(next_horizon, first_horizon, horizon, debug=False):
    r""" Return the list of restart point (breakpoints), if starting from ``first_horizon`` to ``horizon`` with growth function ``next_horizon``.

    - Also return the gap between the last guess for horizon and the true horizon. This gap should not be too large.
    - Nicely print all the values if ``debug=True``.

    - First examples:

    >>> first_horizon = 1000
    >>> horizon = 30000
    >>> breakpoints(next_horizon__arithmetic, first_horizon, horizon)  # doctest: +ELLIPSIS
    ([1000, 2000, 3000, 4000, 5000, ..., 28000, 29000, 30000], 0)
    >>> breakpoints(next_horizon__geometric, first_horizon, horizon)
    ([1000, 10000, 100000], 70000)
    >>> breakpoints(next_horizon__exponential, first_horizon, horizon)
    ([1000, 31622], 1622)
    >>> breakpoints(next_horizon__exponential_slow, first_horizon, horizon)
    ([1000, 1995, 4265, 9838, 24671, 67827], 37827)
    >>> breakpoints(next_horizon__exponential_fast, first_horizon, horizon)
    ([1000, 1000000], 970000)

    - Second examples:

    >>> first_horizon = 5000
    >>> horizon = 1000000
    >>> breakpoints(next_horizon__arithmetic, first_horizon, horizon)  # doctest: +ELLIPSIS
    ([5000, 6000, 7000, ..., 998000, 999000, 1000000], 0)
    >>> breakpoints(next_horizon__geometric, first_horizon, horizon)
    ([5000, 50000, 500000, 5000000], 4000000)
    >>> breakpoints(next_horizon__exponential, first_horizon, horizon)
    ([5000, 353553, 210223755], 209223755)
    >>> breakpoints(next_horizon__exponential_slow, first_horizon, horizon)
    ([5000, 11718, 29904, 83811, 260394, 906137, 3572014], 2572014)
    >>> breakpoints(next_horizon__exponential_fast, first_horizon, horizon)
    ([5000, 25000000], 24000000)

    - Third examples:

    >>> first_horizon = 10
    >>> horizon = 1123456
    >>> breakpoints(next_horizon__arithmetic, first_horizon, horizon, debug=True)  # doctest: +ELLIPSIS
    ([10, 1010, ..., 1122010, 1123010, 1124010], 554)
    >>> breakpoints(next_horizon__geometric, first_horizon, horizon, debug=True)
    ([10, 100, 1000, 10000, 100000, 1000000, 10000000], 8876544)
    >>> breakpoints(next_horizon__exponential, first_horizon, horizon, debug=True)
    ([10, 31, 172, 2255, 107082, 35040856], 33917400)
    >>> breakpoints(next_horizon__exponential_slow, first_horizon, horizon, debug=True)
    ([10, 12, 15, 19, 25, 34, 48, 70, 107, 170, 284, 499, 928, 1837, 3895, 8903, 22104, 60106, 180638, 606024, 2294768], 1171312)
    >>> breakpoints(next_horizon__exponential_fast, first_horizon, horizon, debug=True)
    ([10, 100, 10000, 100000000], 98876544)
    """
    i = 0
    t = max(first_horizon, 2)
    times = [t]
    if debug: print("\n\nFor the growth function {}, named '{}', first guess of the horizon = {} and true horizon = {} ...\n ==> The times will be:".format(next_horizon, getattr(next_horizon, '__latex_name__', '?'), first_horizon, horizon))
    while t < horizon:
        t = next_horizon(i, t)
        i += 1
        times.append(t)
        if debug: print("    The {}th breakpoint is {} ...".format(i, t))  # DEBUG
    assert horizon <= t, "Error: the last guess for horizon = {} was found smaller than the true horizon = {}...".format(t, horizon)  # DEBUG
    gap = t - horizon
    if debug: print("This last guess for horizon = {} gives a gap = {} against the true horizon {}. Relative difference = {:.3%}...".format(t, gap, horizon, gap / float(horizon)))  # DEBUG
    return times, gap


# --- The interesting class

class DoublingTrickWrapper(BasePolicy):
    r""" A policy that acts as a wrapper on another policy `P`, assumed to be *horizon dependent* (has to known :math:`T`), by implementing a "doubling trick".

    - Reference: [[What the Doubling Trick Can or Can't Do for Multi-Armed Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-XXX), to be presented soon.
    """

    def __init__(self, nbArms,
                 full_restart=FULL_RESTART,
                 policy=default_horizonDependent_policy,
                 next_horizon=default_next_horizon,
                 first_horizon=DEFAULT_FIRST_HORIZON,
                 lower=0., amplitude=1.,
                 *args, **kwargs):
        super(DoublingTrickWrapper, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.full_restart = full_restart  #: Constant to know how to refresh the underlying policy.
        # --- Policy
        self._policy = policy  # Class to create the underlying policy
        self._args = args  # To keep them
        if 'params' in kwargs:
            kwargs.update(kwargs['params'])
            del kwargs['params']
        self._kwargs = kwargs  # To keep them
        self.policy = None  #: Underlying policy
        # --- Horizon
        self._i = 0
        self._next_horizon = next_horizon  # Function for the growing horizon
        self.next_horizon_name = getattr(next_horizon, '__latex_name__', '?')  #: Pretty string of the name of this growing function
        self._first_horizon = max(2, first_horizon)  # First guess for the horizon
        self.horizon = max(2, first_horizon)  #: Last guess for the horizon
        # XXX Force it, just for pretty printing...
        self.startGame()

    # --- pretty printing

    def __str__(self):
        # remove the T0 part from string representation of the policy
        str_policy = str(self.policy)
        str_policy = str_policy.replace(r"($T={}$)".format(self._first_horizon), "")
        str_policy = str_policy.replace(r"$T={}$, ".format(self._first_horizon), "")
        return r"{}({})[{}]".format("DT" if self.full_restart else "DTnr", self.next_horizon_name, str_policy)

    # --- Start game by creating new underlying policy

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(DoublingTrickWrapper, self).startGame()
        self._i = 0  # reinitialize this
        self.horizon = self._first_horizon  #: Last guess for the horizon
        try:
            self.policy = self._policy(self.nbArms, horizon=self.horizon, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
        except Exception as e:
            print("WARNING: Received exception {} when trying to create the underlying policy... maybe the 'horizon={}' keyword argument was not understood correctly? Retrying without it...".format(e, self.horizon))  # DEBUG
            self.policy = self._policy(self.nbArms, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
        # now also start game for the underlying policy
        self.policy.startGame()
        self.rewards = self.policy.rewards  # just pointers to the underlying arrays!
        self.pulls = self.policy.pulls      # just pointers to the underlying arrays!

    # --- Pass the call to the subpolicy

    def getReward(self, arm, reward):
        """ Pass the reward, as usual, update t and sometimes restart the underlying policy."""
        # print(" - At time t = {}, got a reward = {} from arm {} ...".format(self.t, arm, reward))  # DEBUG
        # super(DoublingTrickWrapper, self).getReward(arm, reward)
        self.t += 1
        self.policy.getReward(arm, reward)

        # Maybe we have to update the horizon?
        if self.t > self.horizon:
            self._i += 1
            new_horizon = self._next_horizon(self._i, self.horizon)
            assert new_horizon > self.horizon, "Error: the new_horizon = {} is not > the current horizon = {} ...".format(new_horizon, self.horizon)  # DEBUG
            # print("  - At time t = {}, a DoublingTrickWrapper class was running with current horizon T_i = {} and decided to use {} as a new horizon...".format(self.t, self.horizon, new_horizon))  # DEBUG
            self.horizon = new_horizon
            # now we have to update or restart the underlying policy
            if self.full_restart:
                try:
                    self.policy = self._policy(self.nbArms, horizon=self.horizon, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
                except Exception as e:
                    # print("Received exception {} when trying to create the underlying policy... maybe the 'horizon={}' keyword argument was not understood correctly? Retrying without it...".format(e, self.horizon))  # DEBUG
                    self.policy = self._policy(self.nbArms, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
                # now also start game for the underlying policy
                self.policy.startGame()
                # print("   ==> Fully restarting the underlying policy by creating a new object... Now it is = {} ...".format(self.policy))  # DEBUG
            else:
                if hasattr(self.policy, 'horizon'):
                    try:
                        self.policy.horizon = self.horizon
                    except AttributeError:
                        pass
                        # print("Warning: unable to update the parameter 'horizon' of the underlying policy {}... Trying '_horizon' ...".format(self.policy))  # DEBUG
                    # print("   ==> Just updating the horizon parameter of the underlying policy... Now it is = {} ...".format(self.policy))  # DEBUG
                # else:
                #     print("   ==> Nothing to do, as the underlying policy DOES NOT have a 'horizon' or '_horizon' parameter that could have been updated... Maybe you are not using a good policy? I suggest UCBH or ApproximatedFHGittins.")  # DEBUG

    # --- Sub methods

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def index(self):
        r""" Get attribute ``index`` from the underlying policy."""
        return self.policy.index

    def choice(self):
        r""" Pass the call to ``choice`` of the underlying policy."""
        return self.policy.choice()

    def choiceWithRank(self, rank=1):
        r""" Pass the call to ``choiceWithRank`` of the underlying policy."""
        return self.policy.choiceWithRank(rank=rank)

    def choiceFromSubSet(self, availableArms='all'):
        r""" Pass the call to ``choiceFromSubSet`` of the underlying policy."""
        return self.policy.choiceFromSubSet(availableArms=availableArms)

    def choiceMultiple(self, nb=1):
        r""" Pass the call to ``choiceMultiple`` of the underlying policy."""
        return self.policy.choiceMultiple(nb=nb)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        r""" Pass the call to ``choiceIMP`` of the underlying policy."""
        return self.policy.choiceIMP(nb=nb, startWithChoiceMultiple=startWithChoiceMultiple)

    def estimatedOrder(self):
        r""" Pass the call to ``estimatedOrder`` of the underlying policy."""
        return self.policy.estimatedOrder()

    def estimatedBestArms(self, M=1):
        r""" Pass the call to ``estimatedBestArms`` of the underlying policy."""
        return self.policy.estimatedBestArms(M=M)

    def computeIndex(self, arm):
        r""" Pass the call to ``computeIndex`` of the underlying policy."""
        return self.policy.computeIndex(arm)

    def computeAllIndex(self):
        r""" Pass the call to ``computeAllIndex`` of the underlying policy."""
        return self.policy.computeAllIndex()


# # --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
