# -*- coding: utf-8 -*-
r""" A policy that acts as a wrapper on another policy `P`, assumed to be *horizon dependent* (has to known :math:`T`), by implementing a "doubling trick":

- starts to assume that :math:`T=T_1=1000`, and run the policy :math:`P(T_1)`, from :math:`t=1` to :math:`t=T_1`,
- if :math:`t > T_1`, then the "doubling trick" is performed, by either reinitializing or just changing the parameter `horizon` of the policy P, with :math:`T_2 = 10 \times T_1`,
- and keep doing this until :math:`t = T`.

.. note::

   This is implemented in a very generic way, with simply a function `next_horizon(i, horizon)` that gives the next horizon to try when crossing the current guess.
   It can be a simple linear function (`next_horizon(i, horizon) = horizon + 100`), a geometric growth to have the "real" doubling trick (`next_horizon(i, horizon) = horizon * 10`), or even a faster growing function (`next_horizon(i, horizon) = horizon ** 2`).

.. seealso::

   Reference? Not yet, this is my own idea and it is *active* research.
   I will try to experiment on it, prove some things, and if it turns out to be an interesting idea, we will publish something about it.
   Stay posted!

.. warning::

   This is FULLY EXPERIMENTAL! Not tested yet!

.. note::

   My guess is that this "doubling trick" wrapping policy can only be efficient if:

   - the underlying policy `P` is a very efficient hoziron-dependent algorithm, e.g., the :class:`Policies.DoublingTrickWrapper`,
   - the growth function `next_horizon` is growing faster than any geometric rate, so that the number of refresh is :math:`o(\log T)` and not :math:`O(\log T)`.
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

from .BasePolicy import BasePolicy

from .UCBH import UCBH
# from .ApproximatedFHGittins import ApproximatedFHGittins

#: Default horizon-dependent policy
default_horizonDependent_policy = UCBH

#: Default constant to know what to do when restarting the underlying policy
#: with a new horizon parameter. `True` means that a new policy, initialized from scratch,
#: will be created at every breakpoint. `False` means that the same policy object is used
#: but just its attribute `horizon` is updated (default).
FULL_RESTART = True
FULL_RESTART = False


#: Default horizon, used for the first step.
DEFAULT_FIRST_HORIZON = 100


#: Default stepsize for the arithmetic horizon progression.
ARITHMETIC_STEP = 100

def next_horizon__linear(i, horizon):
    r""" The arithmetic horizon progression function:
    
    .. math:: T \mapsto T + 100.
    """
    return horizon + ARITHMETIC_STEP

next_horizon__linear.__latex_name__ = "arithmetic"


#: Default multiplicative constant for the geometric horizon progression.
GEOMETRIC_STEP = 10

def next_horizon__geometric(i, horizon):
    r""" The geometric horizon progression function:
    
    .. math:: T \mapsto T \times 10.
    """
    return horizon * GEOMETRIC_STEP

next_horizon__geometric.__latex_name__ = "geometric"


#: Default exponential constant for the exponential horizon progression.
EXPONENTIAL_STEP = 2

def next_horizon__exponential(i, horizon):
    r""" The exponential horizon progression function:
    
    .. math:: T \mapsto T ** 2.
    """
    return horizon ** EXPONENTIAL_STEP

next_horizon__exponential.__latex_name__ = "exponential"


#: Chose the default horizon growth function.
# default_next_horizon = next_horizon__linear
# default_next_horizon = next_horizon__geometric
default_next_horizon = next_horizon__exponential


# --- The interesting class

class DoublingTrickWrapper(BasePolicy):
    r""" A policy that acts as a wrapper on another policy `P`, assumed to be *horizon dependent* (has to known :math:`T`), by implementing a "doubling trick".
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
        self._policy = default_horizonDependent_policy  # Class to create the underlying policy
        self._args = args
        self._kwargs = kwargs
        self.policy = None  #: Underlying policy
        # --- Horizon
        self.next_horizon = next_horizon  #: Function for the growing horizon
        self.next_horizon_name = getattr(next_horizon, '__latex_name__', '?')  #: Pretty string of the name of this growing function
        self._first_horizon = first_horizon  #: First guess for the horizon
        self.horizon = first_horizon  #: Last guess for the horizon

    def __str__(self):
        return r"DoublingTrick($T_1={:.3g}$, steps: {}{})[{}]".format(self._first_horizon, self.next_horizon_name, ", full restart" if self.full_restart else "", self.policy)

    # --- Start game, and receive rewards

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(DoublingTrickWrapper, self).startGame()
        self.horizon = self._first_horizon  #: Last guess for the horizon
        self.policy = self._policy(self.nbArms, horizon=self.horizon, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
    
    def getReward(self, arm, reward):
        """ Pass the reward, as usual, update t and sometimes restart the underlying policy."""
        # print(" - At time t = {}, got a reward = {} from arm {} ...".format(self.t, arm, reward))  # DEBUG
        super(DoublingTrickWrapper, self).getReward(arm, reward)
        # Maybe we have to update the horizon?
        new_horizon = self.next_horizon(self.t, self.horizon)
        # print("   ==> Considering a new horizon... t = {}, current horizon = {} and next horizon = {} ...".format(self.t, self.horizon, new_horizon))  # DEBUG
        if self.t > self.horizon:
            assert new_horizon > self.horizon, "Error: the new_horizon = {} is not > the current horizon = {} ...".format(new_horizon, self.horizon)  # DEBUG
            print("  - At time t = {}, a DoublingTrickWrapper class was running with current horizon T_i = {} and decided to use {} as a new horizon...".format(self.t, self.horizon, new_horizon))  # DEBUG
            self.horizon = new_horizon
            # now we have to update or restart the underlying policy
            if self.full_restart:
                self.policy = self._policy(self.nbArms, horizon=self.horizon, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
                print("   ==> Fully restarting the underlying policy by creating a new object... Now it is = {} ...".format(self.policy))  # DEBUG
            else:
                if hasattr(self.policy, 'horizon'):
                    try:
                        self.policy.horizon = self.horizon
                    except AttributeError:
                        try:
                            print("Warning: unable to update the parameter 'horizon' of the underlying policy {}... Trying '_horizon' ...".format(self.policy))  # DEBUG
                            self.policy._horizon = self.horizon
                        except AttributeError:
                            print("Warning: unable to update the parameter '_horizon' of the underlying policy {} ...".format(self.policy))  # DEBUG
                    print("   ==> Just updating the horizon parameter of the underlying policy... Now it is = {} ...".format(self.policy))  # DEBUG
                else:
                    print("   ==> Nothing to do, as the underlying policy DOES NOT have a 'horizon' parameter that could have been updated... Maybe you are not using a good policy? I suggest UCBH or ApproximatedFHGittins.")  # DEBUG

    # --- Sub methods

    def choice(self):
        r""" Pass the call to the underlying policy."""
        return self.policy.choice()

    def choiceWithRank(self, rank=1):
        r""" Pass the call to the underlying policy."""
        return self.policy.choiceWithRank(rank=rank)

    def choiceFromSubSet(self, availableArms='all'):
        r""" Pass the call to the underlying policy."""
        return self.policy.choiceFromSubSet(availableArms=availableArms)

    def choiceMultiple(self, nb=1):
        r""" Pass the call to the underlying policy."""
        return self.policy.choiceMultiple(nb=nb)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        r""" Pass the call to the underlying policy."""
        return self.policy.choiceIMP(nb=nb, startWithChoiceMultiple=startWithChoiceMultiple)

    # --- Hack!

    def __getattr__(self, name):
        """ Generic method to capture all attribute/method call and pass them to the underlying policy."""
        # print("Using hacking method DoublingTrickWrapper.__getattr__({}, {})...".format(self, name))  # DEBUG
        if hasattr(self.policy, name):
            return getattr(self.policy, name)
        else:
            raise AttributeError
        