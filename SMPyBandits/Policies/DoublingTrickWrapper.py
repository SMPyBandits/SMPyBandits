# -*- coding: utf-8 -*-
r""" A policy that acts as a wrapper on another policy `P`, assumed to be *horizon dependent* (has to known :math:`T`), by implementing a "doubling trick":

- starts to assume that :math:`T=T_0=1000`, and run the policy :math:`P(T_0)`, from :math:`t=1` to :math:`t=T_0`,
- if :math:`t > T_0`, then the "doubling trick" is performed, by either re-initializing or just changing the parameter `horizon` of the policy P, for instance with :math:`T_2 = 10 \times T_0`,
- and keep doing this until :math:`t = T`.

.. note::

   This is implemented in a very generic way, with simply a function `next_horizon(horizon)` that gives the next horizon to try when crossing the current guess.
   It can be a simple linear function (`next_horizon(horizon) = horizon + 100`), a geometric growth to have the "real" doubling trick (`next_horizon(horizon) = horizon * 10`), or even functions growing exponentially fast (`next_horizon(horizon) = horizon ** 1.1`, `next_horizon(horizon) = horizon ** 1.5`, `next_horizon(horizon) = horizon ** 2`).

.. note::

   My guess is that this "doubling trick" wrapping policy can only be efficient (for stochastic problems) if:

   - the underlying policy `P` is a very efficient horizon-dependent algorithm, e.g., the :class:`Policies.ApproximatedFHGittins`,
   - the growth function `next_horizon` is growing faster than any geometric rate, so that the number of refresh is :math:`o(\log T)` and not :math:`O(\log T)`.

.. seealso::

   Reference: [[What the Doubling Trick Can or Can't Do for Multi-Armed Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-01736357), to be presented soon.

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
try:
    from .BaseWrapperPolicy import BaseWrapperPolicy
    from .UCBH import UCBH
except ImportError:
    from BaseWrapperPolicy import BaseWrapperPolicy
    from UCBH import UCBH
try:
    from .usenumba import jit  # Import numba.jit or a dummy jit(f)=f
except (ValueError, ImportError, SystemError):
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
    ([1000, 1200, 1400, ..., 29800, 30000], 0)
    >>> breakpoints(next_horizon__geometric, first_horizon, horizon)
    ([1000, 2000, 4000, 8000, 16000, 32000], 2000)
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
    ([5000, 5200, ..., 999600, 999800, 1000000], 0)
    >>> breakpoints(next_horizon__geometric, first_horizon, horizon)
    ([5000, 10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000], 280000)
    >>> breakpoints(next_horizon__exponential, first_horizon, horizon)
    ([5000, 353553, 210223755], 209223755)
    >>> breakpoints(next_horizon__exponential_slow, first_horizon, horizon)
    ([5000, 11718, 29904, 83811, 260394, 906137, 3572014], 2572014)
    >>> breakpoints(next_horizon__exponential_fast, first_horizon, horizon)
    ([5000, 25000000], 24000000)

    - Third examples:

    >>> first_horizon = 10
    >>> horizon = 1123456
    >>> breakpoints(next_horizon__arithmetic, first_horizon, horizon)  # doctest: +ELLIPSIS
    ([10, 210, 410, ..., 1123210, 1123410, 1123610], 154)
    >>> breakpoints(next_horizon__geometric, first_horizon, horizon)
    ([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680, 655360, 1310720], 187264)
    >>> breakpoints(next_horizon__exponential, first_horizon, horizon)
    ([10, 31, 172, 2255, 107082, 35040856], 33917400)
    >>> breakpoints(next_horizon__exponential_slow, first_horizon, horizon)
    ([10, 12, 15, 19, 25, 34, 48, 70, 107, 170, 284, 499, 928, 1837, 3895, 8903, 22104, 60106, 180638, 606024, 2294768], 1171312)
    >>> breakpoints(next_horizon__exponential_fast, first_horizon, horizon)
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


# --- Experimental code to plot some doubling sequences and
# check numerically some inequalities :
# like controlling a sum Sigma_i=0^n u_i by a constant times to last term u_n
# and controlling the last term u_{L_T} as a function of T.


#: The constant c in front of the function f.
constant_c_for_the_functions_f = 1.0
constant_c_for_the_functions_f = 0.1
constant_c_for_the_functions_f = 0.5


def function_f__for_geometric_sequences(i, c=constant_c_for_the_functions_f):
    r""" For the *geometric* doubling sequences, :math:`f(i) = c \times \log(i)`."""
    if i <= 0: return 0.0
    return c * np.log(i)


def function_f__for_exponential_sequences(i, c=constant_c_for_the_functions_f):
    r""" For the *exponential* doubling sequences, :math:`f(i) = c \times i`."""
    return c * i


def function_f__for_generic_sequences(i, c=constant_c_for_the_functions_f, d=0.5, e=0.0):
    r""" For a certain *generic* family of doubling sequences, :math:`f(i) = c \times i^{d} \times (\log(i))^{e}`.

    - ``d, e = 0, 1`` gives :func:`function_f__for_geometric_sequences`,
    - ``d, e = 1, 0`` gives :func:`function_f__for_geometric_sequences`,
    - ``d, e = 0.5, 0`` gives an intermediate sequence, growing faster than any geometric sequence and slower than any exponential sequence,
    - any other combination has not been studied yet.

    .. warning:: ``d`` should most probably be smaller than 1.
    """
    i = float(i)
    if i <= 0: return 0.0
    if e == 0:
        assert d > 0, "Error: invalid value of d = {} for function_f__for_generic_sequences.".format(d)  # DEBUG
        return c * (i ** d)
    if d == 0:
        assert e > 0, "Error: invalid value of e = {} for function_f__for_generic_sequences.".format(e)  # DEBUG
        return c * ((np.log(i)) ** e)
    return c * (i ** d) * ((np.log(i)) ** e)


def function_f__for_intermediate_sequences(i):
    return function_f__for_generic_sequences(i, c=constant_c_for_the_functions_f, d=0.5, e=0.0)

def function_f__for_intermediate2_sequences(i):
    return function_f__for_generic_sequences(i, c=constant_c_for_the_functions_f, d=0.3333, e=0.0)

def function_f__for_intermediate3_sequences(i):
    return function_f__for_generic_sequences(i, c=constant_c_for_the_functions_f, d=0.6667, e=0.0)

def function_f__for_intermediate4_sequences(i):
    return function_f__for_generic_sequences(i, c=constant_c_for_the_functions_f, d=0.5, e=0.5)

def function_f__for_intermediate5_sequences(i):
    return function_f__for_generic_sequences(i, c=constant_c_for_the_functions_f, d=1, e=-1)


#: Value of the parameter :math:`\alpha` for the :func:`Ti_from_f` function.
alpha_for_Ti = 0.1
alpha_for_Ti = 1.0
alpha_for_Ti = 0.5


def Ti_from_f(f, alpha=alpha_for_Ti, *args, **kwargs):
    r""" For any non-negative and increasing function :math:`f: i \mapsto f(i)`, the corresponding sequence is defined by:

    .. math:: \forall i\in\mathbb{N},\; T_i := \lfloor \exp(\alpha \times \exp(f(i))) \rfloor.

    .. warning:: :math:`f(i)` can need other parameters, see the examples above. They can be given as ``*args`` or ``**kwargs`` to :func:`Ti_from_f`.

    .. warning:: it should be computed otherwise, I should give :math:`i \mapsto \exp(f(i))` instead of :math:`f: i \mapsto f(i)`. I need to try as much as possible to reduce the risk of overflow errors!
    """
    # WARNING don't forget the floor!
    def Ti(i):
        this_Ti = np.floor(np.exp(alpha * np.exp(f(float(i), *args, **kwargs))))
        if not (np.isinf(this_Ti) or np.isnan(this_Ti)):
            this_Ti = int(this_Ti)
        # print("    For f = {}, i = {} gives Ti = {}".format(f, i, this_Ti))  # DEBUG
        return this_Ti
    return Ti


def Ti_geometric(i, horizon, alpha=alpha_for_Ti, first_horizon=DEFAULT_FIRST_HORIZON, *args, **kwargs):
    """ Sequence :math:`T_i` generated from the function :math:`f` = :func:`function_f__for_geometric_sequences`."""
    f = function_f__for_geometric_sequences
    this_Ti = first_horizon + np.floor(np.exp(alpha * np.exp(f(float(i), *args, **kwargs))))
    if not (np.isinf(this_Ti) or np.isnan(this_Ti)): this_Ti = int(this_Ti)
    return this_Ti
Ti_geometric.__latex_name__                 = r"$f(i)=\log(i)$"

def Ti_exponential(i, horizon, alpha=alpha_for_Ti, first_horizon=DEFAULT_FIRST_HORIZON, *args, **kwargs):
    """ Sequence :math:`T_i` generated from the function :math:`f` = :func:`function_f__for_exponential_sequences`."""
    f = function_f__for_exponential_sequences
    this_Ti = first_horizon + np.floor(np.exp(alpha * np.exp(f(float(i), *args, **kwargs))))
    if not (np.isinf(this_Ti) or np.isnan(this_Ti)): this_Ti = int(this_Ti)
    return this_Ti
Ti_exponential.__latex_name__             = r"$f(i)=i$"

def Ti_intermediate_sqrti(i, horizon, alpha=alpha_for_Ti, first_horizon=DEFAULT_FIRST_HORIZON, *args, **kwargs):
    """ Sequence :math:`T_i` generated from the function :math:`f` = :func:`function_f__for_intermediate_sequences`."""
    f = function_f__for_intermediate_sequences
    this_Ti = first_horizon + np.floor(np.exp(alpha * np.exp(f(float(i), *args, **kwargs))))
    if not (np.isinf(this_Ti) or np.isnan(this_Ti)): this_Ti = int(this_Ti)
    return this_Ti
Ti_intermediate_sqrti.__latex_name__      = r"$f(i)=\sqrt{i}$"

def Ti_intermediate_i13(i, horizon, alpha=alpha_for_Ti, first_horizon=DEFAULT_FIRST_HORIZON, *args, **kwargs):
    """ Sequence :math:`T_i` generated from the function :math:`f` = :func:`function_f__for_intermediate2_sequences`."""
    f = function_f__for_intermediate2_sequences
    this_Ti = first_horizon + np.floor(np.exp(alpha * np.exp(f(float(i), *args, **kwargs))))
    if not (np.isinf(this_Ti) or np.isnan(this_Ti)): this_Ti = int(this_Ti)
    return this_Ti
Ti_intermediate_i13.__latex_name__        = r"$f(i)=i^{1/3}$"

def Ti_intermediate_i23(i, horizon, alpha=alpha_for_Ti, first_horizon=DEFAULT_FIRST_HORIZON, *args, **kwargs):
    """ Sequence :math:`T_i` generated from the function :math:`f` = :func:`function_f__for_intermediate3_sequences`."""
    f = function_f__for_intermediate3_sequences
    this_Ti = first_horizon + np.floor(np.exp(alpha * np.exp(f(float(i), *args, **kwargs))))
    if not (np.isinf(this_Ti) or np.isnan(this_Ti)): this_Ti = int(this_Ti)
    return this_Ti
Ti_intermediate_i23.__latex_name__        = r"$f(i)=i^{2/3}$"

def Ti_intermediate_i12_logi12(i, horizon, alpha=alpha_for_Ti, first_horizon=DEFAULT_FIRST_HORIZON, *args, **kwargs):
    """ Sequence :math:`T_i` generated from the function :math:`f` = :func:`function_f__for_intermediate4_sequences`."""
    f = function_f__for_intermediate4_sequences
    this_Ti = first_horizon + np.floor(np.exp(alpha * np.exp(f(float(i), *args, **kwargs))))
    if not (np.isinf(this_Ti) or np.isnan(this_Ti)): this_Ti = int(this_Ti)
    return this_Ti
Ti_intermediate_i12_logi12.__latex_name__ = r"$f(i)=\sqrt{i \log(i)}$"

def Ti_intermediate_i_by_logi(i, horizon, alpha=alpha_for_Ti, first_horizon=DEFAULT_FIRST_HORIZON, *args, **kwargs):
    """ Sequence :math:`T_i` generated from the function :math:`f` = :func:`function_f__for_intermediate5_sequences`."""
    f = function_f__for_intermediate5_sequences
    this_Ti = first_horizon + np.floor(np.exp(alpha * np.exp(f(float(i + 1), *args, **kwargs))))
    if not (np.isinf(this_Ti) or np.isnan(this_Ti)): this_Ti = int(this_Ti)
    return this_Ti
Ti_intermediate_i_by_logi.__latex_name__  = r"$f(i)=i / \log(i)$"


def last_term_operator_LT(Ti, max_i=10000):
    r""" For a certain function representing a doubling sequence, :math:`T: i \mapsto T_i`, this :func:`last_term_operator_LT` function returns the function :math:`L: T \mapsto L_T`, defined as:

    .. math:: \forall T\in\mathbb{N},\; L_T := \min\{ i \in\mathbb{N},\; T \leq T_i \}.

    :math:`L_T` is the only integer which satisfies :math:`T_{L_T - 1} < T \leq T_{L_T}`.
    """
    def LT(T, max_i=max_i):
        i = 0
        while Ti(i) < T:
            i += 1
            if i >= max_i:
                raise ValueError("LT(T={T}) was unable to find a i <= {max_i} such that T_i >= T.".format(T, max_i))  # DEBUG
        assert Ti(i - 1) < T <= Ti(i), "Error: i = {} was computed as LT for T = {} and Ti = {} but does not satisfy T_(i-1) < T <= T(i)".format(i, T, Ti)  # DEBUG
        # print("  For LT: i = {} was computed as LT for T = {} and Ti = {} and satisfies T(i-1) = {} < T <= T(i) = {}".format(i, T, Ti, Ti(i-1), Ti(i)))  # DEBUG
        return i
    return LT


import matplotlib.pyplot as plt
import seaborn as sns


def plot_doubling_sequences(
        i_min=1, i_max=30,
        list_of_f=(
            function_f__for_geometric_sequences,
            function_f__for_intermediate_sequences,
            function_f__for_intermediate2_sequences,
            function_f__for_intermediate3_sequences,
            function_f__for_intermediate4_sequences,
            function_f__for_exponential_sequences,
            ),
        label_of_f=(
            "Geometric    doubling (d=0, e=1)",
            "Intermediate doubling (d=1/2, e=0)",
            "Intermediate doubling (d=1/3, e=0)",
            "Intermediate doubling (d=2/3, e=0)",
            "Intermediate doubling (d=1/2, e=1/2)",
            "Exponential  doubling (d=1, e=0)",
            ),
        *args, **kwargs
    ):
    r""" Display a plot to illustrate the values of the :math:`T_i` as a function of :math:`i` for some i.

    - Can accept many functions f (and labels).
    """
    # Make unique markers
    nb = len(list_of_f)
    allmarkers = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']
    longlist = allmarkers * (1 + int(nb / float(len(allmarkers))))  # Cycle the good number of time
    markers = longlist[:nb]  # Truncate
    # Make unique colors
    colors = sns.hls_palette(nb + 1)[:nb]

    fig = plt.figure()
    # plt.hold(True)

    i_s = np.arange(i_min, i_max)
    # now for each function f
    for num_f, (f, la) in enumerate(zip(list_of_f, label_of_f)):
        print("\n\nThe {}th function is referred to as {} and is {}".format(num_f, la, f))  # DEBUG

        Ti = Ti_from_f(f)
        values_of_Ti = np.array([ Ti(i) for i in i_s ])
        plt.plot(i_s, values_of_Ti, label=la, lw=3, ms=3, color=colors[num_f], marker=markers[num_f])
    plt.legend()
    plt.xlabel(r"Value of the time horizon $i = {},...,{}$".format(i_min, i_max))
    plt.title(r"Comparison of the values of $T_i$")
    plt.show()
    return fig


def plot_quality_first_upper_bound(
        Tmin=10, Tmax=int(1e8), nbTs=100,
        gamma=0.0, delta=1.0,  # XXX bound in RT <= log(T)
        # gamma=0.5, delta=0.0,  # XXX bound in RT <= sqrt(T)
        # gamma=0.5, delta=0.5,  # XXX bound in RT <= sqrt(T * log(T))
        # gamma=0.66667, delta=1.0,  # XXX another weird bound in RT <= T^2/3 * log(T)
        list_of_f=(
            function_f__for_geometric_sequences,
            function_f__for_intermediate_sequences,
            function_f__for_intermediate2_sequences,
            function_f__for_intermediate3_sequences,
            function_f__for_intermediate4_sequences,
            function_f__for_exponential_sequences,
            ),
        label_of_f=(
            "Geometric    doubling (d=0, e=1)",
            "Intermediate doubling (d=1/2, e=0)",
            "Intermediate doubling (d=1/3, e=0)",
            "Intermediate doubling (d=2/3, e=0)",
            "Intermediate doubling (d=1/2, e=1/2)",
            "Exponential  doubling (d=1, e=0)",
            ),
        show_Ti_m_Tim1=True,
        # show_Ti_m_Tim1=False,  # DEBUG
        *args, **kwargs
    ):
    r""" Display a plot to compare numerically between the following sum :math:`S` and the upper-bound we hope to have, :math:`T^{\gamma} (\log T)^{\delta}`, as a function of :math:`T` for some values between :math:`T_{\min}` and :math:`T_{\max}`:

    .. math:: S := \sum_{i=0}^{L_T} (T_i - T_{i-1})^{\gamma} (\log (T_i - T_{i-1}))^{\delta}.

    - Can accept many functions f (and labels).
    - Can use :math:`T_i` instead of :math:`T_i - T_{i-1}` if ``show_Ti_m_Tim1=False`` (default is to use the smaller possible bound, with difference of sequence lengths, :math:`T_i - T_{i-1}`).

    .. warning:: This is still ON GOING WORK.
    """
    # Make unique markers
    nb = len(list_of_f)
    allmarkers = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']
    longlist = allmarkers * (1 + int(nb / float(len(allmarkers))))  # Cycle the good number of time
    markers = longlist[:nb]  # Truncate
    # Make unique colors
    colors = sns.hls_palette(nb + 1)[:nb]

    fig = plt.figure()
    # plt.hold(True)

    Ts = np.floor(np.linspace(Tmin, Tmax, num=nbTs))
    the_bound_we_want = (Ts ** gamma) * (np.log(Ts) ** delta)

    # plt.plot(Ts, the_bound_we_want, label=r"$T^{\gamma} (\log T)^{\delta}$", lw=3, ms=3, color=colors[0], marker=markers[0])
    # compute the sequence lengths to use, either T_i or T_i - T_{i-1}
    Ts_for_f = np.copy(Ts)
    if show_Ti_m_Tim1: Ts_for_f[1:] = np.diff(Ts)

    # now for each function f
    for num_f, (f, la) in enumerate(zip(list_of_f, label_of_f)):
        print("\n\nThe {}th function is referred to as {} and is {}".format(num_f, la, f))  # DEBUG
        Ti = Ti_from_f(f)
        LT = last_term_operator_LT(Ti)
        the_sum_we_have = np.zeros_like(Ts_for_f)
        for j, (Tj, dTj) in enumerate(zip(Ts, Ts_for_f)):
            LTj = LT(Tj)
            the_sum_we_have[j] = sum(
                (dTj ** gamma) * (np.log(dTj) ** delta)
                for i in range(0, LTj + 1)
            )
            print("For j = {}, Tj = {}, dTj = {}, gives LTj = {}, and the value of the sum from i=0 to LTj is = {}.".format(j, Tj, dTj, LTj, the_sum_we_have[j]))  # DEBUG
        print("the_sum_we_have =", the_sum_we_have)  # DEBUG
        plt.plot(Ts, the_sum_we_have / the_bound_we_want, label=la, lw=3, ms=3, color=colors[num_f], marker=markers[num_f])

    plt.legend()
    plt.xlabel(r"Value of the time horizon $T = {},...,{}$".format(Tmin, Tmax))
    str_of_Tj_or_dTj = "T_i - T_{i-1}" if show_Ti_m_Tim1 else "T_i"
    plt.title(r"Ratio of the sum $\sum_{i=0}^{L_T} (%s)^{\gamma} (\log(%s))^{\delta}$ and the upper-bound $T^{\gamma} \log(T)^{\delta}$, for $\gamma=%.3g$, $\delta=%.3g$." % (str_of_Tj_or_dTj, str_of_Tj_or_dTj, gamma, delta))  # DEBUG
    plt.show()
    return fig


# --- The interesting class

#: If the sequence Ti does not grow enough, artificially increase i until T_inext > T_i
MAX_NB_OF_TRIALS = 500


class DoublingTrickWrapper(BaseWrapperPolicy):
    r""" A policy that acts as a wrapper on another policy `P`, assumed to be *horizon dependent* (has to known :math:`T`), by implementing a "doubling trick".

    - Reference: [[What the Doubling Trick Can or Can't Do for Multi-Armed Bandits, Lilian Besson and Emilie Kaufmann, 2018]](https://hal.inria.fr/hal-01736357), to be presented soon.
    """

    def __init__(self, nbArms,
                 full_restart=FULL_RESTART,
                 policy=default_horizonDependent_policy,
                 next_horizon=default_next_horizon,
                 first_horizon=DEFAULT_FIRST_HORIZON,
                 *args, **kwargs):
        super(DoublingTrickWrapper, self).__init__(nbArms, policy=policy, *args, **kwargs)
        self.full_restart = full_restart  #: Constant to know how to refresh the underlying policy.
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
        super(BaseWrapperPolicy, self).startGame()
        # super(DoublingTrickWrapper, self).startGame()  # WARNING no
        self._i = 0  # reinitialize this
        self.horizon = self._first_horizon  #: Last guess for the horizon
        try:
            self.policy = self._policy(self.nbArms, horizon=self.horizon, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
        except Exception as e:
            print("WARNING: Received exception {} when trying to create the underlying policy... maybe the 'horizon={}' keyword argument was not understood correctly? Retrying without it...".format(e, self.horizon))  # DEBUG
            self.policy = self._policy(self.nbArms, lower=self.lower, amplitude=self.amplitude, *self._args, **self._kwargs)
        # now also start game for the underlying policy
        self.policy.startGame()

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
            # XXX <!-- small hack if the sequence is not growing fast enough
            nb_of_trials = 1
            while nb_of_trials < MAX_NB_OF_TRIALS and new_horizon <= self.horizon:
                self._i += 1
                nb_of_trials += 1
                new_horizon = self._next_horizon(self._i, self.horizon)
            # XXX end of small hack -->
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


# # --- Debugging

if __name__ == "__main__":
    import sys
    if "plot" in sys.argv[1:]:
        plt.ion()
        # plot_doubling_sequences()
        for gamma, delta in [
            (0.0, 1.0),  # XXX bound in RT <= log(T)
            (0.5, 0.0),  # XXX bound in RT <= sqrt(T)
            (0.5, 0.5),  # XXX bound in RT <= sqrt(T * log(T))
            (0.66667, 1.0),  # XXX another weird bound in RT <= T^2/3 * log(T)
        ]:
            plot_quality_first_upper_bound(gamma=gamma, delta=delta, show_Ti_m_Tim1=True)
            plot_quality_first_upper_bound(gamma=gamma, delta=delta, show_Ti_m_Tim1=False)
        sys.exit(0)

    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
