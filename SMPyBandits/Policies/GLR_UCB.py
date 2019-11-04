# -*- coding: utf-8 -*-
r""" The GLR-UCB policy and variants, for non-stationary bandits.

- Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
- It runs on top of a simple policy, e.g., :class:`UCB`, and :class:`BernoulliGLR_IndexPolicy` is a wrapper:

    >>> policy = BernoulliGLR_IndexPolicy(nbArms, UCB)
    >>> # use policy as usual, with policy.startGame(), r = policy.choice(), policy.getReward(arm, r)

- It uses an additional :math:`\mathcal{O}(\tau_\max)` memory for a game of maximum stationary length :math:`\tau_\max`.

.. warning:: It can only work on basic index policy based on empirical averages (and an exploration bias), like :class:`UCB`, and cannot work on any Bayesian policy (for which we would have to remember all previous observations in order to reset the history with a small history)!
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np
from math import log, sqrt, isinf

try:
    from .with_proba import with_proba
    from .UCB import UCB as DefaultPolicy
    from .CD_UCB import CD_IndexPolicy
except ImportError:
    from with_proba import with_proba
    from UCB import UCB as DefaultPolicy
    from CD_UCB import CD_IndexPolicy


#: Whether to be verbose when doing the change detection algorithm.
VERBOSE = False

#: Default probability of random exploration :math:`\alpha`.
PROBA_RANDOM_EXPLORATION = 0.1

#: For any algorithm with uniform exploration and a formula to tune it, :math:`\alpha` is usually too large and leads to larger regret. Multiplying it by a 0.1 or 0.2 helps, a lot!
# ALPHA0_SCALE_FACTOR = 1
ALPHA0_SCALE_FACTOR = 0.1

#: Should we reset one arm empirical average or all? Default is ``True``, it's usually more efficient!
PER_ARM_RESTART = True

#: Should we fully restart the algorithm or simply reset one arm empirical average? Default is ``False``, it's usually more efficient!
FULL_RESTART_WHEN_REFRESH = False

#: XXX Be lazy and try to detect changes only X steps, where X is small like 10 for instance.
#: It is a simple but efficient way to speed up CD tests, see https://github.com/SMPyBandits/SMPyBandits/issues/173
#: Default value is 0, to not use this feature, and 10 should speed up the test by x10.
LAZY_DETECT_CHANGE_ONLY_X_STEPS = 1
LAZY_DETECT_CHANGE_ONLY_X_STEPS = 10

#: XXX Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``. Default is to have ``steps_s=1``, but only using ``steps_s=2`` should already speed up by 2.
#: It is a simple but efficient way to speed up GLR tests, see https://github.com/SMPyBandits/SMPyBandits/issues/173
#: Default value is 1, to not use this feature, and 10 should speed up the test by x10.
LAZY_TRY_VALUE_S_ONLY_X_STEPS = 1
LAZY_TRY_VALUE_S_ONLY_X_STEPS = 10


#: Default value of ``use_localization`` for policies. All the experiments I tried showed that the localization always helps improving learning, so the default value is set to True.
USE_LOCALIZATION = False
USE_LOCALIZATION = True


# --- Generic GLR for 1-dimensional exponential families

eps = 1e-10  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]

# --- Simple Kullback-Leibler divergence for known distributions


def klBern(x, y):
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


def klGauss(x, y, sig2x=1):
    r""" Kullback-Leibler divergence for Gaussian distributions of means ``x`` and ``y`` and variances ``sig2x`` and ``sig2y``, :math:`\nu_1 = \mathcal{N}(x, \sigma_x^2)` and :math:`\nu_2 = \mathcal{N}(y, \sigma_x^2)`:

    .. math:: \mathrm{KL}(\nu_1, \nu_2) = \frac{(x - y)^2}{2 \sigma_y^2} + \frac{1}{2}\left( \frac{\sigma_x^2}{\sigma_y^2} - 1 \log\left(\frac{\sigma_x^2}{\sigma_y^2}\right) \right).

    See https://en.wikipedia.org/wiki/Normal_distribution#Other_properties
    """
    return (x - y) ** 2 / (2. * sig2x)


def threshold_GaussianGLR(t, horizon=None, delta=None, variant=None):
    r""" Compute the value :math:`c from the corollary of of Theorem 2 from ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].

    - The threshold is computed as (with :math:`t_0 = 0`):

    .. math:: \beta(t_0, t, \delta) := \left(1 + \frac{1}{t - t_0 + 1}\right) 2 \log\left(\frac{2 (t - t_0) \sqrt{(t - t_0) + 2}}{\delta}\right).
    """
    if delta is None:
        delta = 1.0 / int(max(1, horizon))
    c = (1 + (1.0 / t)) * log((2 * t**(3/2)) / delta)
    if c < 0 or isinf(c):
        c = float('+inf')
    return c


# --- Intermediate functions to define the optimal threshold for Bernoulli GLR tests

def function_h(u):
    r""" The function :math:`h(u) = u - \log(u)`."""
    if u <= 1:
        raise ValueError("Error: the function h only accepts values larger than 1, not x = {}".format(u))
    return u - log(u)

from scipy.optimize import root_scalar
from scipy.special import lambertw
from math import exp

def function_h_minus_one(x):
    r""" The inverse function of :math:`h(u)`, that is :math:`h^{-1}(x) = u \Leftrightarrow h(u) = x`. It is given by the Lambert W function, see :func:`scipy.special.lambertw`:

    .. math:: h^{-1}(x) = - \mathcal{W}(- \exp(-x)).

    - Example:

    >>> np.random.seed(105)
    >>> y = np.random.randn() ** 2
    >>> print(f"y = {y}")
    y = 0.060184682907834595
    >>> x = function_h(y)
    >>> print(f"h(y) = {x}")
    h(y) = 2.8705220786966508
    >>> z = function_h_minus_one(x)
    >>> print(f"h^-1(x) = {z}")
    h^-1(x) = 0.060184682907834595
    >>> assert np.isclose(z, y), "Error: h^-1(h(y)) = z = {z} should be very close to y = {}...".format(z, y)
    """
    if x <= 1:
        raise ValueError("Error: the function h inverse only accepts values larger than 1, not x = {}".format(x))
    sol = root_scalar(lambda u: function_h(u) - x, x0=x, x1=2*x)
    if sol.converged:
        return sol.root
    else:
        z = - lambertw(- exp(- x))
        return z.real

#: The constant :math:`\frac{3}{2}`, used in the definition of functions :math:`h`, :math:`h^{-1}`, :math:`\tilde{h}` and :math:`\mathcal{T}`.
constant_power_function_h = 3.0 / 2.0

#: The constant :math:`h^{-1}(1/\log(\frac{3}{2}))`, used in the definition of function :math:`\tilde{h}`.
threshold_function_h_tilde = function_h_minus_one(1 / log(constant_power_function_h))

#: The constant :math:`\log(\log(\frac{3}{2}))`, used in the definition of function :math:`\tilde{h}`.
constant_function_h_tilde = log(log(constant_power_function_h))

def function_h_tilde(x):
    r""" The function :math:`\tilde{h}(x)`, defined by:

    .. math::

        \tilde{h}(x) = \begin{cases} e^{1/h^{-1}(x)} h^{-1}(x) & \text{ if } x \ge h^{-1}(1/\ln (3/2)), \\
        (3/2) (x-\ln \ln (3/2)) & \text{otherwise}. \end{cases}
    """
    if x >= threshold_function_h_tilde:
        y = function_h_minus_one(x)
        return exp(1 / y) * y
    else:
        return constant_power_function_h * (x - constant_function_h_tilde)

#: The constant :math:`\zeta(2) = \frac{\pi^2}{6}`.
zeta_of_two = np.pi**2 / 6
# import scipy.special
# assert np.isclose(scipy.special.zeta(2), zeta_of_two)

constant_function_T_mathcal = log(2 * zeta_of_two)

def function_T_mathcal(x):
    r""" The function :math:`\mathcal{T}(x)`, defined by:

    .. math:: \mathcal{T}(x) = 2 \tilde h\left(\frac{h^{-1}(1+x) + \ln(2\zeta(2))}{2}\right).
    """
    return 2 * function_h_tilde((function_h_minus_one(1 + x) + constant_function_T_mathcal) / 2.0)

def approximation_function_T_mathcal(x):
    r""" An efficiently computed approximation of :math:`\mathcal{T}(x)`, valid for :math:`x \geq 5`:

    .. math:: \mathcal{T}(x) \simeq x + 4 \log(1 + x + \sqrt(2 x)).
    """
    return x + 4 * log(1 + x + sqrt(2 * x))


def threshold_BernoulliGLR(t, horizon=None, delta=None, variant=None):
    r""" Compute the value :math:`c` from the corollary of of Theorem 2 from ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].

    .. warning:: This is still experimental, you can try different variants of the threshold function:

    - Variant #0 (*default*) is:

    .. math:: \beta(t, \delta) := \log\left(\frac{3 t^{3/2}}{\delta}\right) = \log(\frac{1}{\delta}) + \log(3) + 3/2 \log(t).

    - Variant #1 is smaller:

    .. math:: \beta(t, \delta) := \log(\frac{1}{\delta}) + \log(1 + \log(t)).

    - Variant #2 is using :math:`\mathcal{T}`:

    .. math:: \beta(t, \delta) := 2 \mathcal{T}\left(\frac{\log(2 t^{3/2}) / \delta}{2}\right) + 6 \log(1 + \log(t)).

    - Variant #3 is using :math:`\tilde{\mathcal{T}}(x) = x + 4 \log(1 + x + \sqrt{2x})` an approximation of :math:`\mathcal{T}(x)` (valid and quite accurate as soon as :math:`x \geq 5`):

    .. math:: \beta(t, \delta) := 2 \tilde{\mathcal{T}}\left(\frac{\log(2 t^{3/2}) / \delta}{2}\right) + 6 \log(1 + \log(t)).
    """
    if delta is None:
        delta = 1.0 / sqrt(horizon)
    # c = -log(delta) + log(1 + log(s)) + log(1 + log(t-s))  # XXX no longer possible
    # c = -log(delta) + log(s) + log(t-s)  # XXX no longer possible
    if variant is not None:
        if variant == 0:
            c = -log(delta) + (3/2) * log(t) + log(3)
        elif variant == 1:
            c = -log(delta) + log(1 + log(t))
        elif variant == 2:
            c = 2 * function_T_mathcal(log(2 * t**(constant_power_function_h) / delta) / 2) + 6 * log(1 + log(t))
        elif variant == 3:
            c = 2 * approximation_function_T_mathcal(log(2 * t**(constant_power_function_h) / delta) / 2) + 6 * log(1 + log(t))
    else:
        c = -log(delta) + (3/2) * log(t) + log(3)
    if c < 0 or isinf(c):
        c = float('+inf')
    return c


EXPONENT_BETA = 1.01  #: The default value of parameter :math:`\beta` for the function :func:`decreasing_alpha__GLR`.
ALPHA_T1 = 0.05  #: The default value of parameter :math:`\alpha_{t=1}` for the function :func:`decreasing_alpha__GLR`.


def decreasing_alpha__GLR(alpha0=None, t=1, exponentBeta=EXPONENT_BETA, alpha_t1=ALPHA_T1):
    r""" Either use a fixed alpha, or compute it with an exponential decay (if ``alpha0=None``).

    .. note:: I am currently exploring the following variant (November 2018):

        - The probability of uniform exploration, :math:`\alpha`, is computed as a function of the current time:

        .. math:: \forall t>0, \alpha = \alpha_t := \alpha_{t=1} \frac{1}{\max(1, t^{\beta})}.

        - with :math:`\beta > 1, \beta` = ``exponentBeta`` (=1.05) and :math:`\alpha_{t=1} < 1, \alpha_{t=1}` = ``alpha_t1`` (=0.01).
        - the only requirement on :math:`\alpha_t` seems to be that `\sum_{t=1}^T \alpha_t < +\infty` (ie. be finite), which is the case for :math:`\alpha_t = \alpha = \frac{1}{T}`, but also any :math:`\alpha_t = \frac{\alpha_1}{t^{\beta}}` for any :math:`\beta>1` (cf. Riemann series).
    """
    assert exponentBeta > 1.0, "Error: decreasing_alpha__GLR should have a exponentBeta > 1 but it was given = {}...".format(exponentBeta)  # DEBUG
    if alpha0 is not None:
        return alpha0
    return alpha_t1 / max(1, t)**exponentBeta


DELTA0_SCALE_FACTOR = 1.0

def smart_delta_from_T_UpsilonT(horizon=1, max_nb_random_events=1, scaleFactor=DELTA0_SCALE_FACTOR, per_arm_restart=PER_ARM_RESTART, nbArms=1):
    r""" Compute a smart estimate of the optimal value for the confidence level :math:`\delta`, with ``scaleFactor`` :math:`= \delta_0\in(0,1)` a constant.

    - If ``per_arm_restart`` is True (**Local** option):

    .. math:: \delta = \frac{\delta_0}{\sqrt{K \Upsilon_T T}.

    - If ``per_arm_restart`` is False (**Global** option):

    .. math:: \delta = \frac{\delta_0}{\sqrt{\Upsilon_T T}.

    Note that if :math:`\Upsilon_T` is unknown, it is assumed to be :math:`\Upsilon_T=1`.
    """
    if max_nb_random_events is None: max_nb_random_events = 1
    product = max_nb_random_events * float(horizon)
    if per_arm_restart:
        product *= nbArms
    if product > 1:
        print("Error: bound Upsilon_T = {} should be smaller than horizon T = {}...".format(max_nb_random_events, horizon))  # DEBUG
        product = 0.1
    delta = scaleFactor / sqrt(product)
    print("DEBUG: smart_delta_from_T_UpsilonT: horizon = {}, max_nb_random_events = {}, gives delta = {}...".format(horizon, max_nb_random_events, delta))  # DEBUG
    return delta


def smart_alpha_from_T_UpsilonT(horizon=1, max_nb_random_events=1, scaleFactor=ALPHA0_SCALE_FACTOR, per_arm_restart=PER_ARM_RESTART, nbArms=1):
    r""" Compute a smart estimate of the optimal value for the *fixed* or *random* forced exploration probability :math:`\alpha` (or tracking based), with ``scaleFactor`` :math:`= \alpha_0\in(0,1)` a constant.

    - If ``per_arm_restart`` is True (**Local** option):

    .. math:: \alpha = \alpha_0 \times \sqrt{\frac{K \Upsilon_T}{T} \log(T)}.

    - If ``per_arm_restart`` is False (**Global** option):

    .. math:: \alpha = \alpha_0 \times \sqrt{\frac{\Upsilon_T}{T} \log(T)}.

    Note that if :math:`\Upsilon_T` is unknown, it is assumed to be :math:`\Upsilon_T=1`.
    """
    if max_nb_random_events is None: max_nb_random_events = 1
    ratio = max_nb_random_events / float(horizon)
    if per_arm_restart:
        ratio *= nbArms
    assert 0 < ratio <= 1, "Error: Upsilon_T = {} should be smaller than horizon T = {}...".format(max_nb_random_events, horizon)  # DEBUG
    alpha = scaleFactor * sqrt(ratio * log(horizon))
    print("DEBUG: smart_alpha_from_T_UpsilonT: horizon = {}, max_nb_random_events = {}, gives alpha = {}...".format(horizon, max_nb_random_events, alpha))  # DEBUG
    return alpha


class GLR_IndexPolicy(CD_IndexPolicy):
    r""" The GLR-UCB generic policy for non-stationary bandits, using the Generalized Likelihood Ratio test (GLR), for 1-dimensional exponential families.

    - It works for any 1-dimensional exponential family, you just have to give a ``kl`` function.
    - For instance :func:`kullback.klBern`, for Bernoulli distributions, gives :class:`GaussianGLR_IndexPolicy`,
    - And :func:`kullback.klGauss` for univariate Gaussian distributions, gives :class:`BernoulliGLR_IndexPolicy`.

    - ``threshold_function`` computes the threshold :math:`\beta(t, \delta)`, it can be for instance :func:`threshold_GaussianGLR` or :func:`threshold_BernoulliGLR`.

    - From ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    def __init__(self, nbArms,
            horizon=None, delta=None, max_nb_random_events=None,
            kl=klGauss,
            alpha0=None, exponentBeta=EXPONENT_BETA, alpha_t1=ALPHA_T1,
            threshold_function=threshold_BernoulliGLR, variant=None,
            use_increasing_alpha=False,
            lazy_try_value_s_only_x_steps=LAZY_TRY_VALUE_S_ONLY_X_STEPS,
            per_arm_restart=PER_ARM_RESTART,
            use_localization=USE_LOCALIZATION,
            *args, **kwargs
        ):
        super(GLR_IndexPolicy, self).__init__(nbArms, epsilon=1, per_arm_restart=per_arm_restart, *args, **kwargs)
        # New parameters
        self.horizon = horizon  #: The horizon :math:`T`.
        self.max_nb_random_events = max_nb_random_events  #: The number of breakpoints :math:`\Upsilon_T`.
        self.use_localization = use_localization  #: experiment to use localization of the break-point, ie, restart memory of arm by keeping observations s+1...n instead of just the last one
        # if delta is None and horizon is not None: delta = 1.0 / horizon
        self._exponentBeta = exponentBeta
        self._alpha_t1 = alpha_t1
        delta = delta if delta is not None else 1.0
        alpha = alpha0 if alpha0 is not None else 1.0
        if horizon is not None and max_nb_random_events is not None:
            delta *= smart_delta_from_T_UpsilonT(horizon=self.horizon, max_nb_random_events=self.max_nb_random_events, per_arm_restart=per_arm_restart, nbArms=nbArms)
            alpha *= smart_alpha_from_T_UpsilonT(horizon=self.horizon, max_nb_random_events=self.max_nb_random_events, per_arm_restart=per_arm_restart, nbArms=nbArms)
        self.delta = delta  #: The confidence level :math:`\delta`. Defaults to :math:`\delta=\frac{1}{\sqrt{T}}` if ``horizon`` is given and ``delta=None`` but :math:`\Upsilon_T` is unknown. Defaults to :math:`\delta=\frac{1}{\sqrt{\Upsilon_T T}}` if both :math:`T` and :math:`\Upsilon_T` are given (``horizon`` and ``max_nb_random_events``).
        self._alpha0 = alpha
        self._variant = variant
        self._use_increasing_alpha = use_increasing_alpha
        self._threshold_function = threshold_function
        self._args_to_kl = tuple()  # Tuple of extra arguments to give to the :attr:`kl` function.
        self.kl = kl  #: The parametrized Kullback-Leibler divergence (:math:`\mathrm{kl}(x,y) = KL(D(x),D(y))`) for the 1-dimensional exponential family :math:`x\mapsto D(x)`. Example: :func:`kullback.klBern` or :func:`kullback.klGauss`.
        self.lazy_try_value_s_only_x_steps = lazy_try_value_s_only_x_steps  #: Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``.

    def compute_threshold_h(self, t):
        """Compute the threshold :math:`h` with :attr:`_threshold_function`."""
        return self._threshold_function(t, horizon=self.horizon, delta=self.delta, variant=self._variant)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def proba_random_exploration(self):
        r"""What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time."""
        if self._alpha0 is not None:
            return self._alpha0
        elif self._use_increasing_alpha:
            ell = max(self.number_of_restart, 1)
            T = self.horizon if self.horizon is not None else self.t
            T = max(T, 1)
            alpha = min(np.sqrt(ell * np.log(T) / T), 1)
            return alpha
        else:
            smallest_time_since_last_restart = np.min(self.last_pulls)
            t = min(self.t, 2 * smallest_time_since_last_restart)
            return decreasing_alpha__GLR(alpha0=self._alpha0, t=t, exponentBeta=self._exponentBeta, alpha_t1=self._alpha_t1)

    def __str__(self):
        class_name = self.__class__.__name__
        name = "Gaussian-"
        if "Bernoulli" in class_name:
            # name = "Bernoulli-"
            name = ""
        if "Sub" in class_name:
            name = "Sub{}-".format(name)
        with_tracking = "tracking" if "WithTracking" in class_name else ""
        with_randomexploration = "random expl." if not with_tracking and "DeterministicExploration" not in class_name else ""
        variant = "" if self._variant is None else "threshold #{}".format(self._variant)
        use_increasing_alpha = r"increasing $\alpha_t$" if self._use_increasing_alpha else ""
        args = ", ".join(s for s in [
            "Local" if self._per_arm_restart else "Global",
            "Localization" if self.use_localization else "",
            # r"$\delta={:.3g}$".format(self.delta) if self.delta is not None else "", # r"$\delta=\frac{1}{\sqrt{T}}$",
            # "", # no need to print alpha as it is chosen based on horizon
            # r"$\alpha={:.3g}$".format(self._alpha0) if self._alpha0 is not None else r"decreasing $\alpha_t$",
            # r"$\alpha={:.3g}$".format(self._alpha0) if self._alpha0 is not None else r"", # r"$\alpha=\sqrt{frac{\log(T)}{T}}$",
            r"$\Delta n={}$".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "",
            r"$\Delta s={}$".format(self.lazy_try_value_s_only_x_steps) if self.lazy_try_value_s_only_x_steps != LAZY_TRY_VALUE_S_ONLY_X_STEPS else "",
            with_tracking,
            with_randomexploration,
            variant,
            use_increasing_alpha,
        ] if s)
        args = "({})".format(args) if args else ""
        policy_name = self._policy.__name__  #.replace("_forGLR", "")
        return r"{}GLR-{}{}".format(name, policy_name, args)

    def getReward(self, arm, reward):
        r""" Do as :class:`CD_UCB` to handle the new reward, and also, update the internal times of each arm for the indexes of :class:`klUCB_forGLR` (or other index policies), which use :math:`f(t - \tau_i(t))` for the exploration function of each arm :math:`i` at time :math:`t`, where :math:`\tau_i(t)` denotes the (last) restart time of the arm.
        """
        super(GLR_IndexPolicy, self).getReward(arm, reward)
        # DONE for this fix!
        if hasattr(self.policy, "t_for_each_arm"):
            # if np.any(self.t != self.t - self.last_restart_times):
            #     print("DEBUG: for {}, the default time step t = {} and the modified time steps t - tau_i(t) = {}...".format(self, self.t, self.t - self.last_restart_times))  # DEBUG
            self.policy.t_for_each_arm = self.t - self.last_restart_times
        else:
            self.policy.t = np.min(self.t - self.last_restart_times)

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the Generalized Likelihood Ratio test (GLR) and the :attr:`kl` function.

        - For each *time step* :math:`s` between :math:`t_0=0` and :math:`t`, compute:

        .. math::

            G^{\mathrm{kl}}_{t_0:s:t} = (s-t_0+1) \mathrm{kl}(\mu_{t_0,s}, \mu_{t_0,t}) + (t-s) \mathrm{kl}(\mu_{s+1,t}, \mu_{t_0,t}).

        - The change is detected if there is a time :math:`s` such that :math:`G^{\mathrm{kl}}_{t_0:s:t} > h`, where :attr:`threshold_h` is the threshold of the test,
        - And :math:`\mu_{a,b} = \frac{1}{b-a+1} \sum_{s=a}^{b} y_s` is the mean of the samples between :math:`a` and :math:`b`.

        .. warning:: This is computationally costly, so an easy way to speed up this test is to use :attr:`lazy_try_value_s_only_x_steps` :math:`= \mathrm{Step_s}` for a small value (e.g., 10), so not test for all :math:`s\in[t_0, t-1]` but only :math:`s\in[t_0, t-1], s \mod \mathrm{Step_s} = 0` (e.g., one out of every 10 steps).
        """
        data_y = self.all_rewards[arm]
        t0 = 0
        t = len(data_y)-1
        threshold_h = self.compute_threshold_h(t + 1)
        mean_all = np.mean(data_y[t0 : t+1])
        mean_before = 0.0
        mean_after = mean_all
        for s in range(t0, t):
            # XXX nope, that was a mistake: it is only true for the Gaussian kl !
            # this_kl = self.kl(mu(s+1, t), mu(s), *self._args_to_kl)
            # glr = ((s - t0 + 1) * (t - s) / (t - t0 + 1)) * this_kl
            # FIXED this is the correct formula!
            # XXX this is not efficient we compute the same means too many times!
            # mean_before = np.mean(data_y[t0 : s+1])
            # mean_after = np.mean(data_y[s+1 : t+1])
            # DONE okay this is efficient we don't compute the same means too many times!
            y = data_y[s]
            mean_before = (s * mean_before + y) / (s + 1)
            mean_after = ((t + 1 - s + t0) * mean_after - y) / (t - s + t0)
            if s % self.lazy_try_value_s_only_x_steps != 0:
                continue
            if np.isclose(mean_before, mean_all) and np.isclose(mean_after, mean_all):
                continue
            kl_before = self.kl(mean_before, mean_all, *self._args_to_kl)
            kl_after  = self.kl(mean_after,  mean_all, *self._args_to_kl)
            glr = (s - t0 + 1) * kl_before + (t - s) * kl_after
            if verbose: print("  - For t0 = {}, s = {}, t = {}, the mean before mu(t0,s) = {} and the mean after mu(s+1,t) = {} and the total mean mu(t0,t) = {}, so the kl before = {} and kl after = {} and GLR = {}, compared to c = {}...".format(t0, s, t, mean_before, mean_after, mean_all, kl_before, kl_after, glr, threshold_h))
            if glr >= threshold_h:
                return True, t0 + s + 1 if self.use_localization else None
        return False, None


class GLR_IndexPolicy_WithTracking(GLR_IndexPolicy):
    """ A variant of the GLR policy where the exploration is not forced to be uniformly random but based on a tracking of arms that haven't been explored enough (with a tracking).

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    def choice(self):
        r""" If any arm is not explored enough (:math:`n_k \leq \frac{\alpha}{K} \times (t - n_k)`, play uniformly at random one of these arms, otherwise, pass the call to :meth:`choice` of the underlying policy.
        """
        number_of_explorations = self.last_pulls
        min_number_of_explorations = self.proba_random_exploration * (self.t - self.last_restart_times) / self.nbArms
        not_explored_enough = np.where(number_of_explorations <= min_number_of_explorations)[0]
        # TODO check numerically what I want to prove mathematically
        # for arm in range(self.nbArms):
        #     if number_of_explorations[arm] > 0:
        #         assert number_of_explorations[arm] >= self.proba_random_exploration * (self.t - self.last_restart_times[arm]) / self.nbArms**2, "Error: for arm k={}, the number of exploration n_k(t) = {} was not >= alpha={} / K={}**2 * (t={} - tau_k(t)={}) and RHS was = {}...".format(arm, number_of_explorations[arm], self.proba_random_exploration, self.nbArms, self.t, self.last_restart_times[arm], self.proba_random_exploration * (self.t - self.last_restart_times[arm]) / self.nbArms**2)  # DEBUG
        if len(not_explored_enough) > 0:
            return np.random.choice(not_explored_enough)
        return self.policy.choice()


class GLR_IndexPolicy_WithDeterministicExploration(GLR_IndexPolicy):
    r""" A variant of the GLR policy where the exploration is not forced to be uniformly random but deterministic, inspired by what M-UCB proposed.

    - If :math:`t` is the current time and :math:`\tau` is the latest restarting time, then uniform exploration is done if:

    .. math::

        A &:= (t - \tau) \mod \lceil \frac{K}{\gamma} \rceil,\\
        A &\leq K \implies A_t = A.

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    def choice(self):
        r""" For some time steps, play uniformly at random one of these arms, otherwise, pass the call to :meth:`choice` of the underlying policy.
        """
        latest_restart_times = np.max(self.last_restart_times)
        if self.proba_random_exploration > 0:
            A = (self.t - latest_restart_times) % int(np.ceil(self.nbArms / self.proba_random_exploration))
            if A < self.nbArms:
                return int(A)
        return self.policy.choice()


# --- GLR for sigma=1 Gaussian
class GaussianGLR_IndexPolicy(GLR_IndexPolicy):
    r""" The GaussianGLR-UCB policy for non-stationary bandits, for fixed-variance Gaussian distributions (ie, :math:`\sigma^2`=``sig2`` known and fixed).
    """
    def __init__(self, nbArms, sig2=0.25, kl=klGauss, threshold_function=threshold_GaussianGLR, *args, **kwargs):
        super(GaussianGLR_IndexPolicy, self).__init__(nbArms, kl=kl, threshold_function=threshold_function, *args, **kwargs)
        self._sig2 = sig2  #: Fixed variance :math:`\sigma^2` of the Gaussian distributions. Extra parameter given to :func:`kullback.klGauss`. Default to :math:`\sigma^2 = \frac{1}{4}`.
        self._args_to_kl = (sig2, )


class GaussianGLR_IndexPolicy_WithTracking(GLR_IndexPolicy_WithTracking, GaussianGLR_IndexPolicy):
    """ A variant of the GaussianGLR-UCB policy where the exploration is not forced to be uniformly random but based on a tracking of arms that haven't been explored enough.
    """
    pass

class GaussianGLR_IndexPolicy_WithDeterministicExploration(GLR_IndexPolicy_WithDeterministicExploration, GaussianGLR_IndexPolicy):
    """ A variant of the GaussianGLR-UCB policy where the exploration is not forced to be uniformly random but deterministic, inspired by what M-UCB proposed.
    """
    pass


# --- GLR for Bernoulli
class BernoulliGLR_IndexPolicy(GLR_IndexPolicy):
    r""" The BernoulliGLR-UCB policy for non-stationary bandits, for Bernoulli distributions.

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    def __init__(self, nbArms, kl=klBern, threshold_function=threshold_BernoulliGLR, *args, **kwargs):
        super(BernoulliGLR_IndexPolicy, self).__init__(nbArms, kl=kl, threshold_function=threshold_function, *args, **kwargs)


class BernoulliGLR_IndexPolicy_WithTracking(GLR_IndexPolicy_WithTracking, BernoulliGLR_IndexPolicy):
    """ A variant of the BernoulliGLR-UCB policy where the exploration is not forced to be uniformly random but based on a tracking of arms that haven't been explored enough.

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    pass

class BernoulliGLR_IndexPolicy_WithDeterministicExploration(GLR_IndexPolicy_WithDeterministicExploration, BernoulliGLR_IndexPolicy):
    """ A variant of the BernoulliGLR-UCB policy where the exploration is not forced to be uniformly random but deterministic, inspired by what M-UCB proposed.

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    pass


# --- GLR for sigma=1 Gaussian
class OurGaussianGLR_IndexPolicy(GLR_IndexPolicy):
    r""" The GaussianGLR-UCB policy for non-stationary bandits, for fixed-variance Gaussian distributions (ie, :math:`\sigma^2`=``sig2`` known and fixed), but with our threshold designed for the sub-Bernoulli case.

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    def __init__(self, nbArms, sig2=0.25, kl=klGauss, threshold_function=threshold_BernoulliGLR, *args, **kwargs):
        super(OurGaussianGLR_IndexPolicy, self).__init__(nbArms, kl=kl, threshold_function=threshold_function, *args, **kwargs)
        self._sig2 = sig2  #: Fixed variance :math:`\sigma^2` of the Gaussian distributions. Extra parameter given to :func:`kullback.klGauss`. Default to :math:`\sigma^2 = \frac{1}{4}`.
        self._args_to_kl = (sig2, )


class OurGaussianGLR_IndexPolicy_WithTracking(GLR_IndexPolicy_WithTracking, OurGaussianGLR_IndexPolicy):
    """ A variant of the GaussianGLR-UCB policy where the exploration is not forced to be uniformly random but based on a tracking of arms that haven't been explored enough, but with our threshold designed for the sub-Bernoulli case, but with our threshold designed for the sub-Bernoulli case.

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    pass

class OurGaussianGLR_IndexPolicy_WithDeterministicExploration(GLR_IndexPolicy_WithDeterministicExploration, OurGaussianGLR_IndexPolicy):
    """ A variant of the GaussianGLR-UCB policy where the exploration is not forced to be uniformly random but deterministic, inspired by what M-UCB proposed, but with our threshold designed for the sub-Bernoulli case.

    - Reference: [["Combining the Generalized Likelihood Ratio Test and kl-UCB for Non-Stationary Bandits. E. Kaufmann and L. Besson, 2019]](https://hal.inria.fr/hal-02006471/)
    """
    pass


# --- Non-Parametric Sub-Gaussian GLR for Sub-Gaussian data

#: Default confidence level for :class:`SubGaussianGLR_IndexPolicy`.
SubGaussianGLR_DELTA = 0.01

#: By default, :class:`SubGaussianGLR_IndexPolicy` assumes distributions are 0.25-sub Gaussian, like Bernoulli or any distributions with support on :math:`[0,1]`.
SubGaussianGLR_SIGMA = 0.25

#: Whether to use the joint or disjoint threshold function (:func:`threshold_SubGaussianGLR_joint` or :func:`threshold_SubGaussianGLR_disjoint`) for :class:`SubGaussianGLR_IndexPolicy`.
SubGaussianGLR_JOINT = True

def threshold_SubGaussianGLR_joint(s, t, delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA):
    r""" Compute the threshold :math:`b^{\text{joint}}_{t_0}(s,t,\delta) according to this formula:

    .. math:: b^{\text{joint}}_{t_0}(s,t,\delta) := \sigma \sqrt{ \left(\frac{1}{s-t_0+1} + \frac{1}{t-s}\right) \left(1 + \frac{1}{t-t_0+1}\right) 2 \log\left( \frac{2(t-t_0)\sqrt{t-t_0+2}}{\delta} \right)}.
    """
    return sigma * sqrt(
        (1.0 / (s + 1) + 1.0 / (t - s)) * (1.0 + 1.0/(t + 1))
        * 2 * max(0, log(( 2 * t * sqrt(t + 2)) / delta ))
    )

def threshold_SubGaussianGLR_disjoint(s, t, delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA):
    r""" Compute the threshold :math:`b^{\text{disjoint}}_{t_0}(s,t,\delta)` according to this formula:

    .. math:: b^{\text{disjoint}}_{t_0}(s,t,\delta) := \sqrt{2} \sigma \sqrt{\frac{1 + \frac{1}{s - t_0 + 1}}{s - t_0 + 1} \log\left( \frac{4 \sqrt{s - t_0 + 2}}{\delta}\right)} + \sqrt{\frac{1 + \frac{1}{t - s + 1}}{t - s + 1} \log\left( \frac{4 (t - t_0) \sqrt{t - s + 1}}{\delta}\right)}.
    """
    return sqrt(2) * sigma * (sqrt(
        ((1.0 + (1.0 / (s + 1))) / (s + 1)) * max(0, log( (4 * sqrt(s + 2)) / delta ))
    ) + sqrt(
        ((1.0 + (1.0 / (t - s + 1))) / (t - s + 1)) * max(0, log( (4 * t * sqrt(t - s + 1)) / delta ))
    ))

def threshold_SubGaussianGLR(s, t, delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA, joint=SubGaussianGLR_JOINT):
    r""" Compute the threshold :math:`b^{\text{joint}}_{t_0}(s,t,\delta)` or :math:`b^{\text{disjoint}}_{t_0}(s,t,\delta)`."""
    if joint:
        return threshold_SubGaussianGLR_joint(s, t, delta=delta, sigma=sigma)
    else:
        return threshold_SubGaussianGLR_disjoint(s, t, delta=delta, sigma=sigma)


class SubGaussianGLR_IndexPolicy(CD_IndexPolicy):
    r""" The SubGaussianGLR-UCB policy for non-stationary bandits, using the Generalized Likelihood Ratio test (GLR), for sub-Gaussian distributions.

    - It works for any sub-Gaussian family of distributions, being :math:`\sigma^2`-sub Gaussian *with known* :math:`\sigma`.

    - From ["Sequential change-point detection: Laplace concentration of scan statistics and non-asymptotic delay bounds", O.-A. Maillard, 2018].
    """
    def __init__(self, nbArms,
            horizon=None, max_nb_random_events=None,
            full_restart_when_refresh=FULL_RESTART_WHEN_REFRESH,
            policy=DefaultPolicy,
            delta=SubGaussianGLR_DELTA, sigma=SubGaussianGLR_SIGMA, joint=SubGaussianGLR_JOINT,
            exponentBeta=1.05, alpha_t1=0.1, alpha0=None,
            lazy_detect_change_only_x_steps=LAZY_DETECT_CHANGE_ONLY_X_STEPS,
            lazy_try_value_s_only_x_steps=LAZY_TRY_VALUE_S_ONLY_X_STEPS,
            use_localization=USE_LOCALIZATION,
            *args, **kwargs
        ):
        super(SubGaussianGLR_IndexPolicy, self).__init__(nbArms, epsilon=1, full_restart_when_refresh=full_restart_when_refresh, policy=policy, lazy_detect_change_only_x_steps=lazy_detect_change_only_x_steps, *args, **kwargs)
        # New parameters
        self.horizon = horizon  #: The horizon :math:`T`.
        self.max_nb_random_events = max_nb_random_events  #: The number of breakpoints :math:`\Upsilon_T`.
        if delta is None and horizon is not None: delta = 1.0 / horizon
        self.delta = delta  #: The confidence level :math:`\delta`. Defaults to :math:`\delta=\frac{1}{T}` if ``horizon`` is given and ``delta=None``.
        self.sigma = sigma  #: Parameter :math:`\sigma` for the Sub-Gaussian-GLR test.
        self.joint = joint  #: Parameter ``joint`` for the Sub-Gaussian-GLR test.
        self._exponentBeta = exponentBeta
        self._alpha_t1 = alpha_t1
        alpha = alpha0 if alpha0 is not None else 1
        if horizon is not None and max_nb_random_events is not None:
            alpha *= smart_alpha_from_T_UpsilonT(horizon=self.horizon, max_nb_random_events=self.max_nb_random_events)
        self._alpha0 = alpha
        self.lazy_try_value_s_only_x_steps = lazy_try_value_s_only_x_steps  #: Be lazy and try to detect changes for :math:`s` taking steps of size ``steps_s``.
        self.use_localization = use_localization  #: experiment to use localization of the break-point, ie, restart memory of arm by keeping observations s+1...n instead of just the last one

    def compute_threshold_h(self, s, t):
        """Compute the threshold :math:`h` with :func:`threshold_SubGaussianGLR`."""
        return threshold_SubGaussianGLR(s, t, delta=self.delta, sigma=self.sigma, joint=self.joint)

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def proba_random_exploration(self):
        r"""What they call :math:`\alpha` in their paper: the probability of uniform exploration at each time."""
        if self._alpha0 is not None:
            return self._alpha0
        smallest_time_since_last_restart = np.min(self.last_pulls)
        t = min(self.t, 2 * smallest_time_since_last_restart)
        return decreasing_alpha__GLR(alpha0=self._alpha0, t=t, exponentBeta=self._exponentBeta, alpha_t1=self._alpha_t1)

    def __str__(self):
        args0 = r"$\delta={}$, $\sigma={:.3g}$".format("{:.3g}".format(self.delta) if self.delta else "1/T", self.sigma)
        args1 = "{}{}".format(
            "joint" if self.joint else "disjoint",
            "" if self._per_arm_restart else ", Global"
            ", Localisation" if self.use_localization else ""
        )
        args2 = "{}{}{}".format(
            r"$\alpha={:.3g}$".format(self._alpha0) if self._alpha0 is not None else r"decreasing $\alpha_t$",
            ", lazy detect {}".format(self.lazy_detect_change_only_x_steps) if self.lazy_detect_change_only_x_steps != LAZY_DETECT_CHANGE_ONLY_X_STEPS else "",
            ", lazy s {}".format(self.lazy_try_value_s_only_x_steps) if self.lazy_try_value_s_only_x_steps != LAZY_TRY_VALUE_S_ONLY_X_STEPS else "")
        return r"SubGaussian-GLR-{}({}, {}, {})".format(self._policy.__name__, args0, args1, args2)

    def detect_change(self, arm, verbose=VERBOSE):
        r""" Detect a change in the current arm, using the non-parametric sub-Gaussian Generalized Likelihood Ratio test (GLR) works like this:

        - For each *time step* :math:`s` between :math:`t_0=0` and :math:`t`, compute:

        .. math:: G^{\text{sub-}\sigma}_{t_0:s:t} = |\mu_{t_0,s} - \mu_{s+1,t}|.

        - The change is detected if there is a time :math:`s` such that :math:`G^{\text{sub-}\sigma}_{t_0:s:t} > b_{t_0}(s,t,\delta)`, where :math:`b_{t_0}(s,t,\delta)` is the threshold of the test,

        - The threshold is computed as:

        .. math:: b_{t_0}(s,t,\delta) := \sigma \sqrt{ \left(\frac{1}{s-t_0+1} + \frac{1}{t-s}\right) \left(1 + \frac{1}{t-t_0+1}\right) 2 \log\left( \frac{2(t-t_0)\sqrt{t-t_0+2}}{\delta} \right)}.

        - And :math:`\mu_{a,b} = \frac{1}{b-a+1} \sum_{s=a}^{b} y_s` is the mean of the samples between :math:`a` and :math:`b`.
        """
        data_y = self.all_rewards[arm]
        t0 = 0
        t = len(data_y)-1
        mean_before = 0.0
        mean_after = np.mean(data_y)
        for s in range(t0, t):
            # XXX this is not efficient we compute the same means too many times!
            # mean_before = np.mean(data_y[t0 : s+1])
            # mean_after = np.mean(data_y[s+1 : t+1])
            # DONE okay this is efficient we don't compute the same means too many times!
            y = data_y[s]
            mean_before = (s * mean_before + y) / (s + 1)
            mean_after = ((t + 1 - s + t0) * mean_after - y) / (t - s + t0)
            if s % self.lazy_try_value_s_only_x_steps != 0:
                continue
            glr = abs(mean_after - mean_before)
            # compute threshold
            threshold_h = self.compute_threshold_h(s, t)
            if verbose: print("  - For t0 = {}, s = {}, t = {}, the mean mu(t0,s) = {} and mu(s+1,t) = {} so glr = {}, compared to c = {}...".format(t0, s, t, mean_before, mean_after, glr, threshold_h))
            if glr >= threshold_h:
                return True, t0 + s + 1 if self.use_localization else None
        return False, None
