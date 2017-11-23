#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
r""" Experimental code to perform complete tree exploration for Multi-Player bandits.

Algorithms:

- Support Selfish 0-greedy, UCB, and klUCB in 3 different variants.
- Support also RhoRand, RandTopM and MCTopM, even though they are *not* memory-less, by using another state representation (inlining the memory of each player, eg the ranks for RhoRand).

Features:

- For the means of each arm, :math:`\mu_1, \dots, \mu_K`, this script can use exact formal computations with sympy, or fractions with Fraction, or float number.
- The graph can contain all nodes from root to leafs, or only leafs (with summed probabilities), and possibly only the absorbing nodes are showed.
- Support export of the tree to a GraphViz dot graph, and can save it to SVG/PNG and LaTeX (with Tikz) and PDF etc.
- By default, the root is highlighted in green and the absorbing nodes are in red.

.. warning:: I still have to fix these issues:

   - TODO : right now, it is not so efficient, could it be improved? I don't think I can do anything in a smarter way, in pure Python.


Requirements:

- 'sympy' module to use formal means :math:`\mu_1, \dots, \mu_K` instead of numbers,
- 'numpy' module for computations on indexes (e.g., ``np.where``),
- 'graphviz' module to generate the graph and save it,
- 'dot2tex' module to generate nice LaTeX (with Tikz) graph and save it to PDF.

.. note::

   To use the 'dot2tex' module, only Python2 is supported.
   However, I maintain an unpublished port of 'dot2tex' for Python3, see
   [here](https://github.com/Naereen/dot2tex), that you can download, and install
   manually (sudo python3 setup.py install) to have 'dot2tex' for Python3 also.

About:

- *Date:* 16/09/2017.
- *Author:* Lilian Besson, (C) 2017
- *Licence:* MIT Licence (http://lbesson.mit-license.org).
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"

from collections import Counter, deque
from fractions import Fraction
from itertools import product
from os import getenv, chdir, getcwd
from os.path import join as os_path_join
from os.path import dirname, basename
from re import sub as re_sub
from textwrap import wrap
import subprocess

try:
    import numpy as np
except ImportError as e:
    print("Warning: the 'numpy' module was not found...\nInstall it with 'sudo pip2/pip3 install numpy' or your system packet manager (eg. 'sudo apt install python3-numpy')")  # XXX
    raise e
try:
    import sympy
except ImportError:
    print("Warning: the 'sympy' module was not found...\nSymbolic computations cannot be performed without sympy.\nInstall it with 'sudo pip2/pip3 install sympy' or your system packet manager (eg. 'sudo apt install python3-sympy')")  # XXX
try:
    from graphviz import Digraph
except ImportError:
    print("Warning: the 'graphviz' module was not found...\nTrees cannot be saved or displayed without graphviz.\nInstall it with 'sudo pip2/pip3 install graphviz'")  # XXX
try:
    # if version_info.major < 3:
    from dot2tex import dot2tex
except ImportError:
    print("Warning: the 'dot2tex' module was not found...\nTrees cannot be saved to LaTeX and PDF formats.\nInstall it with 'sudo pip2 install dot2tex' (require Python 2)\nOr install it from https://github.com/Naereen/dot2tex for Python 3.")  # XXX

oo = float('+inf')  #: Shortcut for float('+inf').

from sys import version_info
if version_info.major < 3:  # Python 2 compatibility if needed
    input = raw_input
    import codecs
    def open(filename, mode):  # https://docs.python.org/3/library/codecs.html#standard-encodings
        return codecs.open(filename, mode=mode, encoding='utf_8')

PLOT_DIR = os_path_join("plots", "trees")  #: Directory for the plots

from Arms.usenumba import jit

def tupleit1(anarray):
    """Convert a non-hashable 1D numpy array to a hashable tuple."""
    return tuple(list(anarray))

def tupleit2(anarray):
    """Convert a non-hashable 2D numpy array to a hashable tuple-of-tuples."""
    return tuple([tuple(r) for r in list(anarray)])

def prod(iterator):
    """Product of the values in this iterator."""
    p = 1
    for v in iterator:
        p *= v
    return p


WIDTH = 200  #: Default value for the ``width`` parameter for :func:`wraptext` and :func:`wraplatex`.

def wraptext(text, width=WIDTH):
    """ Wrap the text, using ``textwrap`` module, and ``width``."""
    return '\n'.join(wrap(text, width=width))


def mybool(s):
    return False if s == 'False' else bool(s)

ONLYLEAFS = True  #: By default, aim at the most concise graph representation by only showing the leafs.
ONLYLEAFS = mybool(getenv('ONLYLEAFS', ONLYLEAFS))

ONLYABSORBING = False  #: By default, don't aim at the most concise graph representation by only showing the absorbing leafs.
ONLYABSORBING = mybool(getenv('ONLYABSORBING', ONLYABSORBING))

CONCISE = True  #: By default, only show :math:`\tilde{S}` and :math:`N` in the graph representations, not all the 4 vectors.
CONCISE = mybool(getenv('CONCISE', CONCISE))

FULLHASH = not CONCISE  #: Use only Stilde, N for hashing the states.
FULLHASH = mybool(getenv('FULLHASH', FULLHASH))

# FORMAT = "pdf"  #: Format used to save the graphs.
FORMAT = "svg"  #: Format used to save the graphs.
FORMAT = getenv("FORMAT", FORMAT)

# --- Implement the bandit algorithms in a purely functional and memory-less flavor

@jit
def FixedArm(j, state):
    """Fake player j that always targets at arm j."""
    return [j]

@jit
def UniformExploration(j, state):
    """Fake player j that always targets all arms."""
    return list(np.arange(state.K))

@jit
def ConstantRank(j, state, decision, collision):
    """Constant rank no matter what."""
    return [state.memories[j]]

@jit
def choices_from_indexes(indexes):
    """For deterministic index policies, if more than one index is maximum, return the list of positions attaining this maximum (ties), or only one position."""
    return np.where(indexes == np.max(indexes))[0]

# --- Selfish 0-greedy variants

@jit
def Selfish_0Greedy_U(j, state):
    """Selfish policy + 0-Greedy index + U feedback."""
    indexes = state.S[j] / state.N[j]
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

@jit
def Selfish_0Greedy_Utilde(j, state):
    """Selfish policy + 0-Greedy index + Utilde feedback."""
    indexes = state.Stilde[j] / state.N[j]
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

@jit
def Selfish_0Greedy_Ubar(j, state):
    """Selfish policy + 0-Greedy index + Ubar feedback."""
    indexes = (state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

default_policy = Selfish_0Greedy_Ubar


# --- Selfish UCB variants
alpha = 0.5

@jit
def Selfish_UCB_U(j, state):
    """Selfish policy + UCB_0.5 index + U feedback."""
    indexes = (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

@jit
def Selfish_UCB(j, state):
    """Selfish policy + UCB_0.5 index + Utilde feedback."""
    indexes = (state.Stilde[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

Selfish_UCB_Utilde = Selfish_UCB

@jit
def Selfish_UCB_Ubar(j, state):
    """Selfish policy + UCB_0.5 index + Ubar feedback."""
    indexes = (state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

# default_policy = Selfish_UCB_Ubar

# --- Selfish kl UCB variants

from Policies import klucbBern
tolerance = 1e-6
klucb = np.vectorize(klucbBern)
c = 1

@jit
def Selfish_KLUCB_U(j, state):
    """Selfish policy + Bernoulli KL-UCB index + U feedback."""
    indexes = klucb(state.S[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

@jit
def Selfish_KLUCB(j, state):
    """Selfish policy + Bernoulli KL-UCB index + Utilde feedback."""
    indexes = klucb(state.Stilde[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

Selfish_KLUCB_Utilde = Selfish_KLUCB

@jit
def Selfish_KLUCB_Ubar(j, state):
    """Selfish policy + Bernoulli KL-UCB index + Ubar feedback."""
    indexes = klucb((state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]), c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

# default_policy = Selfish_KLUCB_Ubar


# --- RhoRand UCB variants

@jit
def choices_from_indexes_with_rank(indexes, rank=1):
    """For deterministic index policies, if more than one index is maximum, return the list of positions attaining the rank-th largest index (with more than one if ties, or only one position)."""
    return np.where(indexes == np.sort(indexes)[-rank])[0]

alpha = 0.5

@jit
def RhoRand_UCB_U(j, state):
    """RhoRand policy + UCB_0.5 index + U feedback."""
    rank = state.memories[j]
    indexes = (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes_with_rank(indexes, rank=rank)

@jit
def RhoRand_UCB_Utilde(j, state):
    """RhoRand policy + UCB_0.5 index + Utilde feedback."""
    rank = state.memories[j]
    indexes = (state.Stilde[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes_with_rank(indexes, rank=rank)

@jit
def RhoRand_UCB_Ubar(j, state):
    """RhoRand policy + UCB_0.5 index + Ubar feedback."""
    rank = state.memories[j]
    indexes = (state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes_with_rank(indexes, rank=rank)

@jit
def RhoRand_KLUCB_U(j, state):
    """RhoRand policy + Bernoulli KL-UCB index + U feedback."""
    rank = state.memories[j]
    indexes = klucb(state.S[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes_with_rank(indexes, rank=rank)

@jit
def RhoRand_KLUCB_Utilde(j, state):
    """RhoRand policy + Bernoulli KL-UCB index + Utilde feedback."""
    rank = state.memories[j]
    indexes = klucb(state.Stilde[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes_with_rank(indexes, rank=rank)

@jit
def RhoRand_KLUCB_Ubar(j, state):
    """RhoRand policy + Bernoulli KL-UCB index + Ubar feedback."""
    rank = state.memories[j]
    indexes = klucb((state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]), c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes_with_rank(indexes, rank=rank)

# So we need tow functions: one takes the decision, one updates the rank after all the decisions are taken

@jit
def RandomNewRank(j, state, decision, collision):
    """RhoRand chooses a new uniform rank in {1,..,M} in case of collision, or keep the same."""
    if collision:  # new random rank
        return list(np.arange(1, 1 + state.M))
    else:  # keep the same rank
        return [state.memories[j]]

default_policy, default_update_memory = RhoRand_UCB_U, RandomNewRank
# default_policy, default_update_memory = RhoRand_KLUCB_U, RandomNewRank


# --- RandTopM, MCTopM variants

@jit
def RandTopM_UCB_U(j, state, collision=False):
    """RandTopM policy + UCB_0.5 index + U feedback."""
    chosen_arm = state.memories[j]
    indexes = (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def RandTopM_UCB_Utilde(j, state, collision=False):
    """RandTopM policy + UCB_0.5 index + Utilde feedback."""
    chosen_arm = state.memories[j]
    indexes = (state.Stilde[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def RandTopM_UCB_Ubar(j, state, collision=False):
    """RandTopM policy + UCB_0.5 index + Ubar feedback."""
    chosen_arm = state.memories[j]
    indexes = (state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def RandTopM_KLUCB_U(j, state, collision=False):
    """RandTopM policy + Bernoulli KL-UCB index + U feedback."""
    chosen_arm = state.memories[j]
    indexes = klucb(state.S[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def RandTopM_KLUCB_Utilde(j, state, collision=False):
    """RandTopM policy + Bernoulli KL-UCB index + Utilde feedback."""
    chosen_arm = state.memories[j]
    indexes = klucb(state.Stilde[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def RandTopM_KLUCB_Ubar(j, state, collision=False):
    """RandTopM policy + Bernoulli KL-UCB index + Ubar feedback."""
    chosen_arm = state.memories[j]
    indexes = klucb((state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]), c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def RandTopM_RandomNewChosenArm(j, state, decision, collision):
    """RandTopM chooses a new arm after a collision or if the chosen arm lies outside of its estimatedBestArms set, uniformly from the set of estimated M best arms, or keep the same."""
    player = state.players[j]
    return player(j, state, collision=collision) if player.__defaults__ else player(j, state)

# default_policy, default_update_memory = RandTopM_UCB_U, RandTopM_RandomNewChosenArm



# --- MCTopM variants

@jit
def write_to_tuple(this_tuple, index, value):
    """Tuple cannot be written, this hack fixes that."""
    this_list = list(this_tuple)
    this_list[index] = value
    return tuple(this_list)

@jit
def MCTopM_UCB_U(j, state, collision=False):
    """MCTopM policy + UCB_0.5 index + U feedback."""
    if not isinstance(state.memories[j], tuple):  # if no sitted information yet
        state.memories = write_to_tuple(state.memories, j, (-1, False))
    assert isinstance(state.memories[j], tuple)
    chosen_arm, sitted = state.memories[j]
    indexes = (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def MCTopM_UCB_Utilde(j, state, collision=False):
    """MCTopM policy + UCB_0.5 index + Utilde feedback."""
    if not isinstance(state.memories[j], tuple):  # if no sitted information yet
        state.memories = write_to_tuple(state.memories, j, (-1, False))
    assert isinstance(state.memories[j], tuple)
    chosen_arm, sitted = state.memories[j]
    indexes = (state.Stilde[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def MCTopM_UCB_Ubar(j, state, collision=False):
    """MCTopM policy + UCB_0.5 index + Ubar feedback."""
    if not isinstance(state.memories[j], tuple):  # if no sitted information yet
        state.memories = write_to_tuple(state.memories, j, (-1, False))
    assert isinstance(state.memories[j], tuple)
    chosen_arm, sitted = state.memories[j]
    indexes = (state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def MCTopM_KLUCB_U(j, state, collision=False):
    """MCTopM policy + Bernoulli KL-UCB index + U feedback."""
    if not isinstance(state.memories[j], tuple):  # if no sitted information yet
        state.memories = write_to_tuple(state.memories, j, (-1, False))
    assert isinstance(state.memories[j], tuple)
    chosen_arm, sitted = state.memories[j]
    indexes = klucb(state.S[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def MCTopM_KLUCB_Utilde(j, state, collision=False):
    """MCTopM policy + Bernoulli KL-UCB index + Utilde feedback."""
    if not isinstance(state.memories[j], tuple):  # if no sitted information yet
        state.memories = write_to_tuple(state.memories, j, (-1, False))
    assert isinstance(state.memories[j], tuple)
    chosen_arm, sitted = state.memories[j]
    indexes = klucb(state.Stilde[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def MCTopM_KLUCB_Ubar(j, state, collision=False):
    """MCTopM policy + Bernoulli KL-UCB index + Ubar feedback."""
    if not isinstance(state.memories[j], tuple):  # if no sitted information yet
        state.memories = write_to_tuple(state.memories, j, (-1, False))
    assert isinstance(state.memories[j], tuple)
    chosen_arm, sitted = state.memories[j]
    indexes = klucb((state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]), c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    estimatedBestArms = np.argsort(indexes)[-state.M:]
    if collision or chosen_arm not in estimatedBestArms:
        return estimatedBestArms
    else:
        return [chosen_arm]

@jit
def MCTopM_RandomNewChosenArm(j, state, decision, collision):
    """RandTopMC chooses a new arm after if the chosen arm lies outside of its estimatedBestArms set, uniformly from the set of estimated M best arms, or keep the same."""
    player = state.players[j]
    chosen_arm, sitted = state.memories[j]
    if not sitted:
        if collision:  # new arm from estimatedBestArms
            chosen_arms = player(j, state, collision=collision) if player.__defaults__ else player(j, state)
            return list(zip(chosen_arms, [False] * len(chosen_arms)))
        else:  # sitted, for now
            return [(decision, True)]
    else:
        # sitted but the chair changed ==> not sitted
        return [(chosen_arm, chosen_arm == decision)]

# default_policy, default_update_memory = MCTopM_UCB_U, MCTopM_RandomNewChosenArm


# --- Generate vector of formal means mu_1,...,mu_K

def symbol_means(K):
    """Better to work directly with symbols and instantiate the results *after*."""
    return sympy.var(['mu_{}'.format(i) for i in range(1, K + 1)])

def random_uniform_means(K):
    """If needed, generate an array of K (numerical) uniform means in [0, 1]."""
    return np.random.rand(K)


def uniform_means(nbArms=3, delta=0.1, lower=0., amplitude=1.):
    """Return a list of means of arms, well spaced:

    - in [lower, lower + amplitude],
    - sorted in increasing order,
    - starting from lower + amplitude * delta, up to lower + amplitude * (1 - delta),
    - and there is nbArms arms.

    >>> np.array(uniform_means(2, 0.1))
    array([ 0.1,  0.9])
    >>> np.array(uniform_means(3, 0.1))
    array([ 0.1,  0.5,  0.9])
    >>> np.array(uniform_means(9, 1 / (1. + 9)))
    array([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
    """
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert amplitude > 0, "Error: 'amplitude' = {:.3g} has to be > 0.".format(amplitude)  # DEBUG
    assert 0. < delta < 1., "Error: 'delta' = {:.3g} has to be in (0, 1).".format(delta)  # DEBUG
    mus = lower + amplitude * np.linspace(delta, 1 - delta, nbArms)
    return sorted(list(mus))

# --- Transform probabilities to float, expr, str

def proba2float(proba, values=None, K=None, names=None):
    """Replace mu_k by a numerical value and evaluation  the formula."""
    if hasattr(proba, "evalf"):
        if values is None and K is not None:
            values = uniform_means(nbArms=K)
        if names is None:
            K = len(values)
            names = symbol_means(K)
        return proba.evalf(subs=dict(zip(names, values)))
    elif isinstance(proba, Fraction):
        return float(proba)
    else:  # a bit of str rewriting
        return proba

def simplify(proba):
    """Try to simplify the expression of the probability."""
    if hasattr(proba, "simplify"):
        return proba.simplify().factor()
    else:
        return proba

def proba2str(proba, latex=False, html_in_var_names=False):
    """Pretty print a proba, either a number, a Fraction, or a sympy expression."""
    if isinstance(proba, float):
        str_proba = "{:.3g}".format(proba)
    elif isinstance(proba, Fraction):
        str_proba = str(proba)
    else:  # a bit of str rewriting
        str_proba = str(simplify(proba))
        if latex:
            str_proba = re_sub(r'\*\*([0-9]+)', r'^{\1}', str_proba)
        elif html_in_var_names:
            str_proba = re_sub(r'\*\*([0-9]+)', r'<SUP>\1</SUP>', str_proba)
        else:
            str_proba = str_proba.replace('**', '^')
        str_proba = str_proba.replace('*', '')
        str_proba = re_sub(r'\(mu_([0-9]+) - 1\)\(mu_([0-9]+) - 1\)', r'(1-mu_\1)(1-mu_\2)', str_proba)
        str_proba = re_sub(r'-mu_([0-9]+) \+ 1', r'1-mu_\1', str_proba)
        str_proba = re_sub(r'-(.*)\(mu_([0-9]+) - 1\)', r'\1(1-mu_\2)', str_proba)
        str_proba = re_sub(r'-\(mu_([0-9]+) - 1\)', r'(1-mu_\1)', str_proba)
        if latex:  # replace mu_12 by mu_{12}
            str_proba = re_sub(r'_([0-9]+)', r'_{\1}', str_proba)
            str_proba = re_sub(r'mu_', r'\mu_', str_proba)
            str_proba = '$' + str_proba + '$'
            str_proba = str_proba.replace('\\', '\\\\')
        elif html_in_var_names:  # replace mu_12 by mu<sub>12</sub>
            str_proba = re_sub(r'_([0-9]+)', r'<SUB>\1</SUB>', str_proba)
        elif version_info.major < 3:  # to avoid dealing with unicode for Python2...
            str_proba = re_sub(r'mu_', r'm', str_proba)
        else:
            str_proba = re_sub(r'mu_', r'µ', str_proba)
    return str_proba


# --- Transform .tex to .pdf

def tex2pdf(filename):
    """Naive call to command line pdflatex, twice."""
    dir1 = getcwd()
    dir2, base = dirname(filename), basename(filename)
    print("Now compiling it to PDF with 'pdflatex {} && pdflatex {}' ...".format(base, base))
    log, gz, aux = base.replace('.tex', '.log'), base.replace('.tex', '.synctex.gz'), base.replace('.tex', '.aux')
    chdir(dir2)  # go in the plots/trees/ subdir
    if subprocess.call(["pdflatex", "-halt-on-error", base], stdout=open("/dev/null", 'w')) >= 0:
        subprocess.call(["pdflatex", "-halt-on-error", base], stdout=open("/dev/null", 'w'))
        subprocess.call(["mv", "-f", log, gz, aux, "/tmp/"])
    else:
        subprocess.call(["pdflatex", "-halt-on-error", base])
    chdir(dir1)  # go back

# --- Data representation'

class State(object):
    """Not space-efficient representation of a state in the system we model.

    - S, Stilde, N, Ntilde: are arrays of size (M, K),
    - depth, t, M, K: integers, to avoid recomputing them,
    - mus: the problem parameters (only for Bernoulli arms),
    - players: is a list of algorithms,
    - probas: list of transition probabilities,
    - children: list of all possible next states (transitions).
    """

    def __init__(self, S, Stilde, N, Ntilde, mus, players, depth=0):
        """Create a new state. Arrays S, Stilde, N, Ntilde are *copied* to avoid modify previous values!"""
        self.S = np.copy(S)  #: sensing feedback
        self.Stilde = np.copy(Stilde)  #: number of sensing trials
        self.N = np.copy(N)  #: number of succesful transmissions
        self.Ntilde = np.copy(Ntilde)  #: number of trials without collisions
        self.mus = mus  # XXX OK memory efficient: only a pointer to the (never modified) list
        self.players = players  # XXX OK memory efficient: only a pointer to the (never modified) list
        # New arguments
        self.depth = depth  #: current depth of the exploration tree
        self.t = np.sum(N[0])  #: current time step. Simply = sum(N[0]) = sum(N[i]) for all player i, but easier to compute it once and store it
        assert np.shape(S) == np.shape(Stilde) == np.shape(N) == np.shape(Ntilde), "Error: difference in shapes of S, Stilde, N, Ntilde."
        self.M = min(np.shape(S))  #: number of players
        assert len(players) == self.M, "Error: 'players' list is not of size M ..."  # DEBUG
        self.K = max(np.shape(S))  #: number of arms (channels)
        assert len(mus) == self.K, "Error: 'mus' list is not of size K ..."  # DEBUG
        self.children = []  #: list of next state, representing all the possible transitions
        self.probas = []  #: probabilities of transitions

    # --- Utility

    def __str__(self, concise=CONCISE):
        if concise:
            return "    State : M = {}, K = {} and t = {}, depth = {}.\n{} =: Stilde\n{} =: N\n".format(self.M, self.K, self.t, self.depth, self.Stilde, self.N)
        else:
            return "    State : M = {}, K = {} and t = {}, depth = {}.\n{} =: S\n{} =: Stilde\n{} =: N\n{} =: Ntilde\n".format(self.M, self.K, self.t, self.depth, self.S, self.Stilde, self.N, self.Ntilde)

    def to_node(self, concise=CONCISE):
        """Print the state as a small string to be attached to a GraphViz node."""
        if concise:
            return "[[" + "], [".join(",".join("{:.3g}/{}".format(st, n) for st, n in zip(st2, n2)) for st2, n2 in zip(self.Stilde, self.N)) + "]]"
        else:
            return "[[" + "], [".join(",".join("{:.3g}:{:.3g}/{}:{}".format(s, st, n, nt) for s, st, n, nt in zip(s2, st2, n2, nt2)) for s2, st2, n2, nt2 in zip(self.S, self.Stilde, self.N, self.Ntilde)) + "]]"

    def to_dot(self,
               title="", name="", comment="",
               latex=False, html_in_var_names=False, ext=FORMAT,
               onlyleafs=ONLYLEAFS, onlyabsorbing=ONLYABSORBING, concise=CONCISE):
        r"""Convert the state to a .dot graph, using GraphViz. See http://graphviz.readthedocs.io/ for more details.

        - onlyleafs: only print the root and the leafs, to see a concise representation of the tree.
        - onlyabsorbing: only print the absorbing leafs, to see a really concise representation of the tree.
        - concise: weather to use the short representation of states (using :math:`\tilde{S}` and :math:`N`) or the long one (using the 4 variables).
        - html_in_var_names: experimental use of ``<SUB>..</SUB>`` and ``<SUP>..</SUP>`` in the label for the tree.
        - latex: experimental use of ``_{..}`` and ``^{..}`` in the label for the tree, to use with dot2tex.
        """
        dot = Digraph(name=name, comment=comment, format=ext)
        print("\nCreating a dot graph from the tree...")
        dot.attr(overlap="false")
        if title: dot.attr(label=wraptext(title))
        node_number = 0
        if onlyleafs:
            root_name, root = "0", self
            dot.node(root_name, root.to_node(concise=concise), color="green")
            complete_probas, leafs = root.get_unique_leafs()
            if len(leafs) > 256:
                raise ValueError("Useless to save a tree with more than 256 leafs, the resulting image will be too large to be viewed.")  # DEBUG
            for proba, leaf in zip(complete_probas, leafs):
                # add a UNIQUE identifier for each node: easy, just do a breath-first search, and use numbers from 0 to big-integer-that-is-computed on the fly
                node_number += 1
                leaf_name = str(node_number)
                if leaf.is_absorbing():
                    dot.node(leaf_name, leaf.to_node(concise=concise), color="red")
                    dot.edge(root_name, leaf_name, label=proba2str(proba, latex=latex, html_in_var_names=html_in_var_names), color="red" if root.is_absorbing() else "black")
                elif not onlyabsorbing:
                    dot.node(leaf_name, leaf.to_node(concise=concise))
                    dot.edge(root_name, leaf_name, label=proba2str(proba, latex=latex, html_in_var_names=html_in_var_names), color="red" if root.is_absorbing() else "black")
        else:
            to_explore = deque([("0", self)])  # BFS using a deque, DFS using a list/recursive call
            nb_node = 1
            # convert each state to a node and a list of edge
            while len(to_explore) > 0:
                nb_node += 1
                root_name, root = to_explore.popleft()
                if root_name == "0":
                    dot.node(root_name, root.to_node(concise=concise), color="green")
                elif root.is_absorbing():
                    dot.node(root_name, root.to_node(concise=concise), color="red")
                elif onlyabsorbing:
                    if root.has_absorbing_child_whole_subtree():
                        dot.node(root_name, root.to_node(concise=concise))
                else:
                    dot.node(root_name, root.to_node(concise=concise))
                for proba, child in zip(root.probas, root.children):
                    # add a UNIQUE identifier for each node: easy, just do a breath-first search, and use numbers from 0 to big-integer-that-is-computed on the fly
                    node_number += 1
                    child_name = str(node_number)
                    # here, if onlyabsorbing, I should only print the *paths* leading to absorbing leafs!
                    if onlyabsorbing:
                        if child.has_absorbing_child_whole_subtree():
                            dot.edge(root_name, child_name, label=proba2str(proba, latex=latex, html_in_var_names=html_in_var_names), color="red" if root.is_absorbing() else "black")
                    else:
                        dot.edge(root_name, child_name, label=proba2str(proba, latex=latex, html_in_var_names=html_in_var_names), color="red" if root.is_absorbing() else "black")
                    to_explore.append((child_name, child))
                if nb_node > 1024:
                    raise ValueError("Useless to save a tree with more than 1024 nodes, the resulting image will be too large to be viewed.")  # DEBUG
        return dot

    def saveto(self, filename, view=True,
               title="", name="", comment="",
               latex=False, html_in_var_names=False, ext=FORMAT,
               onlyleafs=ONLYLEAFS, onlyabsorbing=ONLYABSORBING, concise=CONCISE):
        # Hack to fix the LaTeX output
        title = title.replace('_', ' ')
        name = name.replace('_', ' ')
        comment = comment.replace('_', ' ')
        dot = self.to_dot(title=title, name=name, comment=comment,
                          html_in_var_names=html_in_var_names, latex=latex, ext=ext,
                          onlyleafs=onlyleafs, onlyabsorbing=onlyabsorbing, concise=concise)
        if latex:
            source = dot.source
            if version_info.major < 3:  source = unicode(source, 'utf_8')
            # print("source =\n", source)  # DEBUG
            filename = filename.replace('.gv', '.gv.tex')
            print("Saving the dot graph to '{}'...".format(filename))
            with open(filename, 'w') as f:
                f.write(dot2tex(source, format='tikz', crop=True, figonly=False, texmode='raw'))
            tex2pdf(filename)
            filename = filename.replace('.gv.tex', '__onlyfig.gv.tex')
            print("Saving the dot graph to '{}'...".format(filename))
            with open(filename, 'w') as f:
                f.write(dot2tex(source, format='tikz', crop=True, figonly=True, texmode='raw'))
        else:
            print("Saving the dot graph to '{}.{}'...".format(filename, ext))
            dot.render(filename, view=view)
        # done for saving the graph

    def copy(self):
        """Get a new copy of that state with same S, Stilde, N, Ntilde but no probas and no children (and depth=0)."""
        return State(S=self.S, Stilde=self.Stilde, N=self.N, Ntilde=self.Ntilde, mus=self.mus, players=self.players, depth=self.depth)

    def __hash__(self, full=FULLHASH):
        """Hash the matrix Stilde and N of the state."""
        if full:
            return hash(tupleit2(self.S)) + hash(tupleit2(self.N)) + hash(tupleit2(self.Stilde)) + hash(tupleit2(self.Ntilde) + (self.t, self.depth, ))
        else:
            return hash(tupleit2(self.Stilde)) + hash(tupleit2(self.N))

    def is_absorbing(self):
        """Try to detect if this state is absorbing, ie only one transition is possible, and again infinitely for the only child.

        .. warning:: Still very experimental!
        """
        # FIXME still not sure about the characterization of absorbing states
        # if at least two players have the same S, Stilde, N, Ntilde lines
        if np.min(self.N) < 1:
            return False
        for j1 in range(self.M):
            for j2 in range(j1 + 1, self.M):
                A = [self.S, self.Stilde, self.N, self.Ntilde]
                are_all_equal = [ tupleit1(a[j1]) == tupleit1(a[j2]) for a in A ]
                if all(are_all_equal):
                    # bad_line = add([tupleit1(a[j1]) for a in A])
                    # bad_line = tupleit1(self.S[j1])
                    bad_line = tupleit1(self.Stilde[j1])
                    # and if that line has only different values
                    if len(set(bad_line)) == len(bad_line):
                        return True
        return False

    def has_absorbing_child_whole_subtree(self):
        """Try to detect if this state has an absorbing child in the whole subtree. """
        if self.is_absorbing():
            return True
        else:
            return any(child.has_absorbing_child_whole_subtree() for child in self.children)

    # --- High level view of a depth-1 exploration

    def explore_from_node_to_depth(self, depth=1):
        """Compute recursively the one_depth children of the root and its children."""
        print("\nFor depth = {}, exploring from this node :\n{}".format(depth, self))  # DEBUG
        if depth == 0:
            return
        self.compute_one_depth()
        self.depth = depth
        if depth > 1:
            for child in self.children:
                child.explore_from_node_to_depth(depth=depth-1)

    def compute_one_depth(self):
        """Use all_deltas to store all the possible transitions and their probabilities. Increase depth by 1 at the end."""
        self.depth += 1
        uniq_children = dict()
        uniq_probas = dict()
        for delta, proba in self.all_deltas():
            if proba == 0: continue
            # copy the current state, apply decision of algorithms and random branching
            child = delta(self.copy())
            h = hash(child)  # I guess I could use states directly as key, but this would cost more in terms of memory
            if h in uniq_children:
                uniq_probas[h] += proba
            else:
                assert child.depth == (self.depth - 1)
                uniq_children[h] = child
                uniq_probas[h] = proba
        print("   at depth {} we saw {} different unique children states...".format(self.t, len(uniq_children)))
        self.probas = [simplify(p) for p in uniq_probas.values()]
        self.children = list(uniq_children.values())
        # Done for computing all the children and probability of transitions

    def all_absorbing_states(self, depth=1):
        """Generator that yields all the absorbing nodes of the tree, one by one.

        - It might not find any,
        - It does so without merging common nodes, in order to find the first absorbing node as quick as possible.
        """
        if depth == 0:
            return
        for proba, bad_child in self.absorbing_states_one_depth():
            # print("all_absorbing_states: yielding proba, child = {}, \n{}".format(proba, bad_child))  # DEBUG
            yield proba, bad_child
        self.compute_one_depth()
        self.depth = depth
        if depth > 1:
            for child in self.children:
                for proba, bad_child in child.all_absorbing_states(depth=depth-1):
                    # print("all_absorbing_states: yielding proba, child = {}, \n{}".format(proba, bad_child))  # DEBUG
                    yield proba, bad_child

    def absorbing_states_one_depth(self):
        """Use all_deltas to yield all the absorbing one-depth child and their probabilities."""
        self.depth += 1
        for delta, proba in self.all_deltas():
            if proba == 0: continue
            # copy the current state, apply decision of algorithms and random branching
            child = delta(self.copy())
            if child.is_absorbing():
                # print("absorbing_states_one_depth: yielding proba, child = {}, \n{}".format(proba, child))  # DEBUG
                yield proba, child

    def find_N_absorbing_states(self, N=1, maxdepth=8):
        """Find at least N absorbing states, by considering a large depth."""
        complete_probas, bad_children = [], []
        for proba, bad_child in self.all_absorbing_states(depth=maxdepth):
            assert bad_child.is_absorbing(), "Error: a node was returned by all_absorbing_states() method but was not absorbing!"
            complete_probas.append(proba)
            bad_children.append(bad_child)
            if len(bad_children) >= N:
                return complete_probas, bad_children
        raise ValueError("Impossible to find N = {} absorbing states from this root (max depth = {})...".format(N, maxdepth))

    # --- The hard part is this all_deltas *generator*

    def all_deltas(self):
        """Generator that yields functions transforming state to another state.

        - It is memory efficient as it is a generator.
        - Do not convert that to a list or it might use all your system memory: each returned value is a function with code and variables inside!
        """
        all_decisions = [ player(j, self) for j, player in enumerate(self.players) ]
        number_of_decisions = prod(len(decisions) for decisions in all_decisions)
        for decisions in product(*all_decisions):
            counter = Counter(decisions)
            collisions = [counter.get(k, 0) >= 2 for k in range(self.K)]
            for coin_flips in product([0, 1], repeat=self.K):
                proba_of_this_coin_flip = prod(mu if b else (1 - mu) for b, mu in zip(coin_flips, self.mus))
                # Create a function to apply this transition
                def delta(s):
                    s.t += 1
                    s.depth -= 1
                    for j, Ij in enumerate(decisions):
                        s.S[j, Ij] += coin_flips[Ij]  # sensing feedback
                        s.N[j, Ij] += 1  # number of sensing trials
                        if not collisions[Ij]:  # no collision, receive this feedback for rewards
                            s.Stilde[j, Ij] += coin_flips[Ij]  # number of succesful transmissions
                            s.Ntilde[j, Ij] += 1  # number of trials without collisions
                    return s
                # Compute the probability of this transition
                proba = proba_of_this_coin_flip / number_of_decisions
                if proba == 0: continue
                yield (delta, proba)

    # --- Main functions, all explorations are depth first search (not the best, it's just easier...)

    def pretty_print_result_recursively(self):
        """Print all the transitions, depth by depth (recursively)."""
        if self.is_absorbing():
            print("\n\n")
            print("X "*87)
            print("The state:\n{}\nseems to be absorbing...".format(self))
            print("X "*87)
            # return
        if self.depth > 0:
            print("\n\nFrom this state :\n{}".format(self))
            for (proba, child) in zip(self.probas, self.children):
                print("\n- Probability of transition = {} to this other state:\n{}".format(proba, child))
                child.pretty_print_result_recursively()
            print("\n==> Done for the {} children of this state...\n".format(len(self.children)))

    def get_all_leafs(self):
        """Recurse and get all the leafs. Many different state can be present in the list of leafs, with possibly different probabilities (each correspond to a trajectory)."""
        if self.depth <= 1:
            return self.probas, self.children
        else:
            complete_probas, leafs = [], []
            # assert len(self.probas) > 0
            for (proba, child) in zip(self.probas, self.children):
                # assert child.depth == (self.depth - 1)
                c, l = child.get_all_leafs()
                # assert all([s.depth == 0 for s in l])
                c = [proba * p for p in c]  # one more step, multiply but a proba
                complete_probas.extend(c)
                leafs.extend(l)
            return complete_probas, leafs

    def get_unique_leafs(self):
        """Compute all the leafs (deepest children) and merge the common one to compute their full probabilities."""
        uniq_complete_probas = dict()
        uniq_leafs = dict()
        complete_probas, leafs = self.get_all_leafs()
        for proba, leaf in zip(complete_probas, leafs):
            h = hash(leaf)
            if h in uniq_leafs:
                uniq_complete_probas[h] += proba
            else:
                uniq_complete_probas[h] = proba
                uniq_leafs[h] = leaf
        return [simplify(p) for p in uniq_complete_probas.values()], list(uniq_leafs.values())

    def proba_reaching_absorbing_state(self):
        """Compute the probability of reaching a leaf that is an absorbing state."""
        probas, leafs = self.get_unique_leafs()
        bad_proba = 0
        nb_absorbing = 0
        for proba, leaf in zip(probas, leafs):
            if leaf.is_absorbing():
                bad_proba += proba
                nb_absorbing += 1
        print("\n\nFor depth {}, {} leafs were found to be absorbing, and the probability of reaching any absorbing leaf is {}...".format(self.depth, nb_absorbing, bad_proba))  # DEBUG
        sample_values = uniform_means(self.K)
        print("\n==> Numerically, for means = {}, this probability is = {:.3g} ...".format(np.array(sample_values), proba2float(bad_proba, values=sample_values)))  # DEBUG
        return nb_absorbing, bad_proba


class StateWithMemory(State):
    """State with a memory for each player, to represent and play with RhoRand etc."""

    def __init__(self, S, Stilde, N, Ntilde, mus, players, update_memories, memories=None, depth=0):
        super(StateWithMemory, self).__init__(S, Stilde, N, Ntilde, mus, players, depth=depth)
        self.update_memories = update_memories
        if memories is None:
            memories = tuple(1 for _ in range(self.M))
        self.memories = memories  #: Personal memory for all players, can be a rank in {1,..,M} for rhoRand, or anything else.

    def __str__(self, concise=False):
        if concise:
            return "    StateWithMemory : M = {}, K = {} and t = {}, depth = {}.\n{} =: Stilde\n{} =: N\n{} =: players memory\n".format(self.M, self.K, self.t, self.depth, self.Stilde, self.N, self.memories)
        else:
            return "    StateWithMemory : M = {}, K = {} and t = {}, depth = {}.\n{} =: S\n{} =: Stilde\n{} =: N\n{} =: Ntilde\n{} =: players memory\n".format(self.M, self.K, self.t, self.depth, self.S, self.Stilde, self.N, self.Ntilde, self.memories)

    def to_node(self, concise=CONCISE):
        """Print the state as a small string to be attached to a GraphViz node."""
        if concise:
            # return "[[" + "], [".join(",".join("{:.3g}/{}".format(st, n) for st, n in zip(st2, n2)) for st2, n2 in zip(self.S, self.N)) + "]]" + " r={}".format(list(self.memories))  # if U is used instead of Utilde
            return "[[" + "], [".join(",".join("{:.3g}/{}".format(st, n) for st, n in zip(st2, n2)) for st2, n2 in zip(self.Stilde, self.N)) + "]]" + " r={}".format(list(self.memories))
        else:
            return "[[" + "], [".join(",".join("{:.3g}:{:.3g}/{}:{}".format(s, st, n, nt) for s, st, n, nt in zip(s2, st2, n2, nt2)) for s2, st2, n2, nt2 in zip(self.S, self.Stilde, self.N, self.Ntilde)) + "]]" + " ranks = {}".format(self.memories)

    def copy(self):
        """Get a new copy of that state with same S, Stilde, N, Ntilde but no probas and no children (and depth=0)."""
        return StateWithMemory(S=self.S, Stilde=self.Stilde, N=self.N, Ntilde=self.Ntilde, mus=self.mus, players=self.players, update_memories=self.update_memories, depth=self.depth, memories=self.memories)

    def __hash__(self, full=FULLHASH):
        """Hash the matrix Stilde and N of the state and memories of the players (ie. ranks for RhoRand)."""
        if full:
            return hash(tupleit2(self.S)) + hash(tupleit2(self.N)) + hash(tupleit2(self.Stilde)) + hash(tupleit2(self.Ntilde) + (self.t, self.depth, )) + hash(tupleit1(self.memories))
        else:
            # return hash(tupleit2(self.S) + tupleit2(self.N) + tupleit1(self.memories))  # if U is used instead of Utilde
            return hash(tupleit2(self.Stilde)) + hash(tupleit2(self.N)) + hash(tupleit1(self.memories))

    def is_absorbing(self):
        """Try to detect if this state is absorbing, ie only one transition is possible, and again infinitely for the only child.

        .. warning:: Still very experimental!
        """
        if any('random' in update_memory.__name__.lower() for update_memory in self.update_memories):
            # eg RandomNewRank gives True, ConstantRank gives False
            return False
        else:
            return super(StateWithMemory, self).is_absorbing()

    def all_deltas(self):
        """Generator that yields functions transforming state to another state.

        - It is memory efficient as it is a generator.
        - Do not convert that to a list or it might use all your system memory: each returned value is a function with code and variables inside!
        """
        all_decisions = [ player(j, self) for j, player in enumerate(self.players) ]
        number_of_decisions = prod(len(decisions) for decisions in all_decisions)
        for decisions in product(*all_decisions):
            counter = Counter(decisions)
            collisions = [counter.get(k, 0) >= 2 for k in range(self.K)]
            all_memories = [ update_memory(j, self, decisions[j], collisions[decisions[j]]) for j, update_memory in enumerate(self.update_memories) ]
            number_of_memories = prod(len(memories) for memories in all_memories)
            for memories in product(*all_memories):
                for coin_flips in product([0, 1], repeat=self.K):
                    proba_of_this_coin_flip = prod(mu if b else (1 - mu) for b, mu in zip(coin_flips, self.mus))
                    # Create a function to apply this transition
                    def delta(s):
                        s.memories = memories  # Erase internal ranks etc
                        s.t += 1
                        s.depth -= 1
                        for j, Ij in enumerate(decisions):
                            s.S[j, Ij] += coin_flips[Ij]  # sensing feedback
                            s.N[j, Ij] += 1  # number of sensing trials
                            if not collisions[Ij]:  # no collision, receive this feedback for rewards
                                s.Stilde[j, Ij] += coin_flips[Ij]  # number of succesful transmissions
                                s.Ntilde[j, Ij] += 1  # number of trials without collisions
                        return s
                    # Compute the probability of this transition
                    proba = proba_of_this_coin_flip / (number_of_decisions * number_of_memories)
                    if proba == 0: continue
                    yield (delta, proba)


# --- Main function

def main(depth=1, players=None, update_memories=None, mus=None, M=2, K=2, S=None, Stilde=None, N=None, Ntilde=None, find_only_N=None):
    """Compute all the transitions, and print them."""
    if S is not None:
        M = min(np.shape(S))
        K = max(np.shape(S))
    if mus is None:
        mus = symbol_means(K=K)
    K = len(mus)
    if players is None:
        players = [default_policy for _ in range(M)]
    # if update_memories is None:
    #     update_memories = [default_update_memory for _ in range(M)]
    M = len(players)
    assert 1 <= M <= K <= 10, "Error: only 1 <= M <= K <= 10 are supported... and M = {}, K = {} here...".format(M, K)  # XXX it is probably impossible to have a code managing larger values...
    assert 0 <= depth <= 20, "Error: only 0 <= depth <= 20 is supported... and depth = {} here...".format(depth)  # XXX it is probably impossible to have a code managing larger values...
    # Compute starting state
    if S is None:
        S = np.zeros((M, K), dtype=int)  # Use only integers, to speed up in this case of Bernoulli arms. XXX in the general case it is not true!
    if Stilde is None:
        Stilde = np.zeros((M, K), dtype=int)  # Use only integers, to speed up in this case of Bernoulli arms. XXX in the general case it is not true!
    if N is None:
        N = np.zeros((M, K), dtype=int)
    if Ntilde is None:
        Ntilde = np.zeros((M, K),  dtype=int)
    # Create the root state
    if update_memories is not None:
        root = StateWithMemory(S=S, Stilde=Stilde, N=N, Ntilde=Ntilde, mus=mus, players=players, update_memories=update_memories)
    else:
        root = State(S=S, Stilde=Stilde, N=N, Ntilde=Ntilde, mus=mus, players=players)
    # Should we only look for find_only_N absorbing child?
    print("\nStarting to explore transitions up-to depth {} for this root state:\n{}".format(depth, root))
    print("    Using these policies:")
    for playerId, player in enumerate(players):
        print("  - Player #{}/{} uses {} (which is {})...".format(playerId, M, player.__name__, player))
    if update_memories is not None:
        print("    Using these update_memories:")
        for playerId, update_memory in enumerate(update_memories):
            print("  - Player #{}/{} uses {} (which is {})...".format(playerId, M, update_memory.__name__, update_memory))
    print("    Using these arms:")
    for muId, mu in enumerate(mus):
        print("  - Arm #{}/{} has mean {} ...".format(muId, K, mu))
    if find_only_N is not None:
        complete_probas, leafs = root.find_N_absorbing_states(N=find_only_N, maxdepth=depth)
        print("\n\n\nAs asked, we found {} absorbing nodes or leafs from this root at max depth = {} ...".format(find_only_N, depth))
        for proba, bad_child in zip(complete_probas, leafs):
            print("At depth {}, this node was found to be absorbing with probability {}:\n{}".format(bad_child.t, proba, bad_child))
    else:
        # Explore from the root
        root.explore_from_node_to_depth(depth=depth)
        # Print everything
        # root.pretty_print_result_recursively()
        # Get all leafs
        complete_probas, leafs = root.get_unique_leafs()
        print("\n\n\nThere are {} unique leafs for depth {}...".format(len(leafs), depth))
        for proba, leaf in zip(complete_probas, leafs):
            print("\n Leaf with probability = {}:\n{}".format(proba, leaf))
            if leaf.is_absorbing():
                print("  At depth {}, this leaf was found to be absorbing !".format(depth))
    # Done
    print("\nDone for exploring transitions up-to depth {} for this root state:\n{}".format(depth, root))
    print("    Using these policies:")
    for playerId, player in enumerate(players):
        print("  - Player #{}/{} uses {} (which is {})...".format(playerId, M, player.__name__, player))
    if update_memories is not None:
        print("    Using these update_memories:")
        for playerId, update_memory in enumerate(update_memories):
            print("  - Player #{}/{} uses {} (which is {})...".format(playerId, M, update_memory.__name__, update_memory))
    print("    Using these arms:")
    for muId, mu in enumerate(mus):
        print("  - Arm #{}/{} has mean {} ...".format(muId, K, mu))
    if find_only_N is None:
        print("\nThere were {} unique leafs for depth {}...".format(len(leafs), depth))
    return root, complete_probas, leafs


# --- Main script

def test(depth=1, M=2, K=2, S=None, Stilde=None, N=None, Ntilde=None, mus=None, debug=True, all_players=None, all_update_memories=None, find_only_N=None):
    """Test the main exploration function for various all_players."""
    results = []
    if all_players is None:
        all_players = [FixedArm]
    if all_update_memories is None:
        all_update_memories = [None] * len(all_players)
    for policy, update_memory in zip(all_players, all_update_memories):
        players = [ policy for _ in range(M) ]
        update_memories = [ update_memory for _ in range(M) ] if update_memory is not None else None
        # get the result
        root, complete_probas, leafs = main(depth=depth, players=players, update_memories=update_memories, S=S, N=N, Stilde=Stilde, Ntilde=Ntilde, M=M, K=K, mus=mus, find_only_N=find_only_N)
        if find_only_N is None:
            # computing absorbing states
            nb_absorbing, bad_proba = root.proba_reaching_absorbing_state()
            # XXX save the graph and maybe display it, in different versions
            for onlyabsorbing, onlyleafs in product((True, False), (True, False)):
                if nb_absorbing == 0 and onlyabsorbing:  continue
                if depth == 1 and onlyleafs:  continue
                for latex, ext in ((True, 'svg'), (False, 'svg')):  # , (False, 'png')
                    try:
                        filename = "Tree_exploration_K={}_M={}_depth={}__{}{}{}{}.gv".format(
                                    K, M, depth, policy.__name__,
                                    "__{}".format(update_memory.__name__) if update_memory is not None else "",
                                    "__absorbing" if onlyabsorbing else "",
                                    "__leafs" if onlyleafs else ""
                        )
                        root.saveto(os_path_join(PLOT_DIR, filename), view=debug,
                                    title="Tree exploration for K={} arms and M={} players using {}{}, for depth={} : {} leafs, {} absorbing".format(K, M, policy.__name__,
                                                      " and {}".format(update_memory.__name__) if update_memory is not None else "",
                                                      depth, len(leafs), nb_absorbing),
                                    onlyabsorbing=onlyabsorbing, onlyleafs=onlyleafs,
                                    latex=latex, ext=ext)
                    except ValueError as e:
                        print("    Error when saving:", e)
        else:
            nb_absorbing, bad_proba = len(complete_probas), sum(complete_probas)
        # store everything
        results.append([root, complete_probas, leafs, nb_absorbing, bad_proba])
        # ask for Enter to continue
        if debug:
            print(input("\n\n[Enter] to continue..."))
    return results


if __name__ == '__main__':
    all_update_memories = None
    all_players = [FixedArm]  # XXX just for testing
    all_players = [UniformExploration]  # XXX just for testing

    # --- XXX Test for Selfish Utilde
    # all_update_memories = [ConstantRank]
    all_players = [Selfish_0Greedy_Utilde, Selfish_UCB_Utilde, Selfish_KLUCB_Utilde]  # XXX complete comparison
    # all_players = [Selfish_UCB_Utilde, Selfish_KLUCB_Utilde]  # XXX comparison

    all_players = [Selfish_UCB, Selfish_KLUCB]  # XXX comparison
    # all_players = [Selfish_KLUCB_Utilde]
    # all_players = [Selfish_UCB_Utilde]  # Faster, and probably same error cases as KLUCB

    # # --- XXX Test for RhoRand
    # all_players = [RhoRand_UCB_Utilde, RhoRand_KLUCB_Utilde]  # XXX  comparison
    # all_players = [RhoRand_KLUCB_U]
    # all_players = [RhoRand_UCB_U]  # Faster, and probably same error cases as KLUCB
    # all_update_memories = [RandomNewRank]

    # # --- XXX Test for RandTopM
    # all_players = [RandTopM_UCB_U]  # Faster, and probably same error cases as KLUCB
    # all_update_memories = [RandTopM_RandomNewChosenArm]

    # # --- XXX Test for RandTopMC
    # all_players = [MCTopM_UCB_U]  # Faster, and probably same error cases as KLUCB
    # all_update_memories = [MCTopM_RandomNewChosenArm]

    # --- XXX Faster or symbolic computations?
    mus = None  # use mu_1, .., mu_K as symbols, by default
    # mus = [0, 1]
    # mus = [0.1, 0.9]
    # mus = [0.1, 0.5, 0.9]

    # --- XXX Read parameters from the cli env
    depth = int(getenv("DEPTH", "1"))
    M = int(getenv("M", "2"))
    K = int(getenv("K", "2"))
    DEBUG = mybool(getenv("DEBUG", False))

    find_only_N = int(getenv("FIND_ONLY_N", "0"))
    if find_only_N <= 0: find_only_N = None
    if find_only_N:
        mus = uniform_means(nbArms=K)

    results = test(depth=depth, M=M, K=K, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)

    # # XXX default start state
    # M, K = 1, 1
    # for depth in [8]:
    #     print("For depth = {} ...".format(depth))
    #     results = test(depth=depth, M=M, K=K, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)

    # # XXX default start state
    # M, K = 2, 2
    # # mus = [0.8, 0.2]
    # # mus = [Fraction(4, 5), Fraction(1, 5)]
    # for depth in [1, 2, 3]:
    #     print("For depth = {} ...".format(depth))
    #     results = test(depth=depth, M=M, K=K, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)

    # # XXX What if we start from an absorbing state?
    # M, K = 2, 2
    # S = np.array([[1, 0], [1, 0]])
    # Stilde = np.array([[1, 0], [1, 0]])
    # N = np.array([[2, 1], [2, 1]])
    # Ntilde = np.array([[2, 1], [2, 1]])
    # for depth in [1, 2, 3]:
    #     results = test(depth=depth, M=M, K=K, S=S, Stilde=Stilde, N=N, Ntilde=Ntilde, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)

    # # XXX default start state
    # M, K = 2, 3
    # results = test(depth=depth, M=M, K=K, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)

    # XXX What if we start from an absorbing state?
    # M, K = 2, 3
    # S = np.array([[2, 1, 0], [2, 1, 0]])
    # Stilde = np.array([[2, 1, 0], [2, 1, 0]])
    # N = np.array([[4, 3, 1], [4, 3, 1]])
    # Ntilde = np.array([[4, 3, 1], [4, 3, 1]])
    # # for depth in [1]:
    # for depth in [2, 3, 4]:
    #     # results = test(depth=depth, M=M, K=K, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)
    #     results = test(depth=depth, M=M, K=K, S=S, Stilde=Stilde, N=N, Ntilde=Ntilde, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)

    # M, K = 3, 3
    # results = test(depth=depth, M=M, K=K, mus=mus, all_players=all_players, all_update_memories=all_update_memories, find_only_N=find_only_N, debug=DEBUG)

# End of complete_tree_exploration_for_MP_bandits.py
