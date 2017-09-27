#! /usr/bin/env python2
# -*- coding: utf-8; mode: python -*-
r""" Experimental code to perform complete tree exploration for Multi-Player bandits.

- Support Selfish 0-greedy, UCB, and klUCB in 3 differents variants.
- Support export of the tree to a GraphViz dot graph, and can save it to SVG/PNG/PDF etc.
- For the means of each arm, :math:`\mu_1, \dots, \mu_K`, this script can use exact formal computations with sympy, or fractions with Fraction, or float number.
- The graph can contain all nodes from root to leafs, or only leafs (with summed probabilities), and possibly only the absorbing nodes are showed.
- By default, the root is highlighted in green and the absorbing nodes are in red.

.. warning::

   - TODO : add rhoRand, TopBestM etc. that are *not* memory-less... (rhoRand needs a rank for instance). It's harder!
   - TODO : implement a parallel loop ? for what can be parallelized...?
   - TODO : right now, it is not so efficient, it could be improved!


.. note:: Requirements:

    - 'sympy' module to use formal means :math:`\mu_1, \dots, \mu_K` instead of numbers,
    - 'numpy' module for computations on indexes (e.g., ``np.where``),
    - 'graphviz' module to generate the graph and save it,
    - 'dot2tex' module to generate nice LaTeX graph and save it to PDF.

About:

- *Date:* 16/09/2017.
- *Author:* Lilian Besson, (C) 2017
- *Licence:* MIT Licence (http://lbesson.mit-license.org).
"""

from __future__ import print_function, division  # Python 2 compatibility if needed
__author__ = "Lilian Besson"
__version__ = "0.6"
from sys import version_info
if version_info.major < 3:  # Python 2 compatibility if needed
    input = raw_input

from collections import Counter, deque
from fractions import Fraction
from itertools import product
from os import getenv
from os.path import join as os_path_join
from re import sub as re_sub
from textwrap import wrap

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
    from dot2tex import dot2tex
except ImportError:
    print("Warning: the 'dot2tex' module was not found...\nTrees cannot be saved to LaTeX and PDF formats.\nInstall it with 'sudo pip2 install dot2tex' (require Python 2)")  # XXX

oo = float('+inf')  #: Shortcut for float('+inf').

PLOT_DIR = os_path_join("plots", "trees")  #: Directory for the plots


def tupleit1(anarray):
    """Convert a non-hashable 1D numpy array to a hashable tuple."""
    return tuple(anarray.tolist())

def tupleit2(anarray):
    """Convert a non-hashable 2D numpy array to a hashable tuple-of-tuples."""
    return tuple([tuple(r) for r in anarray.tolist()])

def prod(iterator):
    """Product of the values in this iterator."""
    p = 1
    for v in iterator:
        p *= v
    return p

def choices_from_indexes(indexes):
    """For deterministic index policies, if more than one index is maximum, return the list of positions attaining this maximum (ties), or only one position."""
    return np.where(indexes == np.max(indexes))[0]


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

def FixedArm(j, state):
    """Fake player j that always targets at arm j."""
    return [j]

def UniformExploration(j, state):
    """Fake player j that always targets all arms."""
    return np.arange(state.K)

# --- Selfish 0-greedy variants

def Selfish_0Greedy_U(j, state):
    """Selfish policy + 0-Greedy index + U feedback."""
    indexes = state.S[j] / state.N[j]
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

def Selfish_0Greedy_Utilde(j, state):
    """Selfish policy + 0-Greedy index + Utilde feedback."""
    indexes = state.Stilde[j] / state.N[j]
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

def Selfish_0Greedy_Ubar(j, state):
    """Selfish policy + 0-Greedy index + Ubar feedback."""
    indexes = (state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

default_policy = Selfish_0Greedy_Ubar


# --- Selfish UCB variants
alpha = 0.5

def Selfish_UCB_U(j, state):
    """Selfish policy + UCB_0.5 index + U feedback."""
    indexes = (state[j].S / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

def Selfish_UCB_Utilde(j, state):
    """Selfish policy + UCB_0.5 index + Utilde feedback."""
    indexes = (state.Stilde[j] / state.N[j]) + np.sqrt(alpha * np.log(state.t) / state.N[j])
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

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

def Selfish_KLUCB_U(j, state):
    """Selfish policy + Bernoulli KL-UCB index + U feedback."""
    indexes = klucb(state.S[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

def Selfish_KLUCB_Utilde(j, state):
    """Selfish policy + Bernoulli KL-UCB index + Utilde feedback."""
    indexes = klucb(state.Stilde[j] / state.N[j], c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

def Selfish_KLUCB_Ubar(j, state):
    """Selfish policy + Bernoulli KL-UCB index + Ubar feedback."""
    indexes = klucb((state.Ntilde[j] / state.N[j]) * (state.S[j] / state.N[j]), c * np.log(state.t) / state.N[j], tolerance)
    indexes[state.N[j] < 1] = +oo
    return choices_from_indexes(indexes)

# default_policy = Selfish_KLUCB_Ubar

# --- FIXME write rhoRand, TopBestM, MusicalChair and all variants !
# XXX It is probably harder... rhoRand is NOT memory less!!
# XXX TopBestM and MusicalChair also!!


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
        elif html_in_var_names:  # replace mu_12 by mu<sub>12</sub>
            str_proba = re_sub(r'_([0-9]+)', r'<SUB>\1</SUB>', str_proba)
        else:
            str_proba = re_sub(r'mu_', r'µ', str_proba)
    return str_proba

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
               latex=False, html_in_var_names=False,
               onlyleafs=ONLYLEAFS, onlyabsorbing=ONLYABSORBING, concise=CONCISE):
        r"""Convert the state to a .dot graph, using GraphViz. See http://graphviz.readthedocs.io/ for more details.

        - onlyleafs: only print the root and the leafs, to see a concise representation of the tree.
        - onlyabsorbing: only print the absorbing leafs, to see a really concise representation of the tree.
        - concise: weather to use the short representation of states (using :math:`\tilde{S}` and :math:`N`) or the long one (using the 4 variables).
        - html_in_var_names: experimental use of ``<SUB>..</SUB>`` and ``<SUP>..</SUP>`` in the label for the tree.
        - latex: experimental use of ``_{..}`` and ``^{..}`` in the label for the tree, to use with dot2tex.
        """
        dot = Digraph(name=name, comment=comment, format=FORMAT)
        print("Creating a dot graph from the tree... \n{}".format(dot))
        dot.attr(overlap="false")
        if title: dot.attr(label=wraptext(title))
        node_number = 0
        if onlyleafs:
            root_name, root = "0", self
            dot.node(root_name, root.to_node(concise=concise), color="green")
            complete_probas, leafs = root.get_unique_leafs()
            if len(leafs) > 128:
                raise ValueError("Useless to save a tree with more than 128 leafs, the resulting image will be too large to be viewed.")  # DEBUG
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
                if nb_node > 512:
                    raise ValueError("Useless to save a tree with more than 512 nodes, the resulting image will be too large to be viewed.")  # DEBUG
        return dot

    def saveto(self, filename, view=True, title="", name="", comment="",
               latex=False, html_in_var_names=False, format=FORMAT,
               onlyleafs=ONLYLEAFS, onlyabsorbing=ONLYABSORBING, concise=CONCISE):
        dot = self.to_dot(title=title, name=name, comment=comment, html_in_var_names=html_in_var_names, latex=latex, onlyleafs=onlyleafs, onlyabsorbing=onlyabsorbing, concise=concise)
        if latex:
            # FIXME add support for dot2tex
            filename = filename.replace('.gv', '.gv.tex')
            dot2tex(dot.source, output=filename, format='tikz', crop=True, figonly=True, textmode='raw')
            print("Saving the dot graph to '{}.{}'...".format(filename, format))
            print("Saved to ")
        else:
            print("Saving the dot graph to '{}.{}'...".format(filename, format))
            dot.render(filename, view=view)

    def copy(self):
        """Get a new copy of that state with same S, Stilde, N, Ntilde but no probas and no children (and depth=0)."""
        return State(S=self.S, Stilde=self.Stilde, N=self.N, Ntilde=self.Ntilde, mus=self.mus, players=self.players, depth=self.depth)

    def __hash__(self, full=FULLHASH):
        """Hash the matrix Stilde and N of the state."""
        if full:
            return hash(tupleit2(self.S) + tupleit2(self.N) + tupleit2(self.Stilde) + tupleit2(self.Ntilde) + (self.t, self.depth, ))
        else:
            return hash(tupleit2(self.Stilde) + tupleit2(self.N))

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
        nb_transitions = 0
        for delta, proba in self.all_deltas():
            nb_transitions += 1
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
        """Generator that yield lambda functions transforming state to another state.

        - It is memory efficient as it is a generator.
        - Do not convert that to a list or it might use all your system memory: each returned value is a lambda function with code and variables inside!
        """
        all_decisions = [ player(j, self) for j, player in enumerate(self.players) ]
        number_of_decisions = prod(len(decision) for decision in all_decisions)
        for decisions in product(*all_decisions):
            for coin_flips in product([0, 1], repeat=self.K):
                proba_of_this_coin_flip = prod(mu if b else (1 - mu) for b, mu in zip(coin_flips, self.mus))
                # Create a function to apply this transition
                def delta(s):
                    s.t += 1
                    s.depth -= 1
                    # collisions = [np.count_nonzero(np.array(decisions) == k) >= 2 for k in range(self.K)]
                    counter = Counter(decisions)
                    collisions = [counter.get(k, 0) >= 2 for k in range(self.K)]  # XXX faster with Counter
                    for j, Ij in enumerate(decisions):
                        s.S[j, Ij] += coin_flips[Ij]  # sensing feedback
                        s.N[j, Ij] += 1  # number of sensing trials
                        if not collisions[Ij]:  # no collision, receive this feedback for rewards
                            s.Stilde[j, Ij] += coin_flips[Ij]  # number of succesful transmissions
                            s.Ntilde[j, Ij] += 1  # number of trials without collisions
                    return s
                # Compute the probability of this transition
                proba = proba_of_this_coin_flip / number_of_decisions
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
        print("\n==> Numerically, for means = {}, this probability is = {:.3g} ...".format(sample_values, proba2float(bad_proba, values=sample_values)))  # DEBUG
        return nb_absorbing, bad_proba


# --- Main function

def main(depth=1, players=None, mus=None, M=2, K=2, S=None, Stilde=None, N=None, Ntilde=None, find_only_N=None):
    """Compute all the transitions, and print them."""
    if S is not None:
        M = min(np.shape(S))
        K = max(np.shape(S))
    if mus is None:
        mus = symbol_means(K=K)
    K = len(mus)
    if players is None:
        players = [default_policy for _ in range(M)]
    M = len(players)
    assert 1 <= M <= K <= 10, "Error: only 1 <= M <= K <= 10 are supported... and M = {}, K = {} here...".format(M, K)  # FIXME
    assert 0 <= depth <= 20, "Error: only 0 <= depth <= 20 is supported... and depth = {} here...".format(depth)  # FIXME
    # Compute starting state
    if S is None:
        # S = np.zeros((M, K))
        S = np.zeros((M, K), dtype=int)  # FIXME in the general case it is not true!
    if Stilde is None:
        # Stilde = np.zeros((M, K))
        Stilde = np.zeros((M, K), dtype=int)  # FIXME in the general case it is not true!
    if N is None:
        N = np.zeros((M, K), dtype=int)
    if Ntilde is None:
        Ntilde = np.zeros((M, K),  dtype=int)
    # Create the root state
    root = State(S=S, Stilde=Stilde, N=N, Ntilde=Ntilde, mus=mus, players=players)
    # Should we only look for find_only_N absorbing child?
    print("\nStarting to explore transitions up-to depth {} for this root state:\n{}".format(depth, root))
    print("    Using these policies:")
    for playerId, player in enumerate(players):
        print("  - Player #{}/{} uses {} (which is {})...".format(playerId, M, player.__name__, player))
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
    print("    Using these arms:")
    for muId, mu in enumerate(mus):
        print("  - Arm #{}/{} has mean {} ...".format(muId, K, mu))
    print("\nThere were {} unique leafs for depth {}...".format(len(leafs), depth))
    return root, complete_probas, leafs


# --- Main script

def test(depth=1, M=2, K=2, S=None, Stilde=None, N=None, Ntilde=None, mus=None, debug=True, policies=None, find_only_N=None):
    """Test the main exploration function for various policies."""
    results = []
    if policies is None:
        policies = [FixedArm]
    for policy in policies:
        players = [ policy for _ in range(M) ]
        # get the result
        root, complete_probas, leafs = main(depth=depth, players=players, S=S, N=N, Stilde=Stilde, Ntilde=Ntilde, M=M, K=K, mus=mus, find_only_N=find_only_N)
        if find_only_N is None:
            # computing absorbing states
            nb_absorbing, bad_proba = root.proba_reaching_absorbing_state()
            # XXX save the graph and maybe display it
            for onlyabsorbing, onlyleafs in product([True, False], repeat=2):
                try:
                    root.saveto(os_path_join(PLOT_DIR, "Tree_exploration_K={}_M={}_depth={}__{}{}{}.gv".format(K, M, depth, policy.__name__, "__absorbing" if onlyabsorbing else "", "__leafs" if onlyleafs else "")), view=debug, title="Tree exploration for K={} arms and M={} players using {}, for depth={} : {} leafs, {} absorbing".format(K, M, policy.__name__, depth, len(leafs), nb_absorbing), onlyabsorbing=onlyabsorbing, onlyleafs=onlyleafs, latex=True)
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
    policies = [FixedArm]  # FIXME just for testing
    policies = [UniformExploration]  # FIXME just for testing
    policies = [Selfish_0Greedy_Ubar, Selfish_UCB_Ubar, Selfish_KLUCB_Ubar]  # FIXME complete comparison
    policies = [Selfish_0Greedy_Ubar]
    policies = [Selfish_UCB_Ubar]  # Faster, and probably same error cases as KLUCB
    # policies = [Selfish_KLUCB_Ubar]

    mus = None
    # mus = [0.1, 0.9]
    # mus = [0.1, 0.5, 0.9]

    # FIXME Read parameters from the cli env
    depth = int(getenv("DEPTH", "1"))
    M = int(getenv("M", "2"))
    K = int(getenv("K", "2"))
    DEBUG = mybool(getenv("DEBUG", False))

    find_only_N = int(getenv("FIND_ONLY_N", "0"))
    if find_only_N <= 0: find_only_N = None
    if find_only_N:
        mus = uniform_means(nbArms=K)

    print("For depth = {} ...".format(depth))
    results = test(depth=depth, M=M, K=K, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)

    # # XXX default start state
    # M, K = 1, 1
    # for depth in [8]:
    #     print("For depth = {} ...".format(depth))
    #     results = test(depth=depth, M=M, K=K, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)

    # # XXX default start state
    # M, K = 2, 2
    # # mus = [0.8, 0.2]
    # # mus = [Fraction(4, 5), Fraction(1, 5)]
    # for depth in [1, 2, 3]:
    #     print("For depth = {} ...".format(depth))
    #     results = test(depth=depth, M=M, K=K, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)

    # # XXX What if we start from an absorbing state?
    # M, K = 2, 2
    # S = np.array([[1, 0], [1, 0]])
    # Stilde = np.array([[1, 0], [1, 0]])
    # N = np.array([[2, 1], [2, 1]])
    # Ntilde = np.array([[2, 1], [2, 1]])
    # for depth in [1, 2, 3]:
    #     results = test(depth=depth, M=M, K=K, S=S, Stilde=Stilde, N=N, Ntilde=Ntilde, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)

    # # XXX default start state
    # M, K = 2, 3
    # results = test(depth=depth, M=M, K=K, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)

    # # XXX What if we start from an absorbing state?
    # M, K = 2, 3
    # # S = np.array([[2, 1, 0], [2, 1, 0]])
    # # Stilde = np.array([[2, 1, 0], [2, 1, 0]])
    # # N = np.array([[4, 3, 1], [4, 3, 1]])
    # # Ntilde = np.array([[4, 3, 1], [4, 3, 1]])
    # # for depth in [1]:
    # for depth in [2, 3]:
    #     results = test(depth=depth, M=M, K=K, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)
    #     # results = test(depth=depth, M=M, K=K, S=S, Stilde=Stilde, N=N, Ntilde=Ntilde, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)

    # M, K = 3, 3
    # results = test(depth=depth, M=M, K=K, mus=mus, policies=policies, find_only_N=find_only_N, debug=DEBUG)

# End of complete-tree-exploration-for-MP-bandits.py
