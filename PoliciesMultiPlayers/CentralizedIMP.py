# -*- coding: utf-8 -*-
""" CentralizedIMP: a multi-player policy where ONE policy is used by a centralized agent; asking the policy to select nbPlayers arms at each step, using an hybrid strategy: choose nb-1 arms with maximal empirical averages, then 1 arm with maximal index. Cf. algorithm IMP-TS [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.2"

import numpy as np

from .CentralizedMultiplePlay import CentralizedMultiplePlay


# --- Class for the mother

class CentralizedIMP(CentralizedMultiplePlay):
    """ CentralizedIMP: a multi-player policy where ONE policy is used by a centralized agent; asking the policy to select nbPlayers arms at each step, using an hybrid strategy: choose nb-1 arms with maximal empirical averages, then 1 arm with maximal index. Cf. algorithm IMP-TS [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779].
    """

    def _choice_one(self, playerId):
        """Use `choiceIMP` for each player."""
        if playerId == 0:  # For the first player, run the method
            # FIXED sort it then apply affectation_order, to fix its order ==> will have a fixed nb of switches for CentralizedMultiplePlay
            if self.uniformAllocation:
                self.choices = self.player.choiceIMP(self.nbPlayers)
            else:
                self.choices = np.sort(self.player.choiceIMP(self.nbPlayers))[self.affectation_order]  # XXX Increasing order...
                # self.choices = np.sort(self.player.choiceMultiple(self.nbPlayers))[self.affectation_order][::-1]  # XXX Decreasing order...
            # print("At time t = {} the {} centralized policy chosed arms = {} ...".format(self.player.t, self, self.choices))  # DEBUG
        # For the all players, use the pre-computed result
        return self.choices[playerId]
