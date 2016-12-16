# -*- coding: utf-8 -*-
""" PoliciesMultiPlayers : contains various collision-avoidance protocol for the multi-players setting.

- Selfish: a multi-player policy where every player is selfish, they do not try to handle the collisions.

- CentralizedNotFair: a multi-player policy which uses a centralize intelligence to affect users to a FIXED arm.
- CentralizedFair: a multi-player policy which uses a centralize intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.

- OracleNotFair: a multi-player policy with full knowledge and centralized intelligence to affect users to a FIXED arm, among the best arms.
- OracleFair: a multi-player policy which uses a centralized intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbBestArms, among the best arms.

- rhoRand, ALOHA: implementation of generic collision avoidance algorithms, relying on a single-player bandit policy (eg. UCB, Thompson etc).
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

# Mine, fully decentralized one
from .Selfish import Selfish

# Mine, centralized ones (but only knowledge of nbArms)
from .CentralizedNotFair import CentralizedNotFair
from .CentralizedFair import CentralizedFair

# Mine, centralized ones (with perfect knowledge)
from .OracleNotFair import OracleNotFair
from .OracleFair import OracleFair

# TODO CentralizedMultiplePlay where ONE M multi-play bandit algorithm is ran, instead of decentralized one-play bandits ran by each of the M players
# from .CentralizedMultiplePlay import CentralizedMultiplePlay

from .rhoRand import rhoRand  # Cf. [Anandkumar et al., 2009](http://ieeexplore.ieee.org/document/5462144/)

from .ALOHA import ALOHA, tnext_beta, tnext_log

# FIXME implement it
# from .TDFS import TDFS
