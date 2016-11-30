# -*- coding: utf-8 -*-
""" PoliciesMultiPlayers : contains various collision-avoidance protocol for the multi-players setting.

- Selfish: a multi-player policy where every player is selfish, they do not try to handle the collisions.

- CentralizedNotFair: a multi-player policy which uses a centralize intelligence to affect users to a FIXED arm.
- CentralizedFair: a multi-player policy which uses a centralize intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbArms.

- OracleNotFair: a multi-player policy with full knowledge and centralized intelligence to affect users to a FIXED arm, among the best arms.
- OracleFair: a multi-player policy which uses a centralized intelligence to affect users an offset, each one take an orthogonal arm based on (offset + t) % nbBestArms, among the best arms.
"""

__author__ = "Lilian Besson"
__version__ = "0.1"

# Mine, stupid and decentralized ones
from .Selfish import Selfish

# Mine, centralized ones (but only knowledge of nbArms)
from .CentralizedNotFair import CentralizedNotFair
from .CentralizedFair import CentralizedFair

# Mine, centralized ones (with perfect knowledge)
from .OracleNotFair import OracleNotFair
from .OracleFair import OracleFair


# FIXME implement it
# from .MusicalChair import MusicalChair    # FIXME no it is a single-player policy!
# from .DMC import DMC  # XXX Dynamic setting!

# FIXME implement it
# from .MEGA import MEGA

# FIXME implement it
# from .RhoRand import RhoRand

# FIXME implement it
# from .TDFS import TDFS
