.. SMPyBandits documentation master file, created by
   sphinx-quickstart on Thu Jan 19 17:20:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SMPyBandits documentation!
=====================================

**Open-Source Python package for Single- and Multi-Players multi-armed Bandits algorithms**.

A research framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms: UCB, KL-UCB, Thompson and many more for single-players, and MCTopM & RandTopM, MusicalChair, ALOHA, MEGA, rhoRand for multi-players simulations.
It runs on Python 2 and 3, and is publically released as an open-source software under the `MIT License <https://lbeson.mit-license.org/>`_.

.. note::

    See more on `the GitHub page for this project <https://github.com/SMPyBandits/SMPyBandits/>`_: `<https://github.com/SMPyBandits/SMPyBandits/>`_.
    The project is also hosted on `Inria GForge <https://gforge.inria.fr/project/admin/?group_id=9477>`_, and the documentation can be seen online at `<https://smpybandits.github.io/>`_ or `<http://http://banditslilian.gforge.inria.fr/>`_ or `<https://smpybandits.readthedocs.io/>`_.
    |Website up|

This repository contains the code of `my <https://perso.crans.org/besson/>`_ numerical environment, written in `Python <https://www.python.org/>`_, in order to perform numerical
simulations on *single*-player and *multi*-players `Multi-Armed Bandits
(MAB) <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_ algorithms.

|Open Source? Yes!| |Maintenance| |Ask Me Anything !| |Analytics| |PyPI version| |PyPI implementation| |PyPI pyversions| |Documentation Status| |Build Status|


`I (Lilian Besson) <https://perso.crans.org/besson/>`_ have `started my
PhD <https://perso.crans.org/besson/phd/>`_ in October 2016, and this is
a part of my **on going** research since December 2016.


How to cite this work?
~~~~~~~~~~~~~~~~~~~~~~
If you use this package for your own work, please consider citing it with this piece of BibTeX: ::

    @misc{SMPyBandits,
        title =   {{SMPyBandits: an Open-Source Research Framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms in Python}},
        author =  {Lilian Besson},
        year =    {2018},
        url =     {https://github.com/SMPyBandits/SMPyBandits/},
        howpublished = {Online at: \url{GitHub.com/SMPyBandits/SMPyBandits}},
        note =    {Code at https://github.com/SMPyBandits/SMPyBandits/, documentation at https://smpybandits.github.io/}
    }

I also wrote a small paper to present *SMPyBandits*, and I will send it to `JMLR MLOSS <http://jmlr.org/mloss/>`_.
The paper can be consulted `here on my website <https://perso.crans.org/besson/articles/SMPyBandits.pdf>`_.

----

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    README.md
    docs/modules.rst
    How_to_run_the_code.md
    PublicationsWithSMPyBandits.md
    Aggregation.md
    MultiPlayers.md
    DoublingTrick.md
    SparseBandits.md
    NonStationaryBandits.md
    API.md
    About_parallel_computations.md
    TODO.md
    plots/README.md
    notebooks/README.md
    notebooks/list.rst
    Policies/C/README.md
    Profiling.md
    uml_diagrams/README.md
    logs/README.md
    CONTRIBUTING.md
    CODE_OF_CONDUCT.md


.. note::

    Both this documentation and the code are publicly available, under the `open-source MIT License <https://lbesson.mit-license.org/>`_.
    The code is hosted on GitHub at `github.com/SMPyBandits/SMPyBandits <https://github.com/SMPyBandits/SMPyBandits/>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`classindex`,
* :ref:`funcindex`,
* :ref:`methindex`,
* :ref:`staticmethindex`,
* :ref:`attrindex`,
* :ref:`search`.

|Stars of https://github.com/SMPyBandits/SMPyBandits/| |Contributors of https://github.com/SMPyBandits/SMPyBandits/| |Watchers of https://github.com/SMPyBandits/SMPyBandits/| |Forks of https://github.com/SMPyBandits/SMPyBandits/|

|Releases of https://github.com/SMPyBandits/SMPyBandits/|
|Commits of https://github.com/SMPyBandits/SMPyBandits/| / |Date of last commit of https://github.com/SMPyBandits/SMPyBandits/|

|Issues of https://github.com/SMPyBandits/SMPyBandits/| : |Open issues of https://github.com/SMPyBandits/SMPyBandits/| / |Closed issues of https://github.com/SMPyBandits/SMPyBandits/|

|Pull requests of https://github.com/SMPyBandits/SMPyBandits/| : |Open pull requests of https://github.com/SMPyBandits/SMPyBandits/| / |Closed pull requests of https://github.com/SMPyBandits/SMPyBandits/|

|ForTheBadge uses-badges| |ForTheBadge uses-git|
|forthebadge made-with-python| |ForTheBadge built-with-science|


.. |Open Source? Yes!| image:: https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github
    :target: https://github.com/Naereen/badges/
.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity
.. |Ask Me Anything !| image:: https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg
    :target: https://GitHub.com/Naereen/ama
.. |Analytics| image:: https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/
.. |PyPI version| image:: https://img.shields.io/pypi/v/smpybandits.svg
    :target: https://PyPI.org/project/SMPyBandits/
.. |PyPI implementation| image:: https://img.shields.io/pypi/implementation/smpybandits.svg
    :target: https://PyPI.org/project/SMPyBandits/
.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/smpybandits.svg
    :target: https://PyPI.org/project/SMPyBandits/
.. |Documentation Status| image:: https://readthedocs.org/projects/smpybandits/badge/?version=latest
    :target: https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest
.. |Build Status| image:: https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master
    :target: https://travis-ci.org/SMPyBandits/SMPyBandits
.. |Website up| image:: https://img.shields.io/website-up-down-green-red/http/banditslilian.gforge.inria.fr.svg
    :target: https://smpybandits.github.io/
.. |ForTheBadge uses-badges| image:: http://ForTheBadge.com/images/badges/uses-badges.svg
    :target: http://ForTheBadge.com
.. |ForTheBadge uses-git| image:: http://ForTheBadge.com/images/badges/uses-git.svg
    :target: https://GitHub.com/
.. |forthebadge made-with-python| image:: http://ForTheBadge.com/images/badges/made-with-python.svg
    :target: https://www.python.org/
.. |ForTheBadge built-with-science| image:: http://ForTheBadge.com/images/badges/built-with-science.svg
    :target: https://GitHub.com/Naereen/
.. |MIT license| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://lbesson.mit-license.org/
.. |Stars of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/stars/SMPyBandits/SMPyBandits
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/stargazers
.. |Contributors of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/contributors/SMPyBandits/SMPyBandits
    :target: https://github.com/SMPyBandits/SMPyBandits/graphs/contributors
.. |Watchers of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/watchers/SMPyBandits/SMPyBandits
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/watchers
.. |Forks of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/forks/SMPyBandits/SMPyBandits
    :target: https://github.com/SMPyBandits/SMPyBandits/network/members
.. |Releases of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/release/SMPyBandits/SMPyBandits
    :target: https://github.com/SMPyBandits/SMPyBandits/releases
.. |Commits of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/commits/SMPyBandits/SMPyBandits
    :target: https://github.com/SMPyBandits/SMPyBandits/commits/master
.. |Date of last commit of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/last-commit/SMPyBandits/SMPyBandits
    :target: https://github.com/SMPyBandits/SMPyBandits/commits/master
.. |Issues of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/issues/SMPyBandits/SMPyBandits
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/issues
.. |Open issues of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/open-issues/SMPyBandits/SMPyBandits
    :target: https://github.com/SMPyBandits/SMPyBandits/issues?q=is%3Aopen+is%3Aissue
.. |Closed issues of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/closed-issues/SMPyBandits/SMPyBandits
    :target: https://github.com/SMPyBandits/SMPyBandits/issues?q=is%3Aclosed+is%3Aissue
.. |Pull requests of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/prs/SMPyBandits/SMPyBandits
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/pulls
.. |Open pull requests of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/open-prs/SMPyBandits/SMPyBandits
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/issues?q=is%3Aopen+is%3Apr
.. |Closed pull requests of https://github.com/SMPyBandits/SMPyBandits/| image:: https://badgen.net/github/closed-prs/SMPyBandits/SMPyBandits
    :target: https://GitHub.com/SMPyBandits/SMPyBandits/issues?q=is%3Aclose+is%3Apr
