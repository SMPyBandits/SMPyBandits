.. AlgoBandits documentation master file, created by
   sphinx-quickstart on Thu Jan 19 17:20:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lilian Besson's "AlgoBandits" project documentation!
===============================================================

A research framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms: UCB, KL-UCB, Thompson... and MusicalChair, ALOHA, MEGA, rhoRand etc.

.. note::

   See more on `the GitHub page for this project <https://naereen.github.io/AlgoBandits/>`_: `<https://naereen.github.io/AlgoBandits/>`_.
   The project is also hosted on `Inria GForge <https://gforge.inria.fr/project/admin/?group_id=9477>`_, and the documentation can be seen online at `<http://banditslilian.gforge.inria.fr/>`_.
   |Website up|

Bandit algorithms, Lilian Besson's "AlgoBandits" project
--------------------------------------------------------

This repository contains the code of `my <http://perso.crans.org/besson/>`_ numerical environment, written in `Python <https://www.python.org/>`_, in order to perform numerical
simulations on *single*-player and *multi*-players `Multi-Armed Bandits
(MAB) <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_ algorithms.

|PyPI implementation| |PyPI pyversions| |MIT license|

`I (Lilian Besson) <http://perso.crans.org/besson/>`_ have `started my
PhD <http://perso.crans.org/besson/phd/>`_ in October 2016, and this is
a part of my **on going** research since December 2016.

|Maintenance| |Ask Me Anything|

----

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   README.md
   docs/modules.rst
   How_to_run_the_code.md
   Aggr.md
   notebooks/README.md
   MultiPlayers.md
   API.md
   Policies/C/README.md
   TODO.md
   plots/README.md
   logs/README.md
   uml_diagrams/README.md
   Profiling.md


.. note::

   This documentation is publically available, but the code is not (yet) open-source.
   I will publish it soon, when it will be stable and clean enough to be used by others.

   |GitHub forks| |GitHub stars| |GitHub watchers|

   |GitHub contributors| |GitHub issues| |GitHub issues-closed|


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


.. note:: Should you use bandits?

   In 2015, `Chris Stucchio advised against <https://www.chrisstucchio.com/blog/2015/dont_use_bandits.html>`_
   the use of bandits, in the context of improving A/B testings,
   opposed to his `2013 blog post <https://www.chrisstucchio.com/blog/2012/bandit_algorithms_vs_ab.html>`_ in favor of bandits, also for A/B testings.
   Both articles are worth reading, but in this research we are not studying A/B testing,
   and it has been already proved how efficient bandit algorithms can be for real-world
   and simulated cognitive radio networks.
   (See for instance `this article by Wassim Jouini, Christophe Moy and Jacques Palicot <https://scholar.google.com/scholar?q=Multi-armed+bandit+based+policies+for+cognitive+radio%27s+decision+making+issues+by+W+Jouini%2C+D+Ernst%2C+C+Moy%2C+J+Palicot+2009&btnG=&hl=fr&as_sdt=0%2C39>`_, `["Multi-armed bandit based policies for cognitive radio's decision making issues", W Jouini, D Ernst, C Moy, J Palicot 2009] <http://orbi.ulg.be/bitstream/2268/16757/1/SCS09_Jouini_Wassim.pdf>`_).


|made-with-latex| |made-with-sphinx|

|ForTheBadge uses-badges| |ForTheBadge uses-git|
|forthebadge made-with-python| |ForTheBadge built-with-science|


.. |PyPI implementation| image:: https://img.shields.io/pypi/implementation/ansicolortags.svg
.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/ansicolortags.svg
.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity
.. |Ask Me Anything| image:: https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg
   :target: https://GitHub.com/Naereen/ama
.. |Website up| image:: https://img.shields.io/website-up-down-green-red/http/banditslilian.gforge.inria.fr.svg
   :target: http://banditslilian.gforge.inria.fr/
.. |made-with-latex| image:: https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg
   :target: https://www.latex-project.org/
.. |made-with-sphinx| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://www.sphinx-doc.org/
.. |GitHub forks| image:: https://img.shields.io/github/forks/Naereen/AlgoBandits.svg?style=social&label=Fork&maxAge=2592000
   :target: https://GitHub.com/Naereen/AlgoBandits/network/
.. |GitHub stars| image:: https://img.shields.io/github/stars/Naereen/AlgoBandits.svg?style=social&label=Star&maxAge=2592000
   :target: https://GitHub.com/Naereen/AlgoBandits/stargazers/
.. |GitHub watchers| image:: https://img.shields.io/github/watchers/Naereen/AlgoBandits.svg?style=social&label=Watch&maxAge=2592000
   :target: https://GitHub.com/Naereen/AlgoBandits/watchers/
.. |GitHub contributors| image:: https://img.shields.io/github/contributors/Naereen/AlgoBandits.svg
   :target: https://GitHub.com/Naereen/AlgoBandits/graphs/contributors/
.. |GitHub issues| image:: https://img.shields.io/github/issues/Naereen/AlgoBandits.svg
   :target: https://GitHub.com/Naereen/AlgoBandits/issues/
.. |GitHub issues-closed| image:: https://img.shields.io/github/issues-closed/Naereen/AlgoBandits.svg
   :target: https://GitHub.com/Naereen/AlgoBandits/issues?q=is%3Aissue+is%3Aclosed
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
