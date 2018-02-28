.. SMPyBandits documentation master file, created by
   sphinx-quickstart on Thu Jan 19 17:20:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SMPyBandits documentation!
=====================================

**Open-Source Python package for Single- and Multi-Players multi-armed Bandits algorithms**.

A research framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms: UCB, KL-UCB, Thompson and many more for single-players, and MCTopM & RandTopM, MusicalChair, ALOHA, MEGA, rhoRand for multi-players simulations.
Runs on Python 2 and 3, open-source under the `MIT License <https://lbeson.mit-license.org/>`_.

.. note::

   See more on `the GitHub page for this project <https://github.com/SMPyBandits/SMPyBandits/>`_: `<https://github.com/SMPyBandits/SMPyBandits/>`_.
   The project is also hosted on `Inria GForge <https://gforge.inria.fr/project/admin/?group_id=9477>`_, and the documentation can be seen online at `<http://banditslilian.gforge.inria.fr/>`_.
   |Website up|

Bandit algorithms, SMPyBandits
--------------------------------------------------------

This repository contains the code of `my <http://perso.crans.org/besson/>`_ numerical environment, written in `Python <https://www.python.org/>`_, in order to perform numerical
simulations on *single*-player and *multi*-players `Multi-Armed Bandits
(MAB) <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_ algorithms.

|PyPI implementation| |PyPI pyversions| |MIT license|

`I (Lilian Besson) <http://perso.crans.org/besson/>`_ have `started my
PhD <http://perso.crans.org/besson/phd/>`_ in October 2016, and this is
a part of my **on going** research since December 2016.

|Maintenance| |Ask Me Anything|


How to cite this work?
~~~~~~~~~~~~~~~~~~~~~~
If you use this package for your own work, please consider citing it with this piece of BibTeX: ::

    @misc{SMPyBandits,
        title =   {{SMPyBandits: an Open-Source Research Framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms in Python}},
        author =  {Lilian Besson},
        year =    {2018},
        url =     {https://github.com/SMPyBandits/SMPyBandits/},
        howpublished = {Online at: \url{GitHub.com/SMPyBandits/SMPyBandits}},
        note =    {Code at https://github.com/SMPyBandits/SMPyBandits/, documentation at http://banditslilian.gforge.inria.fr/}
    }

I also wrote a small paper to present *SMPyBandits*, and I will send it to `JOSS <http://joss.theoj.org/>`_.
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
   API.md
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

   Both this documentation and the code are publically available avaible, under the `open-source MIT License <https://lbesson.mit-license.org/>`_.
   The code is hosted on GitHub at `github.com/SMPyBandits/SMPyBandits <https://github.com/SMPyBandits/SMPyBandits/>`_.

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


|made-with-latex| |made-with-sphinx|

|ForTheBadge uses-badges| |ForTheBadge uses-git|
|forthebadge made-with-python| |ForTheBadge built-with-science|


.. |PyPI implementation| image:: https://img.shields.io/pypi/implementation/ansicolortags.svg
.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/ansicolortags.svg
.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity
.. |Ask Me Anything| image:: https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg
   :target: https://GitHub.com/Naereen/ama
.. |Website up| image:: https://img.shields.io/website-up-down-green-red/http/banditslilian.gforge.inria.fr.svg
   :target: http://banditslilian.gforge.inria.fr/
.. |made-with-latex| image:: https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg
   :target: https://www.latex-project.org/
.. |made-with-sphinx| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://www.sphinx-doc.org/
.. |GitHub forks| image:: https://img.shields.io/github/forks/SMPyBandits/SMPyBandits.svg?style=social&label=Fork&maxAge=2592000
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/network/
.. |GitHub stars| image:: https://img.shields.io/github/stars/SMPyBandits/SMPyBandits.svg?style=social&label=Star&maxAge=2592000
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/stargazers/
.. |GitHub watchers| image:: https://img.shields.io/github/watchers/SMPyBandits/SMPyBandits.svg?style=social&label=Watch&maxAge=2592000
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/watchers/
.. |GitHub contributors| image:: https://img.shields.io/github/contributors/SMPyBandits/SMPyBandits.svg
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/graphs/contributors/
.. |GitHub issues| image:: https://img.shields.io/github/issues/SMPyBandits/SMPyBandits.svg
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/issues/
.. |GitHub issues-closed| image:: https://img.shields.io/github/issues-closed/SMPyBandits/SMPyBandits.svg
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/issues?q=is%3Aissue+is%3Aclosed
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
