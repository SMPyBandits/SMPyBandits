*SMPyBandits*
=============

**Open-Source Python package for Single- and Multi-Players multi-armed Bandits algorithms**.

This repository contains the code of `Lilian Besson’s <http://perso.crans.org/besson/>`__ numerical environment, written in `Python (2 or 3) <https://www.python.org/>`__, for numerical simulations on *single*-player and *multi*-players `Multi-Armed Bandits (MAB) <https://en.wikipedia.org/wiki/Multi-armed_bandit>`__ algorithms.

A complete Sphinx-generated documentation is on `SMPyBandits.GitHub.io <https://smpybandits.github.io/>`__.

Quick presentation
------------------

*SMPyBandits* contains the most complete collection of single-player (classical) bandit algorithms on the Internet (`over 65! <https://smpybandits.github.io/docs/Policies.html>`__), as well as implementation of all the state-of-the-art `multi-player algorithms <https://smpybandits.github.io/docs/PoliciesMultiPlayers.html>`__.

I follow very actively the latest publications related to Multi-Armed Bandits (MAB) research, and usually implement quite quickly the new algorithms (see for instance, `Exp3++ <https://smpybandits.github.io/docs/Policies.Exp3PlusPlus.html>`__, `CORRAL <https://smpybandits.github.io/docs/Policies.CORRAL.html>`__ and `SparseUCB <https://smpybandits.github.io/docs/Policies.SparseUCB.html>`__ were each introduced by articles (`for Exp3++ <https://arxiv.org/pdf/1702.06103>`__, `for CORRAL <https://arxiv.org/abs/1612.06246v2>`__, `for SparseUCB <https://arxiv.org/abs/1706.01383>`__) presented at COLT in July 2017, `LearnExp <https://smpybandits.github.io/docs/Policies.LearnExp.html>`__ comes from a `NIPS 2017 paper <https://arxiv.org/abs/1702.04825>`__, and `kl-UCB++ <https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.html>`__ from an `ALT 2017 paper <https://hal.inria.fr/hal-01475078>`__.).

-  Classical MAB have a lot of applications, from clinical trials, A/B testing, game tree exploration, and online content recommendation (my framework does *not* implement contextual bandit - yet).
-  `Multi-player MAB <MultiPlayers.md>`__ have applications in Cognitive Radio, and my framework implements `all the collision models <https://smpybandits.github.io/docs/Environment/CollisionModels.html>`__ found in the literature, as well as all the algorithms from the last 10 years or so (`rhoRand <https://smpybandits.github.io/docs/PoliciesMultiPlayers/rhoRand.html>`__ from 2009, `MEGA <https://smpybandits.github.io/docs/Policies/MEGA.html>`__ from 2015, `MusicalChair <https://smpybandits.github.io/docs/Policies/MusicalChair.html>`__, and our state-of-the-art algorithms `RandTopM <https://smpybandits.github.io/docs/PoliciesMultiPlayers/RandTopM.html>`__ and `MCTopM <https://smpybandits.github.io/docs/PoliciesMultiPlayers/MCTopM.html>`__).

With this numerical framework, simulations can run on a single CPU or a multi-core machine, and summary plots are automatically saved as high-quality PNG, PDF and EPS (ready for being used in research article). Making new simulations is very easy, one only needs to write a configuration script and basically no code! See `these examples <https://github.com/SMPyBandits/SMPyBandits/search?l=Python&q=configuration&type=&utf8=%E2%9C%93>`__ (files named ``configuration_...py``).

A complete `Sphinx <http://sphinx-doc.org/>`__ documentation for each algorithms and every piece of code (included constants in the configurations!) is available here: `SMPyBandits.GitHub.io <https://smpybandits.github.io/>`__.

|PyPI implementation| |PyPI pyversions| |Maintenance| |Ask Me Anything|


.. note::

    - `I (Lilian Besson) <http://perso.crans.org/besson/>`__ have `started my PhD <http://perso.crans.org/besson/phd/>`__ in October 2016, and this is a part of my **on going** research since December 2016.
    - I launched the `documentation <https://smpybandits.github.io/>`__ on March 2017, I wrote my first research articles using this framework in 2017 and decided to (finally) open-source my project in February 2018.

--------------

How to cite this work?
----------------------

If you use this package for your own work, please consider citing it with `this piece of BibTeX <https://github.com/SMPyBandits/SMPyBandits/raw/master/SMPyBandits.bib>`__:

.. code:: bibtex

    @misc{SMPyBandits,
        title =   {{SMPyBandits: an Open-Source Research Framework for Single and Multi-Players Multi-Arms Bandits (MAB) Algorithms in Python}},
        author =  {Lilian Besson},
        year =    {2018},
        url =     {https://github.com/SMPyBandits/SMPyBandits/},
        howpublished = {Online at: \url{github.com/SMPyBandits/SMPyBandits}},
        note =    {Code at https://github.com/SMPyBandits/SMPyBandits/, documentation at https://smpybandits.github.io/}
    }

I also wrote a small paper to present *SMPyBandits*, and I will send it to `JMLR MLOSS <http://jmlr.org/mloss/>`__. The paper can be consulted `here on my website <https://perso.crans.org/besson/articles/SMPyBandits.pdf>`__.

--------------

Other interesting things
------------------------

`Single-player Policies <https://smpybandits.github.io/docs/Policies.html>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  More than 65 algorithms, including all known variants of the `UCB <https://smpybandits.github.io/docs/Policies/UCB.html>`__, `kl-UCB <https://smpybandits.github.io/docs/Policies/klUCB.html>`__, `MOSS <https://smpybandits.github.io/docs/Policies/MOSS.html>`__ and `Thompson Sampling <https://smpybandits.github.io/docs/Policies/Thompson.html>`__ algorithms, as well as other less known algorithms (`OCUCB <https://smpybandits.github.io/docs/Policies/OCUCB.html>`__, `BESA <https://smpybandits.github.io/docs/Policies/OCUCB.html>`__, `OSSB <https://smpybandits.github.io/docs/Policies/OSSB.html>`__ etc).
-  `SparseWrapper <https://smpybandits.github.io/docs/Policies.SparseWrapper.html#module-Policies.SparseWrapper>`__ is a generalization of `the SparseUCB from this article <https://arxiv.org/pdf/1706.01383/>`__.
-  Implementation of very recent Multi-Armed Bandits algorithms, e.g., `kl-UCB++ <https://smpybandits.github.io/docs/Policies.klUCBPlusPlus.html>`__ (from `this article <https://hal.inria.fr/hal-01475078>`__), `UCB-dagger <https://smpybandits.github.io/docs/Policies.UCBdagger.html>`__ (from `this article <https://arxiv.org/pdf/1507.07880>`__), or `MOSS-anytime <https://smpybandits.github.io/docs/Policies.MOSSAnytime.html>`__ (from `this article <http://proceedings.mlr.press/v48/degenne16.pdf>`__).
-  Experimental policies: `BlackBoxOpt <https://smpybandits.github.io/docs/Policies.BlackBoxOpt.html>`__ or `UnsupervisedLearning <https://smpybandits.github.io/docs/Policies.UnsupervisedLearning.html>`__ (using Gaussian processes to learn the arms distributions).

Arms and problems
~~~~~~~~~~~~~~~~~

-  My framework mainly targets stochastic bandits, with arms following
   `Bernoulli <https://smpybandits.github.io/docs/Arms/Bernoulli.html>`__, bounded
   (SMPyBandits/truncated) or unbounded
   `Gaussian <https://smpybandits.github.io/docs/Arms/Gaussian.html>`__,
   `Exponential <https://smpybandits.github.io/docs/Arms/Exponential.html>`__,
   `Gamma <https://smpybandits.github.io/docs/Arms/Gamma.html>`__ or
   `Poisson <https://smpybandits.github.io/docs/Arms/Poisson.html>`__ distributions.
-  The default configuration is to use a fixed problem for N repetitions (e.g. 1000 repetitions, use `MAB.MAB <https://smpybandits.github.io/docs/Environment/MAB.html>`__), but there is also a perfect support for "Bayesian" problems where the mean vector µ1,…,µK change *at every repetition* (see `MAB.DynamicMAB <https://smpybandits.github.io/docs/Environment/MAB.html>`__).
-  There is also a good support for Markovian problems, see `MAB.MarkovianMAB <https://smpybandits.github.io/docs/Environment/MAB.html>`__, even though I didn’t implement any policies tailored for Markovian problems.

Other remarks
~~~~~~~~~~~~~

-  Everything here is done in an imperative, object oriented style. The API of the Arms, Policy and MultiPlayersPolicy classes is documented `in this page <https://smpybandits.github.io/API.html>`__.
-  The code is `clean <https://smpybandits.github.io/logs/main_pylint_log.txt>`__, valid for both `Python 2 <https://smpybandits.github.io/logs/main_pylint_log.txt>`__ and `Python 3 <https://smpybandits.github.io/logs/main_pylint3_log.txt>`__.
-  Some piece of code come from the `pymaBandits <http://mloss.org/software/view/415/>`__ project, but most of them were refactored. Thanks to the initial project!
-  `G.Varoquaux <http://gael-varoquaux.info/>`__\ ’s `joblib <https://joblib.readthedocs.io/>`__ is used for the `Evaluator <https://smpybandits.github.io/docs/Environment/Evaluator.html>`__ and `EvaluatorMultiPlayers <https://smpybandits.github.io/docs/Environment/EvaluatorMultiPlayers.html>`__ classes, so the simulations are easily parallelized on multi-core machines. (Put ``n_jobs = -1`` or ``PARALLEL = True`` in the config file to use all your CPU cores, as it is by default).

--------------

`How to run the experiments ? <How_to_run_the_code.md>`__
---------------------------------------------------------

    See this document: `How_to_run_the_code.md <https://smpybandits.github.io/How_to_run_the_code.html>`__ for more details.

TL;DR: this short bash snippet shows how to clone the code, install the requirements for Python 3 (in a `virtualenv <https://virtualenv.pypa.io/en/stable/>`__, and starts some simulation for N=100 repetitions of the default non-Bayesian Bernoulli-distributed problem, for K=9 arms, an horizon of T=10000 and on 4 CPUs (it should take about 20 minutes for each simulations):

.. code:: bash

    cd /tmp/
    # just be sure you have the latest virtualenv from Python 3
    sudo pip3 install --upgrade --force-reinstall virtualenv

    # create and active the virtualenv
    virtualenv venv
    . venv/bin/activate
    type pip  # check it is /tmp/venv/bin/pip
    type python  # check it is /tmp/venv/bin/python

    pip install SMPyBandits  # pulls latest version from https://pypi.org/project/SMPyBandits/
    # or you can also
    pip install git+https://github.com/SMPyBandits/SMPyBandits/#egg=SMPyBandits[full]  # pulls latest version from https://github.com/SMPyBandits/SMPyBandits/

    # run a single-player simulation!
    N=100 T=10000 K=9 N_JOBS=4 make single
    # run a multi-player simulation!
    N=100 T=10000 M=3 K=9 N_JOBS=4 make more

..

-  If speed matters to you and you want to use algorithms based on `kl-UCB <https://smpybandits.github.io/docs/Policies/klUCB.html>`__, you should take the time to build and install the fast C implementation of the utilities KL functions. Default is to use `kullback.py <https://smpybandits.github.io/docs/Policies/kullback.html>`__, but using `the C version from Policies/C/ <github.com/SMPyBandits/SMPyBandits/tree/master/SMPyBandits/Policies/C/>`__ really speeds up the computations. Just follow the instructions, it should work well (you need ``gcc`` to be installed).
-  And if speed matters, be sure that you have a working version of `Numba <https://numba.pydata.org/>`__, it is used by many small functions to (try to automatically) speed up the computations.

--------------

Warning
~~~~~~~

-  This work is still **experimental**! It’s `active research <https://github.com/SMPyBandits/SMPyBandits/graphs/contributors>`__. It should be completely bug free and every single module/file should work perfectly(as `this pylint log <https://smpybandits.github.io/logs/main_pylint_log.txt>`__ and `this other one <https://smpybandits.github.io/logs/main_pylint3_log.txt>`__ says), but bugs are sometimes hard to spot so if you encounter any issue, `please fill a bug ticket <https://github.com/SMPyBandits/SMPyBandits/issues/new>`__.
-  Whenever I add a new feature, I run experiments to check that nothing is broken. But *there is no unittest* (I don’t have time). You would have to trust me!
-  This project is NOT meant to be a library that you can use elsewhere, but a research tool. In particular, I don’t take ensure that any of the Python modules can be imported from another directory than the main directory.

Contributing?
-------------

Contributions (issues, questions, pull requests) are of course welcome, but this project is and will stay a personal environment designed for quick research experiments, and will never try to be an industry-ready module for applications of Multi-Armed Bandits algorithms.

If you want to contribute, please have a look to the `CONTRIBUTING <https://smpybandits.github.io/CONTRIBUTING.html>`__ page, and if you want to be more seriously involved, read the `CODE_OF_CONDUCT <https://smpybandits.github.io/CODE_OF_CONDUCT.html>`__ page.

-  You are welcome to `submit an issue <https://github.com/SMPyBandits/SMPyBandits/issues/new>`__, if it was not previously answered,
-  If you have interesting example of use of SMPyBandits, please share it! (`Jupyter Notebooks <https://www.jupyter.org/>`__ are preferred). And fill a pull request to `add it to the notebooks examples <https://smpybandits.github.io/notebooks/README.html>`__.

--------------

License ? |GitHub license|
--------------------------

`MIT Licensed <https://lbesson.mit-license.org/>`__ (file `LICENSE <https://smpybandits.github.io/LICENSE>`__).

© 2016-2018 `Lilian Besson <https://GitHub.com/Naereen>`__.

|Maintenance| |Ask Me Anything| |Analytics| |PyPI implementation|
|PyPI pyversions| |Documentation Status| |ForTheBadge uses-badges| |ForTheBadge uses-git|

|forthebadge made-with-python| |ForTheBadge built-with-science|

.. |PyPI implementation| image:: https://img.shields.io/pypi/implementation/smpybandits.svg
.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/smpybandits.svg
.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity
.. |Ask Me Anything| image:: https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg
   :target: https://GitHub.com/Naereen/ama
.. |GitHub license| image:: https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg
   :target: https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE
.. |Analytics| image:: https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel
   :target: https://GitHub.com/SMPyBandits/SMPyBandits/
.. |Documentation Status| image:: https://readthedocs.org/projects/smpybandits/badge/?version=latest
   :target: https://smpybandits.readthedocs.io/en/latest/?badge=latest
.. |ForTheBadge uses-badges| image:: http://ForTheBadge.com/images/badges/uses-badges.svg
   :target: http://ForTheBadge.com
.. |ForTheBadge uses-git| image:: http://ForTheBadge.com/images/badges/uses-git.svg
   :target: https://GitHub.com/
.. |forthebadge made-with-python| image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/
.. |ForTheBadge built-with-science| image:: http://ForTheBadge.com/images/badges/built-with-science.svg
   :target: https://GitHub.com/Naereen/
