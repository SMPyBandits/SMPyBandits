# :boom: TODO
> For others things to do, and issues to solve, see [the issue tracker on GitHub](https://github.com/SMPyBandits/SMPyBandits/issues).

---

## Publicly release it and document it - OK

## Other aspects
- [x] publish on GitHub!

## Presentation paper
- [x] A summary describing the high-level functionality and purpose of the software
for a diverse, non-specialist audience
- [x] A clear statement of need that illustrates the purpose of the software
- [x] A list of key references including a link to the software archive
- [x] Mentions (if applicable) of any ongoing research projects using the software
or recent scholarly publications enabled by it

---

## Clean up things - OK

## Initial things to do! - OK

## Improve and speed-up the code? - OK

---

## More single-player MAB algorithms? - OK

## Contextual bandits?
- [ ] I should try to add support for (basic) contextual bandit.

## Better storing of the simulation results
- [ ] use [hdf5](https://www.hdfgroup.org/HDF5/) (with [`h5py`](http://docs.h5py.org/en/latest/quick.html#core-concepts)) to store the data, *on the run* (to never lose data, even if the simulation gets killed).
- [ ] even more "secure": be able to *interrupt* the simulation, *save* its state and then *load* it back if needed (for instance if you want to leave the office for the weekend).

---

## Multi-players simulations - OK

### Other Multi-Player algorithms
- [ ] ["Dynamic Musical Chair"](https://arxiv.org/abs/1512.02866) that regularly reinitialize "Musical Chair"...
- [ ] ["TDFS"](https://arxiv.org/abs/0910.2065v3) from [[Liu & Zhao, 2009]](https://arxiv.org/abs/0910.2065v3).

### Dynamic settings
- [ ] add the possibility to have a varying number of dynamic users for multi-users simulations…
- [ ] implement the experiments from [Musical Chair], [rhoRand] articles, and Navik Modi's experiments?

---

## C++ library / bridge to C++
- [ ] Finish to write a perfectly clean CLI client to my Python server
- [ ] Write a small library that can be included in any other C++ program to do : 1. start the socket connexion to the server, 2. then play one step at a time,
- [ ] Check that the library can be used within a GNU Radio block !

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/SMPyBandits/SMPyBandits/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)](https://pypi.org/project/SMPyBandits)
![![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)](https://pypi.org/project/SMPyBandits)
![![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg)](https://pypi.org/project/SMPyBandits)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master)](https://travis-ci.org/SMPyBandits/SMPyBandits)
[![Stars of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/stars/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/stargazers)
[![Releases of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/release/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/releases)
