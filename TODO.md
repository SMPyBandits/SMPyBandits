# :boom: TODO
> For others things to do, and issues to solve, see [the issue tracker on GitHub](https://github.com/Naereen/AlgoBandits/issues).

---

## Publicly release it and document it - OK

## Other aspects
- [ ] publish on GitHub!

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
- [ ] add the possibility to have a varying number of dynamic users for multi-users simulations
- [ ] implement the experiments from [Musical Chair], [rhoRand] articles, and Navik Modi's experiments

---

## C++ library
- [ ] Finish to write a perfectly clean CLI client to my Python server
- [ ] Write a small library that can be included in any other C++ program to do : 1. start the socket connexion to the server, 2. then play one step at a time,
- [ ] Check that the library can be used within a GNU Radio block !

----

## :scroll: License ? [![GitHub license](https://img.shields.io/github/license/Naereen/AlgoBandits.svg)](https://github.com/Naereen/AlgoBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/AlgoBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/Naereen/AlgoBandits/README.md?pixel)](https://GitHub.com/Naereen/AlgoBandits/)
![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
[![ForTheBadge uses-badges](http://ForTheBadge.com/images/badges/uses-badges.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)
