## A note on execution times, speed and profiling
- About (time) profiling with Python (2 or 3): `cProfile` or `profile` [in Python 2 documentation](https://docs.python.org/2/library/profile.html) ([in Python 3 documentation](https://docs.python.org/2/library/profile.html)), [this StackOverflow thread](https://stackoverflow.com/a/7693928/5889533), [this blog post](https://www.huyng.com/posts/python-performance-analysis), and the documentation of [`line_profiler`](https://github.com/rkern/line_profiler) (to profile lines instead of functions) and [`pycallgraph`](http://pycallgraph.slowchop.com/en/master/) (to illustrate function calls) and [`yappi`](https://pypi.python.org/pypi/yappi/) (which seems to be thread aware).
- See also [`pyreverse`](https://www.logilab.org/blogentry/6883) to get nice UML-like diagrams illustrating the relationships of packages and classes between each-other.

### *A better approach?*
In January, I tried to use the [PyCharm](https://www.jetbrains.com/pycharm/download/) Python IDE, and it has an awesome profiler included!
But it was too cumbersome to use...

### *An even better approach?*
Well now... I know my codebase, and I know how costly or efficient every new piece of code should be, if I find empirically something odd, I explore with one of the above-mentionned module...

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

© 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[[Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/SMPyBandits/SMPyBandits/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)
![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master)](https://travis-ci.org/SMPyBandits/SMPyBandits)
[![Stars of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/stars/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/stargazers)
[![Releases of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/release/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/releases)
