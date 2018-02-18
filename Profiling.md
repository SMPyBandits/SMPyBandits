## A note on execution times, speed and profiling
- About (time) profiling with Python (2 or 3): `cProfile` or `profile` [in Python 2 documentation](https://docs.python.org/2/library/profile.html) ([in Python 3 documentation](https://docs.python.org/2/library/profile.html)), [this StackOverflow thread](https://stackoverflow.com/a/7693928/5889533), [this blog post](https://www.huyng.com/posts/python-performance-analysis), and the documentation of [`line_profiler`](https://github.com/rkern/line_profiler) (to profile lines instead of functions) and [`pycallgraph`](http://pycallgraph.slowchop.com/en/master/) (to illustrate function calls) and [`yappi`](https://pypi.python.org/pypi/yappi/) (which seems to be thread aware).
- See also [`pyreverse`](https://www.logilab.org/blogentry/6883) to get nice UML-like diagrams illustrating the relationships of packages and classes between each-other.

### *A better approach?*
In January, I tried to use the [PyCharm](https://www.jetbrains.com/pycharm/download/) Python IDE, and it has an awesome profiler included!
But it was too cumbersome to use...

### *An even better approach?*
Well now... I know my codebase, and I know how costly or efficient every new piece of code should be, if I find empirically something odd, I explore with one of the above-mentionned module...
