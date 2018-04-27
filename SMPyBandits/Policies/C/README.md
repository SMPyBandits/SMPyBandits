# Fast C versions of the utilities in [`kullpack.py`](../kullback.py)

## Prefer the Cython version?
WARNING: I have now written a Cython version of this module, see [`kullback_cython.pyx`](../kullback_cython.pyx).
It has all the advantages of the C version (speed and memory efficiency), and all the advantages of the Python version (documentation, optional arguments).

You can have a look to the first examples in [`kullback_cython.pyx`](../kullback_cython.pyx) to see a small comparison between the Cython and C versions.

TL;DR: I don't recommend that you try using this C version, it's not worth it: the C version is only 2 times faster than the Cython one, and both are between 100 to 200 times faster than the naive Python versions!

### Requirements?
You need either the `cython` package for your version of Python (if you want to compile the  [`kullback_cython.pyx`](../kullback_cython.pyx) file before running your extension), or both the `cython` and `pyximport` packages, if you want to be able to directly import the Cython version with:

```python
>>> import pyximport; pyximport.install()
>>> import kullback_cython as kullback
>>> # then use kullback.klucbBern or others, as if they came from the pure Python version!
```

---

## Build it
To create the module use

```bash
python setup.py build
python3 setup.py build
```

Or simply use the provided [`Makefile`](Makefile):

```bash
make build
```

The compiled module (`.so` file) will appear in `build/lib.???` (typically `yoursys-yourarch-yourversion`).

## Clean-up
Temporary files in `build/temp.*` can be removed with

```bash
python setup.py clean
python3 setup.py clean
```

Or simply use the provided [`Makefile`](Makefile):

```bash
make build
```

## Requirements
Building requires the header files and static library, typically available in a package called `python-dev` (on Linux systems).
See [the Python documentation](https://docs.python.org/3/c-api/) for more details.
