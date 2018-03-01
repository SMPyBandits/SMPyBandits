# Fast C versions of the utilities in [`kullpack.py`](../kullback.py)
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
See [the Python documentation](https://docs.python.org/3.6/c-api/) for more details.
