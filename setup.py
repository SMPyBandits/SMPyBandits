#!/usr/bin/env python
# -*- coding: utf-8; mode: python -*-
"""
setup.py script for the SMPyBandits project (https://github.com/SMPyBandits/SMPyBandits)

References:
- https://packaging.python.org/en/latest/distributing/#setup-py
- https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/creation.html#setup-py-description
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = "SMPyBandits: Open-Source Python package for Single- and Multi-Players multi-armed Bandits algorithms."
README = path.join(here, "SMPyBandits", "README.rst")
if path.exists(README):
    with open(README, encoding="utf-8") as f:
        long_description = f.read()
        # print("Using a long_description of length,", len(long_description), "from file", README)  # DEBUG

version = "0.9.6"
try:
    from SMPyBandits import __version__ as version
except ImportError:
    print("Error: cannot import version from SMPyBandits.")
    print("Are you sure you are building in the correct folder?")

# FIXME revert when done uploading the first version to PyPI
# version = "0.0.2.dev2"


setup(name="SMPyBandits",
    version=version,
    description="SMPyBandits: Open-Source Python package for Single- and Multi-Players multi-armed Bandits algorithms.",
    long_description=long_description,
    author="Lilian Besson",
    author_email="naereen AT crans DOT org".replace(" AT ", "@").replace(" DOT ", "."),
    url="https://github.com/SMPyBandits/SMPyBandits/",
    download_url="https://github.com/SMPyBandits/SMPyBandits/releases/",
    license="MIT",
    platforms=["GNU/Linux"],
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="multi-arm-bandits simulations learning-theory centralized-algorithms decentralized-algorithms cognitive-radio",
    # py_modules=["SMPyBandits"],
    packages=[
        "SMPyBandits",
        "SMPyBandits.Arms",
        "SMPyBandits.Policies",
        "SMPyBandits.Policies.Posterior",
        "SMPyBandits.PoliciesMultiPlayers",
        "SMPyBandits.Environment",
    ],
    install_requires=[
        "numpy",
        "scipy > 0.9",
        "matplotlib >= 2",
        "joblib",
        "seaborn",
        "scikit-learn",
        "scikit-optimize",
    ],
    extras_require={
        "full": [
            "tqdm",
            "numba",
            "docopt",
            "ipython",
        ],
        "doc": [
            "sphinx_rtd_theme",
            "recommonmark",
            "nbsphinx",
            "pyreverse",
        ]
    },
    package_data={
        'SMPyBandits': [
            'LICENSE',
            'README.rst',
        ]
    },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/SMPyBandits/SMPyBandits/issues",
        "Source":      "https://github.com/SMPyBandits/SMPyBandits/tree/master/",
        "Documentation": "https://smpybandits.github.io/",
    },
)

# End of setup.py
