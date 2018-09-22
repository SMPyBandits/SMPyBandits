#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    # If in OSX, we need to install python by hand.
    # We do that using homebrew, pyenv and pyenv-virtualenv
    # You should normally not change anything in here
    brew update >/dev/null
    # Per the `pyenv homebrew recommendations <https://github.com/yyuu/pyenv/wiki#suggested-build-environment>`_.
    brew install openssl readline
    # See https://docs.travis-ci.com/user/osx-ci-environment/#A-note-on-upgrading-packages.
    # I didn't do this above because it works and I'm lazy.
    brew outdated pyenv || brew upgrade pyenv
    # virtualenv doesn't work without pyenv knowledge. venv in Python 3.3
    # doesn't provide Pip by default. So, use `pyenv-virtualenv <https://github.com/yyuu/pyenv-virtualenv/blob/master/README.md>`_.
    brew install pyenv-virtualenv
    PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install $PYTHON # Try to fix matplotlib backend issue #6, repeated in tests branch
    # I would expect something like ``pyenv init; pyenv local $PYTHON`` or
    # ``pyenv shell $PYTHON`` would work, but ``pyenv init`` doesn't seem to
    # modify the Bash environment. ??? So, I hand-set the variables instead.
    export PYENV_VERSION=$PYTHON
    export PATH="/Users/travis/.pyenv/shims:${PATH}"
    pyenv-virtualenv venv
    source venv/bin/activate
    # A manual check that the correct version of Python is running.
    python --version
else
    # Additional installation instructions for UNIX
    sudo apt-get install -qq gcc g++
    echo "Not on osx"
fi