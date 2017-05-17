# GNU Make makefile to build the kullback C extension
all: clean build install clean

build: build2 build3

build2: kullback.c setup.py
	python2 setup.py build

build3: kullback_py3.c setup.py3
	python3 setup.py3 build

install:
	\cp build/lib*/kullback.* ../

clean: setup.py
	python2 setup.py clean
	python3 setup.py3 clean
	#rm -rvf build/*
