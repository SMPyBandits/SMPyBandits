# GNU Make makefile to build the lempel_ziv_complexity_cython.pyx C extension
all: clean ccode build install clean

ccode:
	cython -v -3 lempel_ziv_complexity_cython.pyx

build: build2 build3

build2: lempel_ziv_complexity_cython.c setup.py
	python2 setup.py build

build3: lempel_ziv_complexity_cython.c setup.py3
	python3 setup.py3 build

install:
	\cp build/lib*/lempel_ziv_complexity*.* ../

clean: setup.py
	python2 setup.py clean
	python3 setup.py3 clean

cleanall:
	mv -vf build /tmp/
