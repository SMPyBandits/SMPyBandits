# Basic Makefile to compile a Cython extension.
# It is used to compile the kullback_cython extension, by running 'make kullback_cython'

kullback_cython3:	kullback_cython
kullback_cython:
	python3 setup.py build_ext --inplace
	mv -vf SMPyBandits/Policies/*.so ./
	chmod -x ./*.so
	chmod g-w ./*.so
	chmod o-w ./*.so
	ls -larth ./*.so
	-trash ./build ./SMPyBandits ./kullback_cython.c

kullback_cython2:
	python2 setup.py build_ext --inplace
	mv -vf SMPyBandits/Policies/*.so ./
	chmod -x ./*.so
	chmod g-w ./*.so
	chmod o-w ./*.so
	ls -larth ./*.so
	-trash ./build ./SMPyBandits ./kullback_cython.c
