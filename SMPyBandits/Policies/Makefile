# Basic Makefile to compile a Cython extension.
# It is used to compile the cython_extensions extension, by running 'make cython_extensions'

cython_extensions3:	cython_extensions
cython_extensions:
	python3 setup.py build_ext --inplace
	-cp -vf SMPyBandits/Policies/*.so ./
	-chmod -x ./*.so
	-chmod g-w ./*.so
	-chmod o-w ./*.so
	-ls -larth ./*.so
	-rm -vfr ./build ./*.c
	# -mv -vf ./SMPyBandits /tmp/

cython_extensions2:
	python2 setup.py build_ext --inplace
	-cp -vf SMPyBandits/Policies/*.so ./
	-chmod -x ./*.so
	-chmod g-w ./*.so
	-chmod o-w ./*.so
	-ls -larth ./*.so
	-rm -vfr ./build ./*.c
	# -mv -vf ./SMPyBandits /tmp/
