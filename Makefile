run:
	make clean ; clear ; make main

clean:
	-mv -vf *.pyc */*.pyc /tmp/
	-rm -vfr __pycache__/ */__pycache__/
	-rm -vf *.pyc */*.pyc /tmp/

main:
	# ipython2 main.py
	python2 ./main.py | tee ./main.py.log
