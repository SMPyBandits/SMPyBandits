# Quick Makefile to:
# - compile easily a LaTeX-powered research report (make pdf)
# - run easily a Python script, while keeping a text log of its full output (make nicetest)
# - install the requirements (make install)

run:
	make clean ; clear ; make main3

# Runners
main:
	time nice -n 20 ipython2 ./main.py
	# time nice -n 20 python2 ./main.py | tee ./main.py.log

main3:
	time nice -n 20 ipython3 ./main.py
	# time nice -n 20 python3 ./main.py | tee ./main.py.log

# Installers
install:
	sudo pip  install -r requirements.txt

install3:
	sudo pip3 install -r requirements.txt

# Cleaner
clean:
	-mv -vf *.pyc */*.pyc /tmp/
	-rm -vfr __pycache__/ */__pycache__/
	-rm -vf *.pyc */*.pyc /tmp/

# Linters
# NPROC = `nproc`
# NPROC = 1
NPROC = `getconf _NPROCESSORS_ONLN`

lint:
	pylint -j $(NPROC) ./*.py ./*/*.py | tee ./pylint.log.txt

lint3:
	pylint --py3k -j $(NPROC) ./*.py ./*/*.py | tee ./pylint3.log.txt
