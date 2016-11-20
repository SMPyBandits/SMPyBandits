# Quick Makefile to:
# - run easily a Python script, while keeping a text log of its full output (make run)
# - lint the Python code (make lint lint3)
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

# Stats
stats:
	git-complete-stats.sh | tee complete-stats.txt
	git-cal --ascii | tee -a complete-stats.txt
	git wdiff complete-stats.txt

# Linters
# NPROC = `nproc`
# NPROC = 1
NPROC = `getconf _NPROCESSORS_ONLN`

lint:
	pylint -j $(NPROC) ./*.py ./*/*.py | tee ./pylint.log.txt

lint3:
	pylint --py3k -j $(NPROC) ./*.py ./*/*.py | tee ./pylint3.log.txt

2to3:
	-echo "FIXME this does not work from make (Makefile), but work from Bash"
	echo 'for i in {,*/}*.py; do clear; echo $i; 2to3 -p $i 2>&1 | grep -v "root:" | colordiff ; read; done'
