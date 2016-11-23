# Quick Makefile to:
# - run easily a Python script, while keeping a text log of its full output (make run)
# - lint the Python code (make lint lint3)
# - install the requirements (make install)

run:
	make clean ; clear ; make main3

# Runners
main:
	time nice -n 20 ipython2 ./main.py | tee ./main_py2_log.txt
	# time nice -n 20 python2 ./main.py | tee ./main_py2_log.txt

main3:
	time nice -n 20 ipython3 ./main.py | tee ./main_py3_log.txt
	# time nice -n 20 python3 ./main.py | tee ./main_py3_log.txt

# Time profilers
profile:
	time nice -n 20 python2 -m cProfile -s cumtime ./main.py | tee ./main_py2_profile_log.txt
	# time nice -n 20 python2 -m cProfile -s cumtime ./main.py | tee ./main_py2_profile_log.txt

profile3:
	time nice -n 20 python3 -m cProfile -s cumtime ./main.py | tee ./main_py3_profile_log.txt
	# time nice -n 20 python3 -m cProfile -s cumtime ./main.py | tee ./main_py3_profile_log.txt

# Line time profilers
line_profile:
	time nice -n 20 python2 -m line_profile -s cumtime ./main.py | tee ./main_py2_line_profile_log.txt
	# time nice -n 20 python2 -m line_profile -s cumtime ./main.py | tee ./main_py2_line_profile_log.txt

profile3:
	time nice -n 20 python3 -m line_profile -s cumtime ./main.py | tee ./main_py3_line_profile_log.txt
	# time nice -n 20 python3 -m line_profile -s cumtime ./main.py | tee ./main_py3_line_profile_log.txt

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
	pylint -j $(NPROC) ./*.py ./*/*.py | tee ._pylint_log.txt.txt

lint3:
	pylint --py3k -j $(NPROC) ./*.py ./*/*.py | tee ._pylint3_log.txt.txt

2to3:
	-echo "FIXME this does not work from make (Makefile), but work from Bash"
	echo 'for i in {,*/}*.py; do clear; echo $i; 2to3 -p $i 2>&1 | grep -v "root:" | colordiff ; read; done'

pyreverse:
	pyreverse -o dot -my -f ALL -p AlgoBandits ./*.py ./*/*.py
	# Output packages and classes graphs to SVG...
	dot -Tsvg packages_AlgoBandits.dot > packages_AlgoBandits.svg
	dot -Tsvg classes_AlgoBandits.dot > classes_AlgoBandits.svg
	# Output packages and classes graphs to PNG...
	dot -Tpng packages_AlgoBandits.dot > packages_AlgoBandits.png
	dot -Tpng classes_AlgoBandits.dot > classes_AlgoBandits.png
	# Output packages and classes graphs to PDF...
	# dot -Tpdf packages_AlgoBandits.dot > packages_AlgoBandits.pdf
	# dot -Tpdf classes_AlgoBandits.dot > classes_AlgoBandits.pdf
