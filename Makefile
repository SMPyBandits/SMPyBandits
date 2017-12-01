# Quick Makefile to:
# - run easily a Python script, while keeping a text log of its full output (make run)
# - lint the Python code (make lint lint3)
# - install the requirements (make install)

# Using bash and not sh, cf. http://stackoverflow.com/a/589300/
SHELL := /bin/bash -o pipefail

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

single:
	make clean ; clear ; make singleplayer3
multi:
	make clean ; clear ; make multiplayers3
moremulti:
	make clean ; clear ; make moremultiplayers3
sparsemulti:
	make clean ; clear ; make sparsemultiplayers3

alllint:	lint lint3 pyreverse stats doc
doc:	clean-doc
	make html clean send

# Runners
singleplayer:	singleplayer3
singleplayer3:
	time nice -n 19 ipython3 ./main.py | tee ./logs/main_py3_log.txt
singleplayer2:
	time nice -n 19 python2 ./main.py | tee ./logs/main_py2_log.txt

comparing_aggregation_algorithms:	comparing_aggregation_algorithms3
comparing_aggregation_algorithms3:
	time nice -n 19 ipython3 ./main.py configuration_comparing_aggregation_algorithms | tee ./logs/main_py3_log.txt
comparing_aggregation_algorithms2:
	time nice -n 19 python2 ./main.py configuration_comparing_aggregation_algorithms | tee ./logs/main_py2_log.txt

sparse:	sparse3
sparse3:
	time nice -n 19 ipython3 ./main.py configuration_sparse | tee ./logs/main_py3_log.txt
sparse2:
	time nice -n 19 python2 ./main.py configuration_sparse | tee ./logs/main_py2_log.txt

markovian:	markovian3
markovian3:
	time nice -n 19 ipython3 ./main.py configuration_markovian | tee ./logs/main_py3_log.txt
markovian2:
	time nice -n 19 python2 ./main.py configuration_markovian | tee ./logs/main_py2_log.txt

multiplayers:	multiplayers3
multiplayers3:
	time nice -n 19 ipython3 ./main_multiplayers.py | tee ./logs/main_multiplayers_py3_log.txt
multiplayers2:
	time nice -n 19 python2 ./main_multiplayers.py | tee ./logs/main_multiplayers_py2_log.txt

moremultiplayers: moremultiplayers3
moremultiplayers3:
	time nice -n 19 ipython3 ./main_multiplayers_more.py | tee ./logs/main_multiplayers_more_py3_log.txt
moremultiplayers2:
	time nice -n 19 python2 ./main_multiplayers_more.py | tee ./logs/main_multiplayers_more_py2_log.txt

sparsemultiplayers: sparsemultiplayers3
sparsemultiplayers3:
	time nice -n 19 ipython3 ./main_sparse_multiplayers.py | tee ./logs/main_sparse_multiplayers_py3_log.txt
sparsemultiplayers2:
	time nice -n 19 python2 ./main_sparse_multiplayers.py | tee ./logs/main_sparse_multiplayers_py2_log.txt

treeexploration: treeexploration3
treeexploration3:
	time nice -n 19 ipython3 ./complete_tree_exploration_for_MP_bandits.py | tee ./logs/complete_tree_exploration_for_MP_bandits_py3_log.txt
treeexploration2:
	time nice -n 19 python2 ./complete_tree_exploration_for_MP_bandits.py | tee ./logs/complete_tree_exploration_for_MP_bandits_py2_log.txt

policy_server_py:
	clear
	time ./policy_server.py --port=10000 --host=127.0.0.1 '{"nbArms": 9, "archtype": "UCBalpha", "params": { "alpha": 1.0 }}'
	@# time ./policy_server.py --port=10000 --host=127.0.0.1 '{"nbArms": 9, "archtype": "ApproximatedFHGittins", "params": { "alpha": 0.5, "horizon": 10000 }}'
	@# time ./policy_server.py --port=10000 --host=127.0.0.1 '{"nbArms": 9, "archtype": "DoublingTrickWrapper", "params": { "alpha": 0.5, "policy": "ApproximatedFHGittins", "next_horizon": "next_horizon__exponential_slow" }}'
	@# time ./policy_server.py --port=10000 --host=127.0.0.1 '{"nbArms": 9, "archtype": "SparseWrapper", "params": { "policy": "ApproximatedFHGittins", "sparsity": 3, "horizon": 10000, "alpha": 0.5 }}'
	@# time ./policy_server.py --port=10000 --host=127.0.0.1 '{"nbArms": 9, "archtype": "SparseWrapper", "params": { "policy": "DoublingTrickWrapper", "sparsity": 3, "params": { "alpha": 0.5, "policy": "ApproximatedFHGittins", "next_horizon": "next_horizon__exponential_slow" } }}'
	@#time ./policy_server.py --port=10000 --host=127.0.0.1 '{"nbArms": 9, "archtype": "DoublingTrickWrapper", "params": { "policy": "SparseWrapper", "next_horizon": "next_horizon__exponential_slow", "params": { "alpha": 0.5, "policy": "ApproximatedFHGittins", "sparsity": 3 } }}'

env_client_py:
	clear
	# time ./env_client.py --speed=1000 --port=10000 --host=127.0.0.1 '{"arm_type": "Bernoulli", "params": [0, 0, 0, 0, 0, 0, 0, 0.7, 0.8, 0.9]}'
	# A Bayesian problem: every repetition use a different mean vectors!
	time ./env_client.py dynamic --speed=10 --port=10000 --host=127.0.0.1 '{"arm_type": "Bernoulli", "params": {"function": "randomMeans", "args": {"nbArms": 9, "isSorted": false}}}'

env_client.exe:
	clear
	g++ -Wall -Iinclude -o env_client.exe include/docopt.cpp env_client.cpp

env_client_cpp:	env_client.exe
	time ./env_client.exe --speed=1000 --port=10000 --host=127.0.0.1

# test_sub:
# 	clear
# 	g++ -std=c++11 -Iinclude -o test_sub.exe test_sub.cpp -pthread
# 	./test_sub.exe

# Time profilers
profile: profile3
profile2:
	time nice -n 19 python2 -m cProfile -s cumtime ./main.py | tee ./logs/main_py2_profile_log.txt
profile3:
	time nice -n 19 python3 -m cProfile -s cumtime ./main.py | tee ./logs/main_py3_profile_log.txt

# Line time profilers
line_profiler:	kernprof lprof
kernprof:
	@echo "Running the script 'main.py' with the 'kernprof' command line profiler ..."
	@echo "See 'https://github.com/rkern/line_profiler#kernprof' if needed"
	time nice -n 19 kernprof -l ./main.py | tee ./logs/main_py3_log.txt
lprof:
	@echo "Time profile, line by line, for the script 'main.py' ..."
	@echo "See 'https://github.com/rkern/line_profiler#line-profiler' if needed"
	time nice -n 19 python3 -m line_profiler ./main.py.lprof | tee ./logs/main_py3_line_profiler_log.txt

# Python Call Graph, XXX does not work well as far as now
callgraph:
	@echo "Running the script 'main.py' with the pycallgraph command line profiler ..."
	@echo "See 'http://pycallgraph.slowchop.com/en/master/guide/command_line_usage.html#examples' if needed"
	# time nice -n 19 pycallgraph --verbose --max-depth 10 graphviz --output-file=logs/pycallgraph.svg -- ./main.py | tee ./logs/main_pycallgraph_log.txt
	# time nice -n 19 /usr/local/bin/pycallgraph --verbose --threaded --memory graphviz --output-file=logs/pycallgraph.svg -- ./main.py | tee ./logs/main_pycallgraph_log.txt
	time nice -n 19 pycallgraph --verbose --max-depth 10 gephi --output-file=logs/pycallgraph.gdf -- ./main.py | tee ./logs/main_pycallgraph_log.txt
	# -convert logs/pycallgraph.svg logs/pycallgraph.png

# Installers
# FIXME make a virtualenv
install:
	sudo -H pip  install -U -r requirements.txt
install2:
	sudo -H pip2 install -U -r requirements.txt
install3:
	sudo -H pip3 install -U -r requirements.txt

# Senders:
send_ws3:	clean
	CP ~/AlgoBandits.git/ lilian_besson@ws3:~/These/src/AlgoBandits.git/

# receive_plots_ws3:
# 	CP lilian_besson@ws3:~/These/src/AlgoBandits.git/plots ./

# Cleaner
clean:
	-mv -vf *.pyc */*.pyc /tmp/
	-rm -vfr __pycache__/ **/__pycache__/
	-rm -vf *.pyc */*.pyc

# Stats
stats:
	git-complete-stats.sh | tee complete-stats.txt
	git-cal --ascii | tee -a complete-stats.txt
	git wdiff complete-stats.txt

# Backup
ZIPFILE = ~/Dropbox/AlgoBandits.git.zip
zip:	clean
	zip -r -9 -y -v $(ZIPFILE) ./ -x plots/*/ plots/*/*
	zipinfo $(ZIPFILE) | tac
	ls -larth $(ZIPFILE)

# Linters
# NPROC = `nproc`
# NPROC = 1
NPROC = `getconf _NPROCESSORS_ONLN`

lint:
	-pylint -j $(NPROC) ./*.py ./*/*.py | tee ./logs/main_pylint_log.txt
lint3:
	-pylint --py3k -j $(NPROC) ./*.py ./*/*.py | tee ./logs/main_pylint3_log.txt

2to3:
	-echo "FIXME this does not work from make (Makefile), but work from Bash"
	echo 'for i in {,*/}*.py; do clear; echo $i; 2to3 -p $i 2>&1 | grep -v "root:" | colordiff ; read; done'

pyreverse:
	-mkdir uml_diagrams/
	pyreverse -o dot -my -f ALL -p AlgoBandits ./*.py ./*/*.py
	-mv -vf packages_AlgoBandits.dot classes_AlgoBandits.dot uml_diagrams/
	# Output packages and classes graphs to PNG...
	dot -Tpng uml_diagrams/packages_AlgoBandits.dot   > uml_diagrams/packages_AlgoBandits.png
	dot -Tpng uml_diagrams/classes_AlgoBandits.dot    > uml_diagrams/classes_AlgoBandits.png
	-advpng -z -2 ./uml_diagrams/*.png
	# Output packages and classes graphs to SVG...
	dot -Tsvg uml_diagrams/packages_AlgoBandits.dot   > uml_diagrams/packages_AlgoBandits.svg
	dot -Tsvg uml_diagrams/classes_AlgoBandits.dot    > uml_diagrams/classes_AlgoBandits.svg
	# Output packages and classes graphs to PDF...
	dot -Tpdf uml_diagrams/packages_AlgoBandits.dot > uml_diagrams/packages_AlgoBandits.pdf
	dot -Tpdf uml_diagrams/classes_AlgoBandits.dot  > uml_diagrams/classes_AlgoBandits.pdf
	-PDFCompress -f ./uml_diagrams/*.pdf

ignorelogs:
	git checkout -- logs/


# -----------------------------------------
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON        = python3
SPHINXOPTS    =
# SPHINXBUILD   = sphinx-build
SPHINXBUILD   = /home/lilian/publis/sphinx.git/sphinx-build.py
SPHINXPROJ    = AlgoBandits
SOURCEDIR     = .
BUILDDIR      = _build

clean-doc:
	mkdir --parents $(BUILDDIR)/html
	-rm -rfv /tmp/$(BUILDDIR)/
	mv -vf $(BUILDDIR)/ /tmp/
	mkdir --parents $(BUILDDIR)/html/

.PHONY: help

send:	send_zamok send_gforge
send_zamok:
	cd notebooks ; make send_zamok ; cd ..
	CP "$(BUILDDIR)"/html/ ${Szam}phd/AlgoBandits/
	# -ssh ${SZAM} "rm -rfv /tmp/besson/_modules/ ; mv -vf /home/besson/www/phd/AlgoBandits/_modules/ /tmp/besson/"
send_gforge:
	CP "$(BUILDDIR)"/html/ lbesson@scm.gforge.inria.fr:/home/groups/banditslilian/htdocs/
	# -ssh lbesson@scm.gforge.inria.fr "rm -rfv /tmp/banditslilian/_modules/ ; mv -vf /home/groups/banditslilian/htdocs/_modules/ /tmp/banditslilian/"

apidoc:
	-mkdir -vp /tmp/AlgoBandits/docs/
	-mv -vf docs/*.rst /tmp/AlgoBandits/docs/
	# @echo "==> Showing you which .rst files will be created in docs/"
	# sphinx-apidoc -n -o docs -e -M .
	# @echo "==> OK to generate these files ? [Enter for OK, Ctrl+C to cancel]"
	# @read
	sphinx-apidoc -o docs -e -M .

# # Catch-all target: route all unknown targets to Sphinx using the new
# # "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %:
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# \cp notebooks/*.html "$(BUILDDIR)"/html/notebooks/  # no need thanks to http://nbsphinx.readthedocs.io/
	-./notebooks/symlinks.sh
	-./.fixes_html_in_doc.sh
	\cp uml_diagrams/*.svg "$(BUILDDIR)"/html/uml_diagrams/
	\cp logs/*.txt "$(BUILDDIR)"/html/logs/
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."


changes:
	$(SPHINXBUILD) -b changes $(SPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(SPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

coverage:
	$(SPHINXBUILD) -b coverage $(SPHINXOPTS) $(BUILDDIR)/coverage
	@echo "Testing of coverage in the sources finished, look at the " \
	      "results in $(BUILDDIR)/coverage/python.txt."

