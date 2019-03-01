# Quick Makefile to:
# - run easily a Python script, while keeping a text log of its full output (make run)
# - lint the Python code (make lint lint3)
# - generate documentation (make apidoc doc)
# - install the requirements (make install)

# __author__ = "Lilian Besson"
# __version__ = "0.9"

# Using bash and not sh, cf. http://stackoverflow.com/a/589300/
SHELL := /bin/bash -o pipefail

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

example: example_single
example_single:
	make clean ; clear ; cd ./SMPyBandits/ ; time ipython3 example_of_main_singleplayer.py
very_simple:
	make clean ; clear ; cd ./SMPyBandits/ ; time ipython3 example_of_main_singleplayer.py very_simple_configuration
example_multi:
	make clean ; clear ; cd ./SMPyBandits/ ; time ipython3 example_of_main_multiplayers_more.py

single:
	make clean ; clear ; make singleplayer3
multi:
	make clean ; clear ; make multiplayers3
moremulti:
	make clean ; clear ; make moremultiplayers3
moremulti_with_aggregation:
	make clean ; clear ; make moremultiplayers_with_aggregation3
sparsemulti:
	make clean ; clear ; make sparsemultiplayers3

alllint:	lint lint3 uml stats doc
doc:	ignorelogs
	make html clean send
pushdoctogithub:
	cd ~/SMPyBandits.github.io/ ; git add . ; git commit -m "Automated update of the docs from 'make pushdoctogithub'." ; git push

# Runners
singleplayer:	singleplayer3
singleplayer3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py | tee ./logs/main_py3_log.txt
singleplayer2:
	time nice -n 19 python2 ./SMPyBandits/main.py | tee ./logs/main_py2_log.txt

all_singleplayer:	all_singleplayer3
all_singleplayer3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_all_singleplayer | tee ./logs/main_all_singleplayer_py3_log.txt
all_singleplayer2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_all_singleplayer | tee ./logs/main_all_singleplayer_py2_log.txt

comparing_aggregation_algorithms:	comparing_aggregation_algorithms3
comparing_aggregation_algorithms3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_comparing_aggregation_algorithms | tee ./logs/main_comparing_aggregation_algorithms_py3_log.txt
comparing_aggregation_algorithms2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_comparing_aggregation_algorithms | tee ./logs/main_comparing_aggregation_algorithms_py2_log.txt

comparing_doubling_algorithms:	comparing_doubling_algorithms3
comparing_doubling_algorithms3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_comparing_doubling_algorithms | tee ./logs/main_comparing_doubling_algorithms_py3_log.txt
comparing_doubling_algorithms2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_comparing_doubling_algorithms | tee ./logs/main_comparing_doubling_algorithms_py2_log.txt

nonstationary:	nonstationary3
nonstationary3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_nonstationary | tee ./logs/main_nonstationary_py3_log.txt
nonstationary2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_nonstationary | tee ./logs/main_nonstationary_py2_log.txt

sparse:	sparse3
sparse3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_sparse | tee ./logs/main_sparse_py3_log.txt
sparse2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_sparse | tee ./logs/main_sparse_py2_log.txt

markovian:	markovian3
markovian3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_markovian | tee ./logs/main_markovian_py3_log.txt
markovian2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_markovian | tee ./logs/main_markovian_py2_log.txt

multiplayers:	multiplayers3
multiplayers3:
	time nice -n 19 ipython3 ./SMPyBandits/main_multiplayers.py | tee ./logs/main_multiplayers_py3_log.txt
multiplayers2:
	time nice -n 19 python2 ./SMPyBandits/main_multiplayers.py | tee ./logs/main_multiplayers_py2_log.txt

multiplayers_nonstationary:	multiplayers_nonstationary3
multiplayers_nonstationary3:
	time nice -n 19 ipython3 ./SMPyBandits/main_multiplayers.py configuration_multiplayers_nonstationary | tee ./logs/main_multiplayers_nonstationary_py3_log.txt
multiplayers_nonstationary2:
	time nice -n 19 python2 ./SMPyBandits/main_multiplayers.py configuration_multiplayers_nonstationary | tee ./logs/main_multiplayers_nonstationary_py2_log.txt

moremultiplayers: moremultiplayers3
moremultiplayers3:
	time nice -n 19 ipython3 ./SMPyBandits/main_multiplayers_more.py | tee ./logs/main_multiplayers_more_py3_log.txt
moremultiplayers2:
	time nice -n 19 python2 ./SMPyBandits/main_multiplayers_more.py | tee ./logs/main_multiplayers_more_py2_log.txt

moremultiplayers_nonstationary: moremultiplayers_nonstationary3
moremultiplayers_nonstationary3:
	time nice -n 19 ipython3 ./SMPyBandits/main_multiplayers_more.py configuration_multiplayers_nonstationary | tee ./logs/main_multiplayers_nonstationary_more_py3_log.txt
moremultiplayers_nonstationary2:
	time nice -n 19 python2 ./SMPyBandits/main_multiplayers_more.py configuration_multiplayers_nonstationary | tee ./logs/main_multiplayers_nonstationary_more_py2_log.txt

moremultiplayers_with_aggregation: moremultiplayers_with_aggregation3
moremultiplayers_with_aggregation3:
	time nice -n 19 ipython3 ./SMPyBandits/main_multiplayers_more.py configuration_multiplayers_with_aggregation | tee ./logs/main_multiplayers_with_aggregation_py3_log.txt
moremultiplayers_with_aggregation2:
	time nice -n 19 python2 ./SMPyBandits/main_multiplayers_more.py configuration_multiplayers_with_aggregation | tee ./logs/main_multiplayers_with_aggregation_py2_log.txt

sparsemultiplayers: sparsemultiplayers3
sparsemultiplayers3:
	time nice -n 19 ipython3 ./SMPyBandits/main_sparse_multiplayers.py | tee ./logs/main_sparse_multiplayers_py3_log.txt
sparsemultiplayers2:
	time nice -n 19 python2 ./SMPyBandits/main_sparse_multiplayers.py | tee ./logs/main_sparse_multiplayers_py2_log.txt

treeexploration: treeexploration3
treeexploration3:
	time nice -n 19 ipython3 ./SMPyBandits/complete_tree_exploration_for_MP_bandits.py | tee ./logs/complete_tree_exploration_for_MP_bandits_py3_log.txt
treeexploration2:
	time nice -n 19 python2 ./SMPyBandits/complete_tree_exploration_for_MP_bandits.py | tee ./logs/complete_tree_exploration_for_MP_bandits_py2_log.txt

# --------------------------------------------------------
# Build and upload to PyPI
build_for_pypi:	clean_pypi_build sdist wheel

test_twine:
	twine upload --sign --repository testpypi dist/*.whl
twine:
	twine upload --sign --repository pypi dist/*.whl

clean_pypi_build:
	-mv -vf dist/* /tmp/
sdist:	sdist.zip sdist.tar.gz
sdist.zip:
	python3 setup.py sdist --formats=zip
	# -gpg --detach-sign -a dist/*.zip
	-ls -larth dist/*.zip
sdist.tar.gz:
	python3 setup.py sdist --formats=gztar
	# -gpg --detach-sign -a dist/*.tar.gz
	-ls -larth dist/*.tar.gz
wheel:
	python3 setup.py bdist_wheel --universal
	# -gpg --detach-sign -a dist/*.whl
	-ls -larth dist/*.whl

# --------------------------------------------------------

policy_server_py:
	clear
	# cd SMPyBandits ; time ./policy_server.py --port=10000 --host=127.0.0.1 --means='[0,0,0,0,0,0,0.7,0.8,0.9]' '{"nbArms": 9, "archtype": "UCBalpha", "params": { "alpha": 1.0 }}'
	cd SMPyBandits ; time ./policy_server.py --port=10000 --host=127.0.0.1 --means='[0.1,0.2,0.3,0.4,0.5,0.6,0.85,0.87,0.9]' '{"nbArms": 9, "archtype": "klUCBswitchAnytime", "params": { }}'

env_client_py:
	clear
	# cd SMPyBandits ; time ./env_client.py --speed=50 --port=10000 --host=127.0.0.1 '{"arm_type": "Bernoulli", "params": [0,0,0,0,0,0,0.7,0.8,0.9]}'
	cd SMPyBandits ; time ./env_client.py --speed=20 --port=10000 --host=127.0.0.1 '{"arm_type": "Bernoulli", "params": [0.1,0.2,0.3,0.4,0.5,0.6,0.85,0.87,0.9]}'
	# A Bayesian problem: every repetition use a different mean vectors!
	# cd SMPyBandits ; time ./env_client.py dynamic --speed=50 --port=10000 --host=127.0.0.1 '{"arm_type": "Bernoulli", "params": {"function": "randomMeans", "args": {"nbArms": 9, "isSorted": false}}}'

env_client.exe:
	clear
	cd SMPyBandits ; g++ -Wall -Iinclude -o env_client.exe include/docopt.cpp env_client.cpp

env_client_cpp:	env_client.exe
	clear
	cd SMPyBandits ; time ./env_client.exe --speed=50 --port=10000 --host=127.0.0.1

# test_sub:
# 	clear
# 	g++ -std=c++11 -Iinclude -o test_sub.exe test_sub.cpp -pthread
# 	./test_sub.exe

# Cf. http://www.marinamele.com/7-tips-to-time-python-scripts-and-control-memory-and-cpu-usage
# Time profilers
profile: profile3
profile2:
	time nice -n 19 python2 -m cProfile -s cumtime ./SMPyBandits/main.py | tee ./logs/main_py2_profile_log.txt
profile3:
	time nice -n 19 python3 -m cProfile -s cumtime ./SMPyBandits/main.py | tee ./logs/main_py3_profile_log.txt

# Line memory profilers
memory_profiler:
	@echo "Running the script 'main.py' with the 'python -m memory_profiler' command line profiler ..."
	@echo "See 'https://pypi.python.org/pypi/memory_profiler' if needed"
	time nice -n 19 python3 -m memory_profiler ./SMPyBandits/main.py | tee ./logs/main_py3_memory_profiler_log.txt

# Line time profilers
line_profiler:	kernprof lprof
kernprof:
	@echo "Running the script 'main.py' with the 'kernprof' command line profiler ..."
	@echo "See 'https://github.com/rkern/line_profiler#kernprof' if needed"
	time nice -n 19 kernprof -l -v ./SMPyBandits/main.py | tee ./logs/main_py3_kernprof_log.txt
lprof:
	@echo "Time profile, line by line, for the script 'main.py' ..."
	@echo "See 'https://github.com/rkern/line_profiler#line-profiler' if needed"
	time nice -n 19 python3 -m line_profiler ./main.py.lprof | tee ./logs/main_py3_line_profiler_log.txt

# Python Call Graph, XXX does not work well as far as now
callgraph:
	@echo "Running the script 'main.py' with the pycallgraph command line profiler ..."
	@echo "See 'http://pycallgraph.slowchop.com/en/master/guide/command_line_usage.html#examples' if needed"
	# time nice -n 19 pycallgraph --verbose --max-depth 10 graphviz --output-file=logs/pycallgraph.svg -- ./SMPyBandits/main.py | tee ./logs/main_pycallgraph_log.txt
	# time nice -n 19 /usr/local/bin/pycallgraph --verbose --threaded --memory graphviz --output-file=logs/pycallgraph.svg -- ./SMPyBandits/main.py | tee ./logs/main_pycallgraph_log.txt
	time nice -n 19 pycallgraph --verbose --max-depth 10 gephi --output-file=logs/pycallgraph.gdf -- ./SMPyBandits/main.py | tee ./logs/main_pycallgraph_log.txt
	# -convert logs/pycallgraph.svg logs/pycallgraph.png

# Installers
install:
	sudo -H pip  install -U -r requirements.txt
install2:
	sudo -H pip2 install -U -r requirements.txt
install3:
	sudo -H pip3 install -U -r requirements.txt

# Senders:
send_ws3:	clean
	CP ~/SMPyBandits.git/ lilian_besson@ws3:~/SMPyBandits.git

# Cleaner
clean:
	-rm -vfr __pycache__/ */__pycache__/ */*/__pycache__/ */*/*/__pycache__/ */*/*/*/__pycache__/
	-rm -vf *.pyc */*.pyc */*/*.pyc */*/*/*.pyc */*/*/*/*.pyc */*/*/*/*.pyc

clean_build:
	-rm -vfr /tmp/SMPyBandits_build/
	-mkdir -p /tmp/SMPyBandits_build/
	-mv -vf build dist SMPyBandits.egg-info /tmp/SMPyBandits_build/

# Stats
stats:
	git-complete-stats.sh | tee complete-stats.txt
	git-cal --ascii | tee -a complete-stats.txt
	git wdiff complete-stats.txt

# Linters
# NPROC = `nproc`
# NPROC = 1
NPROC = `getconf _NPROCESSORS_ONLN`

tests:	alldoctest testsimulations
alldoctest:
	-./other_scripts/run_all_doctest.sh | tee ./logs/run_all_doctest.txt
testsimulations:
	-./other_scripts/run_test_simulations.sh | tee ./logs/run_test_simulations.txt

lint:
	-pylint -j $(NPROC) ./*.py ./*/*.py | tee ./logs/main_pylint_log.txt
lint3:
	-pylint --py3k -j $(NPROC) ./*.py ./*/*.py | tee ./logs/main_pylint3_log.txt

uml:	generate_uml uml2others

generate_uml:
	-mkdir uml_diagrams/
	# pyreverse -o dot -my -f ALL -p SMPyBandits ./SMPyBandits/*.py ./SMPyBandits/*/*.py ./SMPyBandits/*/*/*.py
	pyreverse -o dot -my -f ALL -p SMPyBandits ./SMPyBandits/*.py
	pyreverse -o dot -my -f ALL -p SMPyBandits.Arms ./SMPyBandits/Arms/*.py
	pyreverse -o dot -my -f ALL -p SMPyBandits.Environment ./SMPyBandits/Environment/*.py
	pyreverse -o dot -my -f ALL -p SMPyBandits.PoliciesMultiPlayers ./SMPyBandits/PoliciesMultiPlayers/*.py
	pyreverse -o dot -my -f ALL -p SMPyBandits.Policies ./SMPyBandits/Policies/*.py
	pyreverse -o dot -my -f ALL -p SMPyBandits.Policies.Experimentals ./SMPyBandits/Policies/Experimentals/*.py
	pyreverse -o dot -my -f ALL -p SMPyBandits.Policies.Posterior ./SMPyBandits/Policies/Posterior/*.py
	-mv -vf packages_SMPyBandits*.dot classes_SMPyBandits*.dot uml_diagrams/

uml2others:	uml2png	uml2svg	uml2pdf
uml2png:
	# Output packages and classes graphs to PNG...
	dot -Tpng uml_diagrams/packages_SMPyBandits.dot   > uml_diagrams/packages_SMPyBandits.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.dot    > uml_diagrams/classes_SMPyBandits.png
	dot -Tpng uml_diagrams/packages_SMPyBandits.Arms.dot   > uml_diagrams/packages_SMPyBandits.Arms.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.Arms.dot    > uml_diagrams/classes_SMPyBandits.Arms.png
	dot -Tpng uml_diagrams/packages_SMPyBandits.Environment.dot   > uml_diagrams/packages_SMPyBandits.Environment.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.Environment.dot    > uml_diagrams/classes_SMPyBandits.Environment.png
	dot -Tpng uml_diagrams/packages_SMPyBandits.PoliciesMultiPlayers.dot   > uml_diagrams/packages_SMPyBandits.PoliciesMultiPlayers.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.PoliciesMultiPlayers.dot    > uml_diagrams/classes_SMPyBandits.PoliciesMultiPlayers.png
	dot -Tpng uml_diagrams/packages_SMPyBandits.Policies.dot   > uml_diagrams/packages_SMPyBandits.Policies.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.Policies.dot    > uml_diagrams/classes_SMPyBandits.Policies.png
	dot -Tpng uml_diagrams/packages_SMPyBandits.Policies.Experimentals.dot   > uml_diagrams/packages_SMPyBandits.Policies.Experimentals.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.Policies.Experimentals.dot    > uml_diagrams/classes_SMPyBandits.Policies.Experimentals.png
	dot -Tpng uml_diagrams/packages_SMPyBandits.Policies.Posterior.dot   > uml_diagrams/packages_SMPyBandits.Policies.Posterior.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.Policies.Posterior.dot    > uml_diagrams/classes_SMPyBandits.Policies.Posterior.png
	-advpng -z -2 ./uml_diagrams/*.png

uml2svg:
	# Output packages and classes graphs to SVG...
	dot -Tsvg uml_diagrams/packages_SMPyBandits.dot   > uml_diagrams/packages_SMPyBandits.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.dot    > uml_diagrams/classes_SMPyBandits.svg
	dot -Tsvg uml_diagrams/packages_SMPyBandits.Arms.dot   > uml_diagrams/packages_SMPyBandits.Arms.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.Arms.dot    > uml_diagrams/classes_SMPyBandits.Arms.svg
	dot -Tsvg uml_diagrams/packages_SMPyBandits.Environment.dot   > uml_diagrams/packages_SMPyBandits.Environment.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.Environment.dot    > uml_diagrams/classes_SMPyBandits.Environment.svg
	dot -Tsvg uml_diagrams/packages_SMPyBandits.PoliciesMultiPlayers.dot   > uml_diagrams/packages_SMPyBandits.PoliciesMultiPlayers.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.PoliciesMultiPlayers.dot    > uml_diagrams/classes_SMPyBandits.PoliciesMultiPlayers.svg
	dot -Tsvg uml_diagrams/packages_SMPyBandits.Policies.dot   > uml_diagrams/packages_SMPyBandits.Policies.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.Policies.dot    > uml_diagrams/classes_SMPyBandits.Policies.svg
	dot -Tsvg uml_diagrams/packages_SMPyBandits.Policies.Experimentals.dot   > uml_diagrams/packages_SMPyBandits.Policies.Experimentals.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.Policies.Experimentals.dot    > uml_diagrams/classes_SMPyBandits.Policies.Experimentals.svg
	dot -Tsvg uml_diagrams/packages_SMPyBandits.Policies.Posterior.dot   > uml_diagrams/packages_SMPyBandits.Policies.Posterior.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.Policies.Posterior.dot    > uml_diagrams/classes_SMPyBandits.Policies.Posterior.svg

uml2pdf:
	# Output packages and classes graphs to PDF...
	dot -Tpdf uml_diagrams/packages_SMPyBandits.dot > uml_diagrams/packages_SMPyBandits.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.dot  > uml_diagrams/classes_SMPyBandits.pdf
	dot -Tpdf uml_diagrams/packages_SMPyBandits.Arms.dot   > uml_diagrams/packages_SMPyBandits.Arms.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.Arms.dot    > uml_diagrams/classes_SMPyBandits.Arms.pdf
	dot -Tpdf uml_diagrams/packages_SMPyBandits.Environment.dot   > uml_diagrams/packages_SMPyBandits.Environment.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.Environment.dot    > uml_diagrams/classes_SMPyBandits.Environment.pdf
	dot -Tpdf uml_diagrams/packages_SMPyBandits.PoliciesMultiPlayers.dot   > uml_diagrams/packages_SMPyBandits.PoliciesMultiPlayers.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.PoliciesMultiPlayers.dot    > uml_diagrams/classes_SMPyBandits.PoliciesMultiPlayers.pdf
	dot -Tpdf uml_diagrams/packages_SMPyBandits.Policies.dot   > uml_diagrams/packages_SMPyBandits.Policies.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.Policies.dot    > uml_diagrams/classes_SMPyBandits.Policies.pdf
	dot -Tpdf uml_diagrams/packages_SMPyBandits.Policies.Experimentals.dot   > uml_diagrams/packages_SMPyBandits.Policies.Experimentals.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.Policies.Experimentals.dot    > uml_diagrams/classes_SMPyBandits.Policies.Experimentals.pdf
	dot -Tpdf uml_diagrams/packages_SMPyBandits.Policies.Posterior.dot   > uml_diagrams/packages_SMPyBandits.Policies.Posterior.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.Policies.Posterior.dot    > uml_diagrams/classes_SMPyBandits.Policies.Posterior.pdf
	-PDFCompress -f ./uml_diagrams/*.pdf

ignorelogs:
	git checkout -- logs/


# -----------------------------------------
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON        = python3
SPHINXOPTS    =
# WARNING My fork contains the generic indexes, my PR to Sphinx was never accepted
# WARNING Use the system-wise 'sphinx-build' if you prefer
DEFAULTSPHINXBUILD   = sphinx-build
SPHINXBUILD   = sphinx-build
# SPHINXBUILD   = /home/lilian/publis/sphinx.git/sphinx-build.py
SPHINXPROJ    = SMPyBandits
SOURCEDIR     = .
BUILDDIR      = _build

clean-doc:
	mkdir --parents $(BUILDDIR)/html
	-rm -rfv /tmp/sphinx_html
	mv -vf $(BUILDDIR)/html /tmp/sphinx_html
	mkdir --parents $(BUILDDIR)/html/
	mv -vf /tmp/sphinx_html/.git $(BUILDDIR)/html/
	mv -vf /tmp/sphinx_html/.gitignore /tmp/sphinx_html/.nojekyll /tmp/sphinx_html/logo_large.png /tmp/sphinx_html/ISSUE_TEMPLATE.md /tmp/sphinx_html/LICENSE /tmp/sphinx_html/README.md $(BUILDDIR)/html/

.PHONY: help

send:	send_zamok send_gforge
#send_ws3
send_zamok:
	cd notebooks ; make send_zamok ; cd ..
	CP --exclude=.git "$(BUILDDIR)"/html/ ${Szam}phd/SMPyBandits/

send_gforge:
	CP --exclude=.git "$(BUILDDIR)"/html/ lbesson@scm.gforge.inria.fr:/home/groups/banditslilian/htdocs/

apidoc:
	-mkdir -vp /tmp/SMPyBandits/docs/
	-mv -vf docs/*.rst /tmp/SMPyBandits/docs/
	# @echo "==> Showing you which .rst files will be created in docs/"
	# sphinx-apidoc -n -o docs -e -M .
	# @echo "==> OK to generate these files ? [Enter for OK, Ctrl+C to cancel]"
	# @read
	# sphinx-apidoc -f -o docs -e -M SMPyBandits
	cd SMPyBandits ; mv -vf __init__.py __init__.py.old ; sphinx-apidoc -f -o ../docs -e -M . ; mv -vf __init__.py.old __init__.py  ; cd ..
	-mv -fv /tmp/SMPyBandits/docs/modules.rst ./docs/modules.rst

html:
	-mv -vf ./SMPyBandits/Policies/*.so /tmp/SMPyBandits/
	-mv -vf /tmp/SMPyBandits/*_cython*.so ./SMPyBandits/Policies/
	# XXX Adapt to using either my own sphinxbuild or the system-wide sphinxbuild script
	if [[ -x $(SPHINXBUILD) ]]; then $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); else $(DEFAULTSPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); fi
	# $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# \cp notebooks/*.html "$(BUILDDIR)"/html/notebooks/  # no need thanks to http://nbsphinx.readthedocs.io/
	-cp -vf ./logo_large.png $(BUILDDIR)/html/
	-rm -rvf $(BUILDDIR)/html/SMPyBandits/ $(BUILDDIR)/html/_modules/SMPyBandits/
	#-rm -rvf $(BUILDDIR)/html/_sources/SMPyBandits/
	-./notebooks/symlinks.sh
	-./other_scripts/fixes_html_in_doc.sh
	\cp uml_diagrams/*.svg "$(BUILDDIR)"/html/uml_diagrams/
	\cp logs/*.txt "$(BUILDDIR)"/html/logs/
	-mv -vf /tmp/SMPyBandits/*.so ./SMPyBandits/Policies/
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."


changes:
	-mkdir -p "$(BUILDDIR)"/changes
	$(SPHINXBUILD) -b changes "$(SOURCEDIR)" $(SPHINXOPTS) "$(BUILDDIR)"/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	-mkdir -p "$(BUILDDIR)"/linkcheck
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" $(SPHINXOPTS) "$(BUILDDIR)"/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output or in $(BUILDDIR)/linkcheck/output.txt."

old_doctest:
	-mkdir -p "$(BUILDDIR)"/doctest
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" $(SPHINXOPTS) "$(BUILDDIR)"/doctest
	@echo "Testing of doctests in the sources finished, look at the results in $(BUILDDIR)/doctest/output.txt."

coverage:
	-mkdir -p "$(BUILDDIR)"/coverage
	$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" $(SPHINXOPTS) "$(BUILDDIR)"/coverage
	@echo "Testing of coverage in the sources finished, look at the results in $(BUILDDIR)/coverage/python.txt."

