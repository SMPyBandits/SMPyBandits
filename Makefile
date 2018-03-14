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

single:
	make clean ; clear ; make singleplayer3
multi:
	make clean ; clear ; make multiplayers3
moremulti:
	make clean ; clear ; make moremultiplayers3
sparsemulti:
	make clean ; clear ; make sparsemultiplayers3

alllint:	lint lint3 pyreverse stats doc
doc:
	make html clean send

# Runners
singleplayer:	singleplayer3
singleplayer3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py | tee ./logs/main_py3_log.txt
singleplayer2:
	time nice -n 19 python2 ./SMPyBandits/main.py | tee ./logs/main_py2_log.txt

comparing_aggregation_algorithms:	comparing_aggregation_algorithms3
comparing_aggregation_algorithms3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_comparing_aggregation_algorithms | tee ./logs/main_py3_log.txt
comparing_aggregation_algorithms2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_comparing_aggregation_algorithms | tee ./logs/main_py2_log.txt

comparing_doubling_algorithms:	comparing_doubling_algorithms3
comparing_doubling_algorithms3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_comparing_doubling_algorithms | tee ./logs/main_py3_log.txt
comparing_doubling_algorithms2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_comparing_doubling_algorithms | tee ./logs/main_py2_log.txt

sparse:	sparse3
sparse3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_sparse | tee ./logs/main_py3_log.txt
sparse2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_sparse | tee ./logs/main_py2_log.txt

markovian:	markovian3
markovian3:
	time nice -n 19 ipython3 ./SMPyBandits/main.py configuration_markovian | tee ./logs/main_py3_log.txt
markovian2:
	time nice -n 19 python2 ./SMPyBandits/main.py configuration_markovian | tee ./logs/main_py2_log.txt

multiplayers:	multiplayers3
multiplayers3:
	time nice -n 19 ipython3 ./SMPyBandits/main_multiplayers.py | tee ./logs/main_multiplayers_py3_log.txt
multiplayers2:
	time nice -n 19 python2 ./SMPyBandits/main_multiplayers.py | tee ./logs/main_multiplayers_py2_log.txt

moremultiplayers: moremultiplayers3
moremultiplayers3:
	time nice -n 19 ipython3 ./SMPyBandits/main_multiplayers_more.py | tee ./logs/main_multiplayers_more_py3_log.txt
moremultiplayers2:
	time nice -n 19 python2 ./SMPyBandits/main_multiplayers_more.py | tee ./logs/main_multiplayers_more_py2_log.txt

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
# LATEX=lualatex
# LATEX=xelatex -shell-escape -output-driver="xdvipdfmx -z 0"
# LATEX=xelatex
LATEX=pdflatex

PANDOC=pandoc --verbose --filter pandoc-citeproc --template=.paper_template.tex --number-sections --standalone --toc --natbib --bibliography paper.bib
MARKDOWNOPTIONS=--from=markdown+backtick_code_blocks+implicit_figures+pipe_tables+citations+footnotes+smart

# Generate the paper for http://joss.theoj.org/about#author_guidelines
longpaper: longpaper.tex longpaper.md
	latexmk -f -gg -pdf longpaper.tex
	latexmk -c
	rm -vf longpaper.bbl longpaper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk

paper: paper.tex paper.md
	latexmk -f -gg -pdf paper.tex
	latexmk -c
	rm -vf paper.bbl paper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk

%.tex: %.md
	$(PANDOC) $(MARKDOWNOPTIONS) $< -o $@
%.pdf: %.md
	$(PANDOC) $(MARKDOWNOPTIONS) $< -o $@


# --------------------------------------------------------

policy_server_py:
	clear
	time ./policy_server.py --port=10000 --host=127.0.0.1 '{"nbArms": 9, "archtype": "UCBalpha", "params": { "alpha": 1.0 }}'

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
	time nice -n 19 python2 -m cProfile -s cumtime ./SMPyBandits/main.py | tee ./logs/main_py2_profile_log.txt
profile3:
	time nice -n 19 python3 -m cProfile -s cumtime ./SMPyBandits/main.py | tee ./logs/main_py3_profile_log.txt

# Line time profilers
line_profiler:	kernprof lprof
kernprof:
	@echo "Running the script 'main.py' with the 'kernprof' command line profiler ..."
	@echo "See 'https://github.com/rkern/line_profiler#kernprof' if needed"
	time nice -n 19 kernprof -l ./SMPyBandits/main.py | tee ./logs/main_py3_log.txt
lprof:
	@echo "Time profile, line by line, for the script 'main.py' ..."
	@echo "See 'https://github.com/rkern/line_profiler#line-profiler' if needed"
	time nice -n 19 python3 -m line_profiler ./SMPyBandits/main.py.lprof | tee ./logs/main_py3_line_profiler_log.txt

# Python Call Graph, XXX does not work well as far as now
callgraph:
	@echo "Running the script 'main.py' with the pycallgraph command line profiler ..."
	@echo "See 'http://pycallgraph.slowchop.com/en/master/guide/command_line_usage.html#examples' if needed"
	# time nice -n 19 pycallgraph --verbose --max-depth 10 graphviz --output-file=logs/pycallgraph.svg -- ./SMPyBandits/main.py | tee ./logs/main_pycallgraph_log.txt
	# time nice -n 19 /usr/local/bin/pycallgraph --verbose --threaded --memory graphviz --output-file=logs/pycallgraph.svg -- ./SMPyBandits/main.py | tee ./logs/main_pycallgraph_log.txt
	time nice -n 19 pycallgraph --verbose --max-depth 10 gephi --output-file=logs/pycallgraph.gdf -- ./SMPyBandits/main.py | tee ./logs/main_pycallgraph_log.txt
	# -convert logs/pycallgraph.svg logs/pycallgraph.png

# Installers
# FIXME make a virtualenv automatically?
install:
	sudo -H pip  install -U -r requirements.txt
install2:
	sudo -H pip2 install -U -r requirements.txt
install3:
	sudo -H pip3 install -U -r requirements.txt

# Senders:
send_ws3:	clean
	CP ~/AlgoBandits.git/ lilian_besson@ws3:~/These/src/AlgoBandits.git/

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

tests:
	./run_all_doctest.sh
	./run_test_simulations.sh

lint:
	-pylint -j $(NPROC) ./*.py ./*/*.py | tee ./logs/main_pylint_log.txt
lint3:
	-pylint --py3k -j $(NPROC) ./*.py ./*/*.py | tee ./logs/main_pylint3_log.txt

pyreverse:
	-mkdir uml_diagrams/
	pyreverse -o dot -my -f ALL -p SMPyBandits ./*.py ./*/*.py
	-mv -vf packages_SMPyBandits.dot classes_SMPyBandits.dot uml_diagrams/
	# Output packages and classes graphs to PNG...
	dot -Tpng uml_diagrams/packages_SMPyBandits.dot   > uml_diagrams/packages_SMPyBandits.png
	dot -Tpng uml_diagrams/classes_SMPyBandits.dot    > uml_diagrams/classes_SMPyBandits.png
	-advpng -z -2 ./uml_diagrams/*.png
	# Output packages and classes graphs to SVG...
	dot -Tsvg uml_diagrams/packages_SMPyBandits.dot   > uml_diagrams/packages_SMPyBandits.svg
	dot -Tsvg uml_diagrams/classes_SMPyBandits.dot    > uml_diagrams/classes_SMPyBandits.svg
	# Output packages and classes graphs to PDF...
	dot -Tpdf uml_diagrams/packages_SMPyBandits.dot > uml_diagrams/packages_SMPyBandits.pdf
	dot -Tpdf uml_diagrams/classes_SMPyBandits.dot  > uml_diagrams/classes_SMPyBandits.pdf
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
# SPHINXBUILD   = sphinx-build
SPHINXBUILD   = /home/lilian/publis/sphinx.git/sphinx-build.py
SPHINXPROJ    = SMPyBandits
SOURCEDIR     = .
BUILDDIR      = _build

clean-doc:
	mkdir --parents $(BUILDDIR)/html
	-rm -rfv /tmp/sphinx_html
	mv -vf $(BUILDDIR)/html /tmp/sphinx_html
	mkdir --parents $(BUILDDIR)/html/
	mv -vf /tmp/sphinx_html/.git $(BUILDDIR)/html/
	mv -vf /tmp/sphinx_html/.gitignore .nojekyll logo_large.png ISSUE_TEMPLATE.md LICENSE README.md $(BUILDDIR)/html/

.PHONY: help

send:	send_zamok send_gforge
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
	# XXX Adapt to using either my own sphinxbuild or the system-wide sphinxbuild script
	if [[ -x $(SPHINXBUILD) ]]; then $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); else $(DEFAULTSPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); fi
	# $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# \cp notebooks/*.html "$(BUILDDIR)"/html/notebooks/  # no need thanks to http://nbsphinx.readthedocs.io/
	-cp -vf ./logo_large.png $(BUILDDIR)/html/
	-rm -rvf $(BUILDDIR)/html/SMPyBandits/ $(BUILDDIR)/html/_modules/SMPyBandits/
	#-rm -rvf $(BUILDDIR)/html/_sources/SMPyBandits/
	-./notebooks/symlinks.sh
	-./.fixes_html_in_doc.sh
	\cp uml_diagrams/*.svg "$(BUILDDIR)"/html/uml_diagrams/
	\cp logs/*.txt "$(BUILDDIR)"/html/logs/
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

