# Makefile to convert Jupyter notebooks to HTML pages
SHELL=/usr/bin/env /bin/bash

all:	nb2py nb2html send

send:	send_zamok
send_zamok:
	CP *.html ${Szam}phd/notebooks/

CONTENT_IPYNB=$(wildcard *.ipynb)
CONTENT_HTML=$(CONTENT_IPYNB:.ipynb=.html)
CONTENT_PY=$(CONTENT_IPYNB:.ipynb=.py)

nb2py:	$(CONTENT_PY)
%.py:	%.ipynb
	jupyter-nbconvert $< --to python

nb2html:	$(CONTENT_HTML)
%.html:	%.ipynb
	jupyter-nbconvert $< --to html

venv:
	virtualenv3 ./venv3/
	. ./venv3/bin/activate ; pip install jupyter tqdm numba
	. ./venv3/bin/activate ; pip install git+https://github.com/SMPyBandits/SMPyBandits
