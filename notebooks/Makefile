# Makefile to convert Jupyter notebooks to HTML pages
SHELL=/usr/bin/env /bin/bash

all:	html_notebooks send

send:	send_zamok
send_zamok:
	CP *.html ${Szam}phd/notebooks/

CONTENT_IPYNB=$(wildcard *.ipynb)
CONTENT_HTML=$(CONTENT_IPYNB:.ipynb=.html)

html_notebooks:	$(CONTENT_HTML)

%.html:	%.ipynb
	jupyter-nbconvert $< --to html

venv:
	virtualenv3 ./venv3/
	. ./venv3/bin/activate ; pip install jupyter tqdm numba
	. ./venv3/bin/activate ; pip install git+https://github.com/SMPyBandits/SMPyBandits
