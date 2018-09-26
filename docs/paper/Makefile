# Quick Makefile to:
# - compile a markdown article to PDF (with pandoc)
# - compile a LaTeX article to PDF (with latexmk)

# __author__ = "Lilian Besson"
# __version__ = "0.9"

# Using bash and not sh, cf. http://stackoverflow.com/a/589300/
SHELL := /bin/bash -o pipefail


# --------------------------------------------------------
# LATEX=lualatex
# LATEX=xelatex -shell-escape -output-driver="xdvipdfmx -z 0"
# LATEX=xelatex
LATEX=pdflatex

#  --standalone
#  --natbib
PANDOCOPTIONS=--verbose --filter pandoc-citeproc --template=.paper_template.tex --number-sections --bibliography paper.bib
MARKDOWNOPTIONS=--from=markdown+backtick_code_blocks+implicit_figures+pipe_tables+citations+footnotes+smart

# Generate the paper for http://joss.theoj.org/about#author_guidelines
longpaper: longpaper.tex longpaper.md
	-latexmk -c
	-rm -vf longpaper.bbl longpaper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk
	latexmk -f -gg -pdf -xelatex longpaper.tex
	-latexmk -c
	-rm -vf longpaper.bbl longpaper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk

paper:
	-latexmk -c
	-rm -vf paper.bbl paper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk
	# latexmk -f -gg -pdf -xelatex paper.tex
	latexmk -f -gg -pdf paper.tex
	-latexmk -c
	-rm -vf paper.bbl paper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk

old_paper: paper.tex paper.md
	-latexmk -c
	-rm -vf paper.bbl paper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk
	latexmk -f -gg -pdf -xelatex paper.tex
	-latexmk -c
	-rm -vf paper.bbl paper.synctex.gz .paper_template.aux .paper_template.fls .paper_template.log .paper_template.fdb_latexmk

clean:
	latexmk -c

longpaper.tex: longpaper.md
	pandoc $(PANDOCOPTIONS) --toc $(MARKDOWNOPTIONS) $< -o $@
%.tex: %.md
	pandoc $(PANDOCOPTIONS) $(MARKDOWNOPTIONS) $< -o $@
%.pdf: %.md
	pandoc $(PANDOCOPTIONS) $(MARKDOWNOPTIONS) $< -o $@