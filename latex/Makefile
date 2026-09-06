# Prevođenje rada:  make        -> main.pdf
#                   make clean  -> briše pomoćne datoteke
#                   make watch  -> automatsko prevođenje pri svakoj izmjeni

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preamble.tex references.bib $(wildcard chapters/*.tex) $(wildcard frontmatter/*.tex)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk

.PHONY: all watch clean
