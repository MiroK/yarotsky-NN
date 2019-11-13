.PHONY: all clean

all: pdf view

pdf: notes.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

view: 
	evince notes.pdf &

clean:
	latexmk -C
