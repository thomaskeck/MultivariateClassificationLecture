all: MultivariateClassificationLecture.pdf

print: MultivariateClassificationLecturePrint.pdf
	pdftk MultivariateClassificationLecture.pdf cat 1-4 7-17 22-23 25 27-28 31 33-34 36-40 42-127 129-end output MultivariateClassificationLecturePrint.pdf

%.pdf: %.tex header.tex beamerthemeCSC.sty
	pdflatex -shell-escape $<
	pdflatex -shell-escape $<

clean:
	rm -f *.aux *.log *.nav *.out *.pyg *.snm *.vrb *.toc *~
	rm -rf _minted-*

clobber: clean
	rm -f *.pdf
