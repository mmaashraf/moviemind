.PHONY: pdf docx clean figures

MAIN = moviemind_report
TEXDIR = $(CURDIR)
ROOT = $(TEXDIR)/../..

figures:
	cd "$(ROOT)" && python3 docs/report/generate_figures.py

pdf: figures
	cd "$(TEXDIR)" && tectonic "$(MAIN).tex"

# Best-effort Word export (pandoc does not expand \\input; merges section bodies)
docx: $(MAIN).tex sections/*.tex
	cd "$(TEXDIR)" && pandoc \
		--from=latex \
		--to=docx \
		-o "$(MAIN).docx" \
		sections/abstract.tex \
		sections/introduction.tex \
		sections/literature.tex \
		sections/data_exploration.tex \
		sections/methodology.tex \
		sections/evaluation.tex \
		sections/conclusion.tex \
		sections/references.tex
	@echo "Wrote $(MAIN).docx (check title page and table of contents manually)"

clean:
	rm -f "$(MAIN).pdf" "$(MAIN).docx" "$(MAIN).aux" "$(MAIN).bbl" "$(MAIN).blg" \
		"$(MAIN).log" "$(MAIN).out" "$(MAIN).toc"
