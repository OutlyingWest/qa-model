# Recompile Report

Use the following command to rebuild the LaTeX report PDF from the repository root (or run it from anywhere):

```
cd report && latexmk -pdf -interaction=nonstopmode main.tex
```

Notes
- Output file: `report/main.pdf`.
- Live preview (auto-rebuild on changes): `cd report && latexmk -pdf -pvc main.tex`
- Clean build artifacts: `cd report && latexmk -C`
- Requirements: a LaTeX toolchain with `latexmk` and `pdflatex` (already installed on this machine).

If you ask me to “recompile the report”, I will execute the command above to generate an updated PDF.

