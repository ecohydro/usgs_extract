# Methods and supplement in LaTeX

The methods text is edited in a Claude Doc (Anna's):
https://claude.ai/artifact/B5yfjMgLnP5rrP8Sn8xk1S. The files here are a LaTeX copy of it.

| File | What it is |
|---|---|
| `methods_draft.md` | The doc exported as Markdown (2026-11-17, doc revision 1518; metadata sentence updated 2026-11-20, revision 1521) |
| `md_to_tex.py` | Turns `methods_draft.md` into the two `.tex` files; wording and numbers are copied as they are |
| `methods.tex` | Main-text methods: Overview, Table detection, Digitization, Metadata extraction; Figures M1 and M2, Tables M1 to M3 |
| `supplement.tex` | Supplementary Notes 1 to 4 and Supplementary Tables 1 and 2 |
| `main.tex` | A small wrapper that compiles both on their own |
| `main.pdf` | The compiled result |

To update after the doc changes: export the doc as Markdown (the doc's Export menu) over
`methods_draft.md`, then run `python manuscript/methods/md_to_tex.py`. Do not edit the two `.tex`
files by hand; the next run overwrites them. `main.tex` is written once and can be edited.

The figures are read from `../figures/methods_overview.pdf` and `methods_example.pdf`. Figures and
tables keep the labels used in the text (Figure M1, Table M2, Supplementary Table 1), written into
unnumbered captions, so they can be renumbered when the paper's figure order is fixed. Citations
are still plain text ("Smock et al. 2022").

To compile: `conda run -n latex tectonic main.tex` (Tectonic, installed in the conda environment
`latex` on 2026-11-17; it downloads the LaTeX packages it needs on first use), which writes
`main.pdf` (13 pages). Plain `pdflatex main.tex` also works where TeX Live is installed (packages:
amsmath, graphicx, booktabs, tabularx, array, caption, listings).
