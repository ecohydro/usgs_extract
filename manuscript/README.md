# Manuscript figures and tables

One line per figure and table: what it shows, the code that makes it, and the output file.
Results figures are Annette's (`code/02_inventory/`, `code/04_vignettes/`); methods figures and
tables are Anna's (`code/05_evaluation/`). Figures are saved in `figures/` (PNG + PDF), tables in
`tables/` (one CSV per table, written by a script). Methods outputs are named `methods_*`
(supplement: `supp_methods_*`) until the figure order is fixed. Old figures are in `figures/archived/`.

## Figures

| Figure | Shows | Code | Output |
|---|---|---|---|
| 1 | Measurements by water type and decade (California) | `code/02_inventory/01_measurements_by_type.ipynb` | `figures/fig1_measurements_by_type_decade_values.png` |
| 2 | Maps by water category | `code/02_inventory/fig2_category_maps.py` | `figures/fig2_category_maps.png/.pdf` |
| 3 | Restored sites compared with NWIS | `code/02_inventory/fig3_nwis_comparison.py` | `figures/fig3_nwis_comparison.png/.pdf` |
| 4 | Santa Ynez example | `code/04_vignettes/santa_ynez/fig_santa_ynez.py` | `figures/fig4_santa_ynez.png/.pdf` |
| M1 | Methods overview flowchart | `code/05_evaluation/overview_figure/render_slides.py` (slide `project/slides/overview.html`, mirror of the Claude Slides deck) | `figures/methods_overview.png/.pdf` |
| M2 | One table through the pipeline (report 1264 p31: detected, digitized, metadata) | same script (slide `example.html`); panel images from `render_detection_example.py`, `render_table_examples.py`, `render_metadata_example.py` | `figures/methods_example.png/.pdf` |

## Supplement

| Shows | Code | Output |
|---|---|---|
| Figure 1, table rows instead of values; all locations | `code/02_inventory/01_measurements_by_type.ipynb` | `figures/fig1_*_rows.png`, `figures/supp_all_locations_*` |
| Figure 3 at 0.5 km and 2 km site tolerance | `code/02_inventory/fig3_nwis_comparison.py` | `figures/supp_fig3_nwis_comparison_tol*.png/.pdf` |
| Santa Ynez, Gibraltar annual flows | `code/04_vignettes/santa_ynez/gibraltar_plot.py` | `figures/santa_ynez_gibraltar_annual.png/.pdf` |

## Tables

| Table | Shows | Code | Output |
|---|---|---|---|
| (none yet) | | | |
