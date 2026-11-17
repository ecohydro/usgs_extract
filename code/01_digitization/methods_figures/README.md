# Methods figures M1 and M2

Figure M1 is the pipeline flowchart; Figure M2 follows one table (report 1264, PDF page 31) through
detection, digitization and metadata extraction. Both are slides of one Claude Slides deck, so they
can be edited by hand in the browser: https://claude.ai/artifact/JTxVwHCuvqotvq3jkrPf4x (private
until shared from its Share menu). `project/` is a copy of the deck's files. Edits made in the
browser are not copied here automatically: re-read the deck before changing these files, and
publish changes made here back to it.

## Run order

All scripts run with `/home/annaboser/optimal-sampling/.venv/bin/python` from the repo root.

| Step | Script | Makes |
|---|---|---|
| 1 | `render_detection_example.py` | `assets/detection_1264_p31.png`: the page with the detector's saved boxes and scores (M2 a) |
| 2 | `render_table_examples.py` | `assets/tables/peaks_1264_p31_digitized.png`: Reducto's HTML for table 1, unedited (M2 b); `_scan.png` is the matching crop of the page, not shown |
| 3 | `render_metadata_example.py` | `assets/metadata_1264_p31_t1.png`: the full metadata run's answer for table 1 (M2 c) |
| 4 | `render_slides.py` | `manuscript/figures/methods_overview.png/.pdf` and `methods_example.png/.pdf` |

Steps 1 to 3 only need re-running if the example page changes; the three images are also uploaded
to the deck. Step 4 re-draws both figures from `project/slides/*.html`.

## Where each number on Figure M1 comes from

| On the figure | Source |
|---|---|
| 1,042 reports, published 1879-1980 | `data/digitization_intermediates/01_download_and_preprocess/publication_list.csv` |
| 629 reports, 104,729 pages; 60% | unique reports and pages in the two detector score files on the cluster (`csvs/cali_1970_results.csv`, `cali_1980_results.csv`); 629 / 1,042 |
| 85% of tables found | detection recall at threshold 0.8: `code/01_digitization/02_table_detection/evaluation/README.md` |
| 74,492 pages with a detected table | pages sent to digitization: `code/01_digitization/02_table_detection/evaluation/README.md` |
| 96% of values correct | Reducto raw output, right and findable: `code/01_digitization/03_digitization/evaluation/README.md` |
| 106,191 tables | table CSVs, `data/analysis/table_headers.csv` |
| 91% of fields correct | mean of eight fields on the metadata test set: `code/01_digitization/04_metadata_extraction/README.md` |

The same three accuracies appear on Figure M2. Each slide's speaker notes repeat these sources.
