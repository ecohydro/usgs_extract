# Evaluation of table digitization (draft)

How well do the digitized tables reproduce the printed ones? Twenty pages were typed by hand from
the scans and three outputs are scored against them: raw PaddleOCR, raw Reducto, and the published
CSV. A printed number counts as right only if the number, its row label and its column header all
match ("right and findable").

Run in order, from this folder, with `~/miniconda3/envs/geofm-vlm/bin/python`.
Outputs go to `data/analysis/evaluation/digitization/`.

| Script | What it does |
|---|---|
| `01_draw_sample.py` | Draws the test pages, four per kind of table, one per report. |
| `02_score.py` | Scores every typed page against the three outputs, lenient and strict. |
| `03_review_pages.py`, `review_server.py` | Pages for reading the typed tables against the scans and correcting them in place. |
| `04_verdict_check.py` | Draws scorer verdicts for a check by hand. |
| `05_summarize.py` | Makes every figure reported in the paper. |
| `scoring.py`, `loaders.py`, `crop.py` | The scorer, the readers of each output, and the zoom tool used when typing. |

Settings are in `defaults.py`, the typing rules in `TRUTH_FORMAT.md`, decisions and known limits
in `NOTES.md`.

## Where each number in the paper comes from

Rows of the summary files are for `variant = default` unless stated.

| Number in the paper | Script | Output file |
|---|---|---|
| 20 pages from 57 reports | `01_draw_sample.py` | `sample_pages.csv`; `sample_frame_counts.csv`, row "that also have PaddleOCR output" |
| 32 printed tables | `02_score.py` | `scores/tables.csv`, rows of one output |
| 4,696 printed numbers | `02_score.py` | `scores/cells.csv`, rows of one output; `summary/by_page.csv`, sum of `n_numbers` |
| Table M2: right and findable, with range | `05_summarize.py` | `summary/overall.csv`: `right_findable`, `_lo`, `_hi` |
| Table M2: the six other outcomes | `05_summarize.py` | `summary/overall.csv`: `row_label_wrong` to `missed_table` |
| Table M2: tables found | `05_summarize.py` | `summary/tables.csv`: `found` |
| Strict figure | `05_summarize.py` | `summary/overall.csv`, `variant = _strict` |
| Share of pages by kind of table | `05_summarize.py` | `summary/kind_weights.csv`: `pages` |
| Supplementary Table 1 (by kind) | `05_summarize.py` | `summary/by_kind_pooled.csv` |
| Output numbers that match nothing printed | `05_summarize.py` | `summary/extras.csv`: `share_not_printed` |
| Check of the scorer, 98 of 100 | `04_verdict_check.py` | `scores/verdict_check_graded.csv`: `checker_verdict` |
