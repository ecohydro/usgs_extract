# Table detection evaluation

How well the table detector (`../01_detect.py`, Microsoft Table Transformer) finds table pages, and how often
it flags a page with no table.

- **Recall** is measured on a hand-compiled list of pages with groundwater tables in 139 USGS reports. The
  list gives printed page numbers, so each page is first located in its PDF. Every listed page was then looked
  at, blind to the detector score: pages with no table were moved to the adjacent page when the table is there
  (fold-out tables "facing page N") and removed otherwise. A listed page counts as found when any detection on
  it scores at or above the threshold.
- **False alarms** are measured on the pages the detector flagged in the California reports. Reducto returned
  a table for some and none for others; a blind look at a sample of each group corrects that split.

Outputs are in `data/analysis/evaluation/detection/` (`OUT` below). Settings are in `defaults.py`, decisions
and limits in `NOTES.md`. Python: `~/.venvs/usgs-eval/bin/python`.

## Run order

| Script | What it does |
|---|---|
| `01_page_map.py` | printed page numbers of the list to PDF pages (with the hand-entered `page_map_overrides.csv`) |
| `02_page_number_check.py` | strips of the uncertain pages, read by eye to produce the overrides; then 01 again |
| `03_run_detection.py` | runs the detector on the listed pages and their neighbours (GPU, about 40 minutes) |
| `04_list_check.py` | `draw`, `neighbours`, `finish`: the blind look at every listed page; final page map and labels |
| `05_score_recall.py` | recall by threshold, with a range from resampling reports |
| `06_plot_recall.py` | figure of recall against threshold |
| `07_false_alarm_sample.py` | flagged pages of the corpus, what Reducto returned, the samples to label |
| `08_false_alarm_estimate.py` | corrected false-alarm rate |
| `09_anna_check.py` | the web page on which Anna verified page numbers, borderline pages and 40 calls |

`images.py` holds the shared helpers (page images, contact sheets). Labels are entered by hand in the
`*_labeled.csv` files; the scripts never overwrite them.

## Where each number comes from

| Number | Script | Output file (row, column) |
|---|---|---|
| 3,272 pages in 139 reports | `05_score_recall.py` | `recall_curve.csv` (`cleaned`; `pages`, `reports`) |
| 73 listed pages removed | `04_list_check.py finish` | `listed_page_labels.csv` (`label` = `no_table`) |
| 27 listed pages moved to the adjacent page | `04_list_check.py finish` | `page_map_final.csv` (`method` = `moved_to_adjacent_table`) |
| Recall 87.5% (84.0 to 90.7) at 0.75, 2,863 found | `05_score_recall.py` | `recall_curve.csv` (`cleaned`, 0.75; `recall`, `low`, `high`, `found`) |
| Recall 85.3% (81.5 to 88.5) at 0.8, 2,791 found | `05_score_recall.py` | `recall_curve.csv` (`cleaned`, 0.8) |
| False alarms 11.9% (10.3 to 13.9) at 0.75 | `08_false_alarm_estimate.py` | `false_alarm_estimate.csv` (0.75, `anna_ruling`; `false_alarm_rate`, `low`, `high`) |
| False alarms 11.0% (9.3 to 13.0) at 0.8 | `08_false_alarm_estimate.py` | `false_alarm_estimate.csv` (0.8, `anna_ruling`) |
| Samples of 60 and 300; 57 and 9 with no table | `08_false_alarm_estimate.py` | `false_alarm_estimate.csv` (0.75, `anna_ruling`; `looked_no_table_group`, `really_no_table_there`, `looked_table_group`, `really_no_table_there_2`) |
| 9.7% of flagged pages with no Reducto table | `08_false_alarm_estimate.py` | `false_alarm_estimate.csv` (0.75; `reducto_no_table_share`) |
| 74,492 of 104,729 pages flagged | `07_false_alarm_sample.py` | `corpus_counts.csv` (`both`; `pages_sent_to_reducto`, `pages_scored`) |
| 270 reports at 0.75, 359 at about 0.8 | `07_false_alarm_sample.py` | `corpus_counts.csv` (`batch_two`, `batch_one`; `reports_scored`) |
| Anna verified 30 page numbers, 21 borderline pages, 40 calls | `09_anna_check.py` | `for_anna/check.html` (not in git; rebuilt by the script) |

The samples of 60 and 300 are the sizes at 0.75. At 0.8 only the sampled pages scoring 0.8 or higher count:
54 and 296 (51 and 9 with no table).
