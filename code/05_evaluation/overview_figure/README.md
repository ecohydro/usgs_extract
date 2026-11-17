# Methods figures (Claude Slides deck)

The methods figures are slides of one Claude Slides deck ("Methods figures": overview flowchart, and one
table through the pipeline: detected, digitized, metadata), so it can be edited by hand in the
browser: https://claude.ai/artifact/JTxVwHCuvqotvq3jkrPf4x (private until shared from its Share menu).
Export it as PDF or PNG from the deck's Share menu into `manuscript/figures/methods_overview.pdf`.

This folder mirrors the deck so it is not lost: `project/` holds the deck's files as last published
from here (`deck.json`, `deck.css`, `slides/overview.html`), and `assets/` the two page images
uploaded to it. Edits made in the browser are not copied here automatically; re-read the deck before
changing these files.

| Asset in the deck | Source | sha256 |
|---|---|---|
| `/_blob/629b0c645b981d7b6d6cd5c2a4dc82e9` | `assets/report_cover_553_p1.png` (Water-Supply Paper 961, report 553, PDF page 1, grayscale, 800 px wide) | 4e24f295... |
| `/_blob/8fcfba7036bc99b01910159809744518` | `assets/page_with_table_553_p267.png` (same report, PDF page 267 = printed page 258; no longer shown) | c49cb8d2... |
| `/_blob/2475e258aeebd327fbba37b6896fa13a` | `assets/page_with_table_553_p267_crop.png` (the same page, bottom cropped to the rendered page's shape) | |
| `/_blob/113597b398dbb518ace2bc7eb3977010` | `assets/digitized_553_p267.png` (Reducto's output for that page, typeset by `render_digitized.py`) | |

Scripts that make the images: `render_detection_example.py` (detector boxes on report 1264 p31),
`render_table_examples.py` (table crops at Reducto's boxes and Reducto's output typeset; output in
`assets/tables/`, five extra candidates included), `render_digitized.py` (no longer shown).

`render_metadata_example.py` (the metadata the full run returned for report 1264 p31 table 1).

The separate detection and digitization slides were removed from the deck in the browser (Nov 16,
2026); their last files are kept in `removed_slides/`.

The source of every number and image is in each slide's speaker notes.
