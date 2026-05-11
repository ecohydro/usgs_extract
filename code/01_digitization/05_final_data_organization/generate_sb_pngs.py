#!/usr/bin/env python3
"""
generate_sb_pngs.py

Converts SB County PDF pages to grayscale PNGs at 300 DPI and places them
in the existing per-page folders under data/digitized/{pub_id}/page_{N}/.

Only generates PNGs for pages that already have a page folder (i.e., pages
with digitized table data). Skips pages where a PNG already exists.

PDF -> Publication ID mapping (sb1 excluded — already in main pipeline as 23995):
    sb.pdf  -> 52119    sb2.pdf -> 23894    sb3.pdf -> 23879
    sb4.pdf -> 52056    sb5.pdf -> 23993    sb6.pdf -> 23996
    sb7.pdf -> 23997

Note: sb5.pdf in the scanned SB folder is mislabeled — it is the same 68-page
1962 document as report.pdf (pub 23997 / ofr63103). The override below points
sb5.pdf -> report1963.pdf (42 pages, pub 23993 / ofr64117, 1963), the correct source.
"""

import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

REPO_ROOT   = Path(__file__).resolve().parents[3]
PDF_DIR     = (REPO_ROOT / "Data_Files" / "sb_data_updates"
               / "zipped_SB" / "Scanned_SB" / "Scanned_SB_pdfs")
# Override: use the USGS download for ofr63103 (pub 23997) — the original
# sb7.pdf in the repo is a 42-page incomplete scan; Reducto processed a 68-page
# version matching report.pdf (downloaded from pubs.usgs.gov/publication/ofr63103).
PDF_OVERRIDES = {
    "sb5.pdf": (REPO_ROOT / "Data_Files" / "report1963.pdf", 0),
    "sb7.pdf": (REPO_ROOT / "Data_Files" / "report.pdf", 0),
}
DIGITIZED   = REPO_ROOT / "data" / "digitized"
DPI         = 300

# All offsets are 0: Reducto used PDF physical page numbers (1-indexed) directly,
# and each PDF in PDF_DIR matches the exact page count Reducto processed.
PDF_TO_PUBID = {
    "sb.pdf":  ("52119", 0),
    "sb2.pdf": ("23894", 0),
    "sb3.pdf": ("23879", 0),
    "sb4.pdf": ("52056", 0),
    "sb5.pdf": ("23993", 0),
    "sb6.pdf": ("23996", 0),
    "sb7.pdf": ("23997", 0),
}


def main() -> None:
    stats = {"generated": 0, "skipped_exists": 0, "skipped_no_folder": 0, "errors": 0}

    for pdf_name, (pub_id, offset) in sorted(PDF_TO_PUBID.items()):
        if pdf_name in PDF_OVERRIDES:
            pdf_path, offset = PDF_OVERRIDES[pdf_name]
        else:
            pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping")
            continue

        doc_dir = DIGITIZED / pub_id
        if not doc_dir.exists():
            print(f"WARNING: {doc_dir} not found, skipping {pdf_name}")
            continue

        # Find all existing page folders for this doc
        page_dirs = sorted(
            doc_dir.iterdir(),
            key=lambda p: int(m.group(1)) if (m := re.match(r"page_(\d+)$", p.name)) else 0
        )
        page_nums = [
            int(m.group(1))
            for p in page_dirs
            if p.is_dir() and (m := re.match(r"page_(\d+)$", p.name))
        ]

        if not page_nums:
            print(f"WARNING: no page folders found for {pub_id}")
            continue

        print(f"{pdf_name} -> {pub_id} (offset={offset}): {len(page_nums)} pages to render", flush=True)

        doc = fitz.open(str(pdf_path))
        total_pdf_pages = len(doc)
        mat = fitz.Matrix(DPI / 72, DPI / 72)  # 72 dpi is fitz default

        for page_num in page_nums:
            out_path = doc_dir / f"page_{page_num}" / f"{pub_id}_page_{page_num}.png"

            if out_path.exists():
                stats["skipped_exists"] += 1
                continue

            pdf_idx = page_num + offset - 1  # apply offset; fitz is 0-indexed
            if pdf_idx < 0 or pdf_idx >= total_pdf_pages:
                print(f"  WARN  page {page_num} (PDF idx {pdf_idx+1}) out of range for {pdf_name} ({total_pdf_pages} pages)")
                stats["skipped_no_folder"] += 1
                continue

            try:
                page = doc[pdf_idx]
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                pix.save(str(out_path))
                stats["generated"] += 1
                print(f"  wrote {out_path.name}", flush=True)
            except Exception as e:
                print(f"  ERROR page {page_num}: {e}")
                stats["errors"] += 1

        doc.close()

    print(f"\nDone.")
    print(f"  PNGs generated:    {stats['generated']}")
    print(f"  Already existed:   {stats['skipped_exists']}")
    print(f"  Out-of-range pages:{stats['skipped_no_folder']}")
    print(f"  Errors:            {stats['errors']}")


if __name__ == "__main__":
    main()
