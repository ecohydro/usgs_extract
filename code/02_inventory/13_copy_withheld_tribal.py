"""
Copy pages withheld for tribal data sovereignty into a separate folder.

Reads data/analysis/tribal_lands/tribal_screen_pages.csv (withhold == True),
copies each page folder data/digitized/{doc}/page_{n}/ to
data/withheld_tribal/{tribal_area or 'named_on_page'}/{doc}/page_{n}/ so the
pages can be offered directly to each tribe. Pages touching several areas are
copied into each area's folder. data/digitized is NOT modified (it is the
working archive); publication outputs (catalog, HydroShare) exclude these pages.

data/withheld_tribal/ is git-ignored. Resumable: existing destinations are skipped.
Writes data/withheld_tribal/_withheld_manifest.csv.
"""
import re
import shutil
from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data' / 'digitized'
DST = ROOT / 'data' / 'withheld_tribal'
DST.mkdir(parents=True, exist_ok=True)

p = pd.read_csv(ROOT / 'data/analysis/tribal_lands/tribal_screen_pages.csv', dtype={'id': str})
p = p[p.withhold]


tl = gpd.read_file(ROOT / 'data/analysis/spatial/tribal_lands/aiannh_2023/tl_2023_us_aiannh.shp',
                   ignore_geometry=True)
tl = tl[tl.AIANNHR.eq('F')]
STOP = {'fort', 'san', 'santa', 'big', 'little', 'north', 'south', 'east', 'west', 'lower', 'upper',
        'indian', 'river', 'lake', 'valley', 'creek', 'mission', 'el', 'la', 'los', 'las', 'pit'}
full_names = sorted({n for n in tl.NAME if n.lower() not in STOP and len(n) >= 4}, key=len, reverse=True)
# official long name for a short name; prefer the reservation/rancheria over trust land
long_name = {}
for n, nl in sorted(zip(tl.NAME, tl.NAMELSAD), key=lambda x: ('Trust' in x[1], x[1])):
    long_name.setdefault(n, nl)
first_word = {}
for n, nl in zip(tl.NAME, tl.NAMELSAD):
    w = n.split()[0].lower()
    if w not in STOP and len(w) >= 4:
        first_word.setdefault(w, set()).add(n)


def text_areas(hit):
    """Map reservation names written on the page to TIGER area names."""
    hit = str(hit)
    found = {n for n in full_names if re.search(r'(?<![A-Za-z])' + re.escape(n) + r'(?![A-Za-z])', hit, re.I)}
    if not found:
        for w in re.findall(r"[A-Za-z']+", hit):
            names = first_word.get(w.lower())
            if names and len(names) == 1:
                found |= names
    return sorted(long_name[n] for n in found) or ['named_on_page_unmatched']


def areas_for(r):
    if isinstance(r.tribal_areas, str):
        return r.tribal_areas.split('; ')
    return text_areas(r.text_hit)


rows = []
for r in p.itertuples():
    src = SRC / r.id / f'page_{int(r.page_number)}'
    for a in areas_for(r):
        folder = re.sub(r'[^\w\- ]', '', a).strip().replace(' ', '_')
        dst = DST / folder / r.id / f'page_{int(r.page_number)}'
        status = 'exists' if dst.exists() else ('missing_source' if not src.exists() else 'copied')
        if status == 'copied':
            shutil.copytree(src, dst)
        rows.append((r.id, int(r.page_number), a, folder, r.evidence, status))

m = pd.DataFrame(rows, columns=['id', 'page_number', 'tribal_area', 'folder', 'evidence', 'status'])
m.to_csv(DST / '_withheld_manifest.csv', index=False)
print(m.status.value_counts().to_string())
print(f'{m.folder.nunique()} area folders, {p.shape[0]} pages')
