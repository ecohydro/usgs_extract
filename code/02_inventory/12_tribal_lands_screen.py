"""
Tribal-lands screen (data sovereignty).

Identifies pages whose data lie on tribal lands so they can be withheld from
publication and offered directly to the tribes (Annette's decision, 2026-10-29).
The unit of withholding is the PAGE: a page image/JSON shows every row on it,
so if any record on a page is on tribal land the whole page folder is withheld.

Boundaries: Census TIGER/Line 2023 AIANNH (national; covers out-of-state
records too), downloaded to data/analysis/spatial/tribal_lands/aiannh_2023/.
  - Federally recognized (AIANNHR = 'F'): always screened. All area types
    (reservations, off-reservation trust land, joint-use, statistical areas).
  - State recognized (AIANNHR = 'S'): screened only if INCLUDE_STATE = True.
NOTE: these are CURRENT (2023) boundaries, not historical ones.

Evidence, per metadata row (locations from the rebuilt location table, never
from raw LLM coordinates):
  inside        located point falls inside a tribal area polygon (any precision)
  near_precise  precise point within its precision distance (min MIN_BUFFER_KM) of an area
  text_named    the page itself names a reservation/rancheria ("in Hoopa Valley
                Indian Reservation", "Pala Indian Reservation Canal",
                "<TIGER rancheria name> Rancheria"); index/contents pages excluded
  possible      approximate or basin point whose uncertainty radius touches an area
                (review list; withheld only if WITHHOLD_POSSIBLE = True)
Pages with 'inside', 'near_precise' or 'text_named' evidence are WITHHELD.

Outputs (data/analysis/tribal_lands/):
  tribal_screen_rows.csv    one row per spatially flagged metadata row
  tribal_screen_pages.csv   one row per flagged page: withhold (bool), evidence, areas, text hit
  tribal_screen_summary.csv counts
"""
import re
from pathlib import Path
import geopandas as gpd
import pandas as pd

INCLUDE_STATE = False      # state-recognized tribal areas (optional per Annette)
WITHHOLD_POSSIBLE = False  # also withhold approximate/basin points whose radius touches an area
MIN_BUFFER_KM = 0.5        # minimum buffer for precise points

ROOT = Path(__file__).resolve().parents[2]


def norm_keys(df):
    """Doc ids are written as '10428' in some files and '10428.0' in others."""
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['page_number'] = pd.to_numeric(df['page_number']).astype('Int64')
    return df

OUT = ROOT / 'data' / 'analysis' / 'tribal_lands'
OUT.mkdir(parents=True, exist_ok=True)

tl = gpd.read_file(ROOT / 'data/analysis/spatial/tribal_lands/aiannh_2023/tl_2023_us_aiannh.shp')
keep = tl.AIANNHR.eq('F') | (INCLUDE_STATE & tl.AIANNHR.eq('S'))
tl = tl[keep].to_crs(5070)[['NAME', 'NAMELSAD', 'AIANNHR', 'MTFCC', 'geometry']]
print(f'tribal areas screened: {len(tl)} (federal {tl.AIANNHR.eq("F").sum()}, state {tl.AIANNHR.eq("S").sum()})')

# ---------- spatial evidence ----------
loc = pd.read_csv(ROOT / 'data/analysis/location_validation/locations.csv', dtype={'id': str})
L = norm_keys(loc[loc.lat.notna()].copy())
pts = gpd.GeoDataFrame(L, geometry=gpd.points_from_xy(L.lon, L.lat), crs=4326).to_crs(5070)
nn = gpd.sjoin_nearest(pts, tl[['NAMELSAD', 'AIANNHR', 'geometry']], distance_col='dist_m')
nn = nn[~nn.index.duplicated()].copy()
nn['dist_km'] = nn.dist_m / 1000
buf = nn.precision_km.fillna(0)
prec = nn.precision_class.eq('precise')
nn['evidence'] = None
nn.loc[nn.dist_km.eq(0), 'evidence'] = 'inside'
nn.loc[nn.evidence.isna() & prec & (nn.dist_km <= buf.clip(lower=MIN_BUFFER_KM)), 'evidence'] = 'near_precise'
nn.loc[nn.evidence.isna() & ~prec & (nn.dist_km <= buf), 'evidence'] = 'possible'
sp = nn[nn.evidence.notna()][['meta_row', 'id', 'page_number', 'water_type', 'precision_class',
                             'precision_km', 'evidence', 'NAMELSAD', 'AIANNHR', 'dist_km']]
sp = sp.rename(columns={'NAMELSAD': 'tribal_area', 'AIANNHR': 'recognition'})

# ---------- text evidence (page names a reservation / rancheria) ----------
B = pd.read_parquet(ROOT / 'data/analysis/page_blocks.parquet', columns=['id', 'page_number', 'content'])
B['content'] = B.content.str.replace(r'<[^>]+>', ' ', regex=True)
P = B.groupby(['id', 'page_number']).content.agg(' '.join).reset_index()
P = norm_keys(P)
pt = norm_keys(pd.read_csv(ROOT / 'data/analysis/page_types.csv', dtype={'id': str}))
P = P.merge(pt[['id', 'page_number', 'page_type']], on=['id', 'page_number'], how='left')
P = P[~P.page_type.eq('index_contents')]

# "Indian Reservation" / "Indian Res." always counts.
# "<Name> Rancheria" counts only if <Name> is a TIGER rancheria name (avoids
# "Rancheria Creek", "Rancheria Flat", "Rancheria Gulch" place names).
ranch_names = sorted({re.sub(r'\s+(Indian\s+)?Rancheria.*$', '', n).strip()
                      for n in tl.NAMELSAD if 'Rancheria' in n}, key=len, reverse=True)
ranch_names = [n for n in ranch_names if len(n) >= 3]
pat_res = re.compile(r"((?:[A-Z][\w'.-]+\s+){0,4})Indian\s+Res(?:ervation\b|\.|\b)")
pat_ranch = re.compile(r'\b(' + '|'.join(re.escape(n) for n in ranch_names) + r')\s+(?:Indian\s+)?Rancheria\b', re.I)

text_rows = []
for r in P.itertuples():
    t = re.sub(r'\s+', ' ', r.content)
    hits = [m.group(0).strip() for m in pat_res.finditer(t)] + [m.group(0).strip() for m in pat_ranch.finditer(t)]
    if hits:
        text_rows.append((r.id, r.page_number, '; '.join(sorted(set(hits)))[:300]))
tx = pd.DataFrame(text_rows, columns=['id', 'page_number', 'text_hit'])

# ---------- page table ----------
pg_sp = (sp.groupby(['id', 'page_number'])
           .agg(evidence=('evidence', lambda s: ','.join(sorted(set(s)))),
                tribal_areas=('tribal_area', lambda s: '; '.join(sorted(set(s)))),
                n_rows_flagged=('meta_row', 'size'))
           .reset_index())
pages = pg_sp.merge(tx, on=['id', 'page_number'], how='outer')
pages['evidence'] = pages.evidence.fillna('')
has_t = pages.text_hit.notna()
pages.loc[has_t, 'evidence'] = (pages.loc[has_t, 'evidence'] + ',text_named').str.strip(',')
strong = pages.evidence.str.contains('inside|near_precise|text_named')
pages['withhold'] = strong | (WITHHOLD_POSSIBLE & pages.evidence.str.contains('possible'))
pages['review_only'] = ~pages.withhold
pages = pages.sort_values(['withhold', 'id', 'page_number'], ascending=[False, True, True])

sp.to_csv(OUT / 'tribal_screen_rows.csv', index=False)
pages.to_csv(OUT / 'tribal_screen_pages.csv', index=False)

md = norm_keys(pd.read_csv(ROOT / 'data/metadata/main_metadata.csv', usecols=['id', 'page_number'], dtype={'id': str}))
w = pages[pages.withhold][['id', 'page_number']]
ev = lambda k: int(pages.evidence.str.contains(k).sum())
summ = pd.DataFrame([
    ('settings', f'INCLUDE_STATE={INCLUDE_STATE}, WITHHOLD_POSSIBLE={WITHHOLD_POSSIBLE}, MIN_BUFFER_KM={MIN_BUFFER_KM}'),
    ('pages with any flag', len(pages)),
    ('pages withheld', int(pages.withhold.sum())),
    ('pages review only (possible)', int(pages.review_only.sum())),
    ('pages with inside evidence', ev('inside')),
    ('pages with near_precise evidence', ev('near_precise')),
    ('pages with text_named evidence', ev('text_named')),
    ('pages withheld by text only', int((pages.evidence == 'text_named').sum())),
    ('documents with a withheld page', int(w.id.nunique())),
    ('metadata rows on withheld pages', len(md.merge(w))),
], columns=['item', 'value'])
summ.to_csv(OUT / 'tribal_screen_summary.csv', index=False)
print(summ.to_string(index=False))
print(pages[pages.withhold].tribal_areas.dropna().str.split('; ').explode().value_counts().head(15).to_string())
