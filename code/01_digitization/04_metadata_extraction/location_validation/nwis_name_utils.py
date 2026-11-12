"""
Shared NWIS station-NAME parsing (11_nwis_station_names.py, stream check in 04_build_locations.py).
"<stream> <at|near|above|below> <place>" -> stream name + set of (qualifier, place); USGS
abbreviations expanded (R -> RIVER, C -> CREEK, NF -> NORTH FORK, CYN -> CANYON ...).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gnis_utils import hav

EXPAND = {'R': 'RIVER', 'RIV': 'RIVER', 'C': 'CREEK', 'CR': 'CREEK', 'CK': 'CREEK', 'CRK': 'CREEK',
          'CYN': 'CANYON', 'CN': 'CANAL', 'CNL': 'CANAL', 'LK': 'LAKE', 'RES': 'RESERVOIR',
          'NF': 'NORTH FORK', 'SF': 'SOUTH FORK', 'MF': 'MIDDLE FORK', 'EF': 'EAST FORK',
          'WF': 'WEST FORK', 'FK': 'FORK', 'TR': 'TRIBUTARY', 'TRIB': 'TRIBUTARY',
          'SPGS': 'SPRINGS', 'SPG': 'SPRING', 'PP': 'POWERPLANT', 'DIV': 'DIVERSION',
          'MTN': 'MOUNTAIN', 'HWY': 'HIGHWAY', 'BR': 'BRIDGE', 'SNTA': 'SANTA', 'SL': 'SLOUGH',
          'DAMSITE': 'DAM SITE', 'JCT': 'JUNCTION'}
QUAL_ROW = {'AT': 'near', 'NEAR': 'near', 'NR': 'near', 'ABOVE': 'above', 'AB': 'above',
            'BELOW': 'below', 'BL': 'below', 'BLW': 'below'}
QUAL_NWIS = dict(QUAL_ROW, A='near')        # NWIS writes "A" for "at"
DROP = {'CA', 'CAL', 'CALIF', 'CALIFORNIA', 'THE', 'CONTINUED', 'COUNTY'}


def words(text):
    out = []
    for w in re.findall(r"[A-Z0-9]+", str(text).upper().replace("'", '')):
        out.extend(EXPAND.get(w, w).split())
    return out


def parse(text, qual):
    """'Santa Ynez R bl Gibraltar Dam nr Santa Barbara' -> ('SANTA YNEZ RIVER', {(below, GIBRALTAR DAM), (near, SANTA BARBARA)})."""
    stream, segs, cur = [], [], None
    for w in words(text):
        if w in DROP:
            continue
        if w in qual:
            cur = [qual[w], []]
            segs.append(cur)
        elif cur is None:
            stream.append(w)
        else:
            cur[1].append(w)
    segs = {(q, ' '.join(p)) for q, p in segs if p}
    return ' '.join(stream), segs


def row_text(ws, loc):
    """Water-source name, plus the location text when it begins with a qualifier."""
    ws = '' if pd.isna(ws) else re.sub(r'\(.*?\)', ' ', str(ws))
    loc = '' if pd.isna(loc) else str(loc)
    if not re.search(r'\b(at|near|nr|above|below|ab|bl)\b', ws, re.I) and \
            re.match(r'\s*(at|near|nr\.?|above|below)\b', loc, re.I):
        ws = f"{ws} {loc.split(',')[0]}"
    return ws.split(',')[0] if re.search(r'\b(at|near|nr|above|below)\b', ws.split(',')[0], re.I) else ws




def load_stations(root):
    """NWIS California stream stations with parsed stream / segments; dict stream -> stations."""
    nwis = pd.read_csv(Path(root) / 'data/analysis/nwis_sites/ca_stream_sites_expanded.csv',
                       dtype={'site_no': str}, low_memory=False).drop_duplicates('site_no')
    nwis = nwis[nwis['dec_lat_va'].notna()].copy()
    nwis[['stream', 'segs']] = [parse(n, QUAL_NWIS) for n in nwis['station_nm']]
    return nwis, {s: g for s, g in nwis.groupby('stream') if s}


def stream_spread_km(by_stream):
    """Largest distance between two NWIS stations sharing a stream name (name uniqueness proxy)."""
    out = {}
    for s, g in by_stream.items():
        la, lo = g['dec_lat_va'].values, g['dec_long_va'].values
        out[s] = float(hav(la[:, None], lo[:, None], la[None, :], lo[None, :]).max())
    return out
