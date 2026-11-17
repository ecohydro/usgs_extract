"""Tribal-lands screen for the test set draw (data sovereignty).

Mirrors Annette's rule in code/02_inventory/12_tribal_lands_screen.py (her page list is not on this
machine): a page is withheld if a rebuilt location on it lies inside a federally recognized tribal
area (Census TIGER 2023), a precise point lies within its precision distance (at least 0.5 km) of
one, or the page text names an Indian Reservation or a TIGER rancheria. Index/contents pages are
exempt from the text rule, as in her script.

withheld(doc, page) -> (True/False, evidence string)
Everything written goes to data/analysis/evaluation/metadata/tribal/, which is git-ignored:
the page lists are sensitive and are never committed or shared.
"""
import json
import re

import geopandas as gpd
import pandas as pd

import defaults as D

DIR = D.OUT / "tribal"
SHP = DIR / "aiannh_2023" / "tl_2023_us_aiannh.shp"
SPATIAL = DIR / "spatial_pages.csv"
MIN_BUFFER_KM = 0.5      # as in Annette's screen
_cache = {}


def _areas():
    if "tl" not in _cache:
        tl = gpd.read_file(SHP)
        _cache["tl"] = tl[tl.AIANNHR.eq("F")].to_crs(5070)[["NAMELSAD", "geometry"]]
    return _cache["tl"]


def spatial_pages():
    """{(doc, page): evidence} for every page with 'inside' or 'near_precise' evidence."""
    if "sp" in _cache:
        return _cache["sp"]
    if not SPATIAL.exists():
        loc = pd.read_csv(D.REPO / "data/analysis/location_validation/locations.csv", low_memory=False)
        L = loc[loc.lat.notna()].copy()
        pts = gpd.GeoDataFrame(L, geometry=gpd.points_from_xy(L.lon, L.lat), crs=4326).to_crs(5070)
        nn = gpd.sjoin_nearest(pts, _areas(), distance_col="dist_m")
        nn = nn[~nn.index.duplicated()].copy()
        km = nn.dist_m / 1000
        prec = nn.precision_class.eq("precise")
        nn["evidence"] = None
        nn.loc[km.eq(0), "evidence"] = "inside"
        nn.loc[nn.evidence.isna() & prec & (km <= nn.precision_km.fillna(0).clip(lower=MIN_BUFFER_KM)), "evidence"] = "near_precise"
        sp = nn[nn.evidence.notna()]
        sp = (sp.assign(doc_id=sp.id.astype(float).astype(int), page_number=sp.page_number.astype(float).astype(int))
                .groupby(["doc_id", "page_number"]).evidence.agg(lambda s: ",".join(sorted(set(s)))).reset_index())
        sp.to_csv(SPATIAL, index=False)
    sp = pd.read_csv(SPATIAL)
    _cache["sp"] = {(int(a), int(b)): e for a, b, e in zip(sp.doc_id, sp.page_number, sp.evidence)}
    return _cache["sp"]


def _patterns():
    if "pat" not in _cache:
        names = sorted({re.sub(r"\s+(Indian\s+)?Rancheria.*$", "", n).strip()
                        for n in _areas().NAMELSAD if "Rancheria" in n}, key=len, reverse=True)
        names = [n for n in names if len(n) >= 3]
        _cache["pat"] = (re.compile(r"((?:[A-Z][\w'.-]+\s+){0,4})Indian\s+Res(?:ervation\b|\.|\b)"),
                         re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\s+(?:Indian\s+)?Rancheria\b", re.I))
        pt = pd.read_csv(D.REPO / "data/analysis/page_types.csv", low_memory=False)
        pt = pt[pt.page_type.eq("index_contents")]
        _cache["index"] = set(zip(pt.id.astype(float).astype(int), pt.page_number.astype(float).astype(int)))
    return _cache["pat"]


def text_hit(doc, page):
    pat_res, pat_ranch = _patterns()
    if (doc, page) in _cache["index"]:
        return ""
    f = D.REDUCTO_JSON / str(doc) / f"{doc}_page_{page}.json"
    if not f.exists():
        return ""
    chunks = json.load(open(f, encoding="utf-8")).get("result", {}).get("chunks", [])
    t = " ".join(re.sub(r"<[^>]+>", " ", b.get("content") or "") for c in chunks for b in c.get("blocks", []))
    t = re.sub(r"\s+", " ", t)
    hits = [m.group(0).strip() for m in pat_res.finditer(t)] + [m.group(0).strip() for m in pat_ranch.finditer(t)]
    return "; ".join(sorted(set(hits)))[:300]


def withheld(doc, page):
    ev = [spatial_pages().get((int(doc), int(page)), "")]
    hit = text_hit(int(doc), int(page))
    if hit:
        ev.append("text_named: " + hit)
    ev = [e for e in ev if e]
    return bool(ev), " | ".join(ev)
