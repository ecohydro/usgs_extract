"""
Claude's first-pass classification of the split-station review sample (2026-11-14), written into
data/analysis/site_definitions/split_station_review.csv (claude_class / claude_note; one verdict per
station, applied to all its points). Annette checks annette_class / annette_note.

Classes:
  same_site          same described location; coordinates differ by NWIS-vs-printed coordinate,
                     coordinate revisions, deg-min-only precision, or a garbled digit
  documented_move    the printed description names a different gauge location (bridge, distance)
  different_station  at least one point carries the coordinates of a DIFFERENT station (other
                     station on the same page, wrong link); those records are not this site
  nwis_coord_differs printed coordinates agree with each other; NWIS's own coordinate is km away
Built from reading the printed 'Location' text of every point (split_station_review.csv).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
F = ROOT / 'data/analysis/site_definitions/split_station_review.csv'

V = {
    # <0.5 km
    '10284800': ('same_site', 'printed vs NWIS coordinate'),
    '11013850': ('same_site', 'printed vs NWIS coordinate'),
    '11186340': ('same_site', 'printed vs NWIS coordinate'),
    '11218400': ('same_site', "seconds revised 52'50\" -> 52'47\", same description"),
    '11224500': ('same_site', 'seconds revision, same bridge'),
    '11230500': ('same_site', 'printed vs NWIS coordinate'),
    '11278400': ('same_site', 'seconds revision, same description'),
    '11308600': ('same_site', 'printed vs NWIS coordinate'),
    '11427760': ('same_site', 'printed vs NWIS coordinate'),
    '11449500': ('same_site', "seconds revisions; 1.5 vs 1.6 mi below Widow Cr (description revised)"),
    # 0.5-2 km
    '10277500': ('same_site', "1907 lat 37 01'45\" vs later 37 00'55\", both at Charlies Butte: coordinate revision"),
    '11047500': ('documented_move', "same Second St bridge with revised coords (33 37'15\" -> 37'34\"); 1976: 500 ft downstream of bridge"),
    '11056500': ('documented_move', "1948-54 Beaumont Ave bridge 2.3 mi upstream vs Oak Glen Rd bridge 3.0 mi; 1962-64 point does not match its printed text (pick error)"),
    '11090500': ('documented_move', 'Del Amo St bridge (1941-53) vs Centralia Rd bridge (1950-62)'),
    '11160200': ('different_station', 'two points from a multi-station measurement table (Kings Cr nr Boulder Cr etc.); rest coordinate revision'),
    '11215000': ('same_site', "deg-min-only 37 00'/118 59' (1942-57) vs seconds; same Cliff Camp bridge"),
    '11385500': ('same_site', "long 122 31'00\" revised to 30'30\" (labelled 'revised'), same description"),
    '11416500': ('same_site', "deg-min-only 39 26' vs 39 26'20\"/23\"; same location 1 mi below Bowman Lake"),
    '11433500': ('same_site', "Mountain Quarry Co. plant throughout; 121 00'20\" vs 00'51\" revision; deg-min-only 1943-49"),
    '11528440': ('same_site', "1962 point from garbled text \"123 08'3 33\"\" (parse error)"),
    # 2-10 km
    '11046500': ('nwis_coord_differs', 'printed 33 29\'30" at Camino Capistrano bridge; NWIS-number point 4.6 km away (check which station number is on that page)'),
    '11113900': ('documented_move', '1959-75 ditch 0.7 mi below Santa Clara R; 1976-77 diversion works 1.9 mi E of Saticoy'),
    '11114000': ('nwis_coord_differs', "printed 34 14'30\" consistent 1928-77; NWIS coordinate 6 km away"),
    '11147030': ('different_station', 'multi-station tables (Cottontail Cr trib etc.); two points carry other stations\' coordinates'),
    '11195520': ('different_station', 'both points belong to other stations on the page (Golden Trout Cr, Pleito Cr)'),
    '11312000': ('documented_move', '1927 below channel confluence 1.3 mi SW; 1941 highway bridge 3 mi S of Lockeford; main site 0.8 mi SE (compilation lists several sites)'),
    '11370500': ('same_site', "deg-min-only 40 37'/40 36' (1938-49); 1968 point from garbled 'Lat 0 36'04\"'"),
    '11382000': ('documented_move', "1970s description 1.2 mi downstream from Digger Cr (was 0.25 mi upstream); 1951 long 33'57\" typo; deg-min 1945-51"),
    '11409500': ('different_station', '1950 point = 0.2 mi above Moonshine Cr (Middle Yuba R below Oregon Cr); others deg-min precision'),
    '11459500': ('documented_move', '1949-65 500 ft below highway bridge vs 1966-77 100 ft above 7th St bridge; 1948 deg-min only'),
    # >10 km
    '10265700': ('documented_move', '1936 at Sherwin Hill, 3 mi above Pine Cr vs 1960-77 above diversion to Little Round Valley (11 km)'),
    '11038500': ('documented_move', '1944 at Pala diversion dam 1.8 mi E of Pala vs 1968 (revised) 4 mi SW of Pala (22 km)'),
    '11218500': ('different_station', 'temperature/WQ site 1 mi below North Fork vs gauge 0.8 mi; plss point = diversion weir with garbled long'),
    '11220000': ('different_station', '1977 point on same page = another station (37.8 N)'),
    '11265000': ('different_station', 'same page 9716 p19 lists two stations; one point is the other station'),
    '11335000': ('different_station', 'Antioch tidal station coordinates linked to Cosumnes R at Michigan Bar'),
    '11376500': ('documented_move', '1941 0.8 mi above mouth, 7 mi NE of Red Bluff vs 3 / 6.3 mi above mouth (separate earlier site)'),
    '11385000': ('different_station', '1916 point = Stony Cr above Stony Gorge Res coordinates'),
    '11470500': ('different_station', '1969 point = site near Van Arsdale Dam (other station)'),
    '11475000': ('different_station', '1966 point = Eel R at McCann (other station)'),
}

r = pd.read_csv(F, dtype=str)
key = r.site_B.str.replace('nwis:', '', regex=False)
r['claude_class'] = key.map(lambda k: V.get(k, ('', ''))[0])
r['claude_note'] = key.map(lambda k: V.get(k, ('', ''))[1])
r.to_csv(F, index=False)
st = r.drop_duplicates('site_B')
print(pd.crosstab(st.stratum, st.claude_class, margins=True).to_string())
missing = set(key) - set(V)
print('unclassified stations:', missing or 'none')
