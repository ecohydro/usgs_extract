"""
Shared NWIS station-name matching (05_nwis_station_numbers.py for California stations,
10_out_of_state.py for stations in other states): distinctive name words.
"""
import re

ABBREV = {'R': 'RIVER', 'C': 'CREEK', 'CR': 'CREEK', 'CK': 'CREEK', 'NR': 'NEAR', 'AB': 'ABOVE',
          'BL': 'BELOW', 'BLW': 'BELOW', 'CN': 'CANAL', 'CNL': 'CANAL', 'RES': 'RESERVOIR',
          'NF': 'NORTH FORK', 'SF': 'SOUTH FORK', 'MF': 'MIDDLE FORK', 'EF': 'EAST FORK',
          'WF': 'WEST FORK', 'FK': 'FORK', 'TR': 'TRIBUTARY', 'TRIB': 'TRIBUTARY', 'L': 'LAKE',
          'SPGS': 'SPRINGS', 'SPG': 'SPRING', 'PP': 'POWERPLANT', 'DIV': 'DIVERSION', 'MTN': 'MOUNTAIN'}
GENERIC = {'RIVER', 'CREEK', 'NEAR', 'ABOVE', 'BELOW', 'CANAL', 'RESERVOIR', 'NORTH', 'SOUTH',
           'MIDDLE', 'EAST', 'WEST', 'FORK', 'TRIBUTARY', 'LAKE', 'SPRING', 'SPRINGS', 'CALIF',
           'CALIFORNIA', 'MOUTH', 'STATION', 'GAGE', 'DIVERSION', 'POWERPLANT', 'DITCH', 'MILE',
           'MILES', 'BRIDGE', 'COUNTY', 'BASIN', 'WATER', 'DISCHARGE', 'INFLOW', 'OUTFLOW',
           'MAIN', 'UPPER', 'LOWER', 'LITTLE', 'GRAND', 'CITY', 'VALLEY', 'FORKS', 'NEAR'}


def tokens(text):
    words = re.findall(r'[A-Z]+', str(text).upper())
    out = set()
    for w in words:
        for x in ABBREV.get(w, w).split():
            if len(x) >= 4 and x not in GENERIC:
                out.add(x)
    return out
