import csv, re, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

path = r'c:\Users\aeliz\Dropbox\usgs_extract\data\metadata\main_metadata.csv'

def extract_start_year(dates_str):
    m = re.search(r'\b(1[0-9]{3})\b', dates_str)
    if m:
        return int(m.group(1))
    return None

results = []
with open(path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        source = (row.get('watersource_name','') + ' ' + row.get('location','')).lower()
        if 'kennett' not in source:
            continue
        if row.get('water_type','').strip().lower() != 'stream discharge':
            continue
        dates = row.get('dates_of_recording','')
        start_yr = extract_start_year(dates)
        results.append({
            'id': row['id'],
            'page': row['page_number'],
            'source': row['watersource_name'],
            'location': row['location'],
            'dates': dates,
            'res': row['temporal_resolution'],
            'units': row['units_of_measurement'],
            'start_yr': start_yr,
            'lat': row.get('actual_latitude') or row.get('inferred_latitude'),
            'lon': row.get('actual_longitude') or row.get('inferred_longitude'),
        })

print(f'Kennett stream discharge rows: {len(results)}')
print()
for r in sorted(results, key=lambda x: (x['start_yr'] or 9999, x['id'], int(x['page']))):
    print(f"  doc {r['id']}  page {r['page']:>4}  |  {r['source']}")
    print(f"    dates: {r['dates']}")
    print(f"    res: {r['res']}")
    print(f"    units: {r['units']}")
    print(f"    coords: {r['lat']}, {r['lon']}")
    print()
