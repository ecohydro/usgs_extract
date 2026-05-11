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
        if 'sacramento' not in source:
            continue
        if row.get('water_type','').strip().lower() != 'stream discharge':
            continue
        dates = row.get('dates_of_recording','')
        start_yr = extract_start_year(dates)
        if start_yr and start_yr < 1945:
            results.append({
                'id': row['id'],
                'page': row['page_number'],
                'source': row['watersource_name'],
                'dates': dates,
                'res': row['temporal_resolution'],
                'units': row['units_of_measurement'],
                'start_yr': start_yr,
            })

print(f'Sacramento stream discharge rows with start year < 1945: {len(results)}')
print()

by_source = defaultdict(list)
for r in results:
    by_source[r['source']].append(r)

print(f'Unique site names: {len(by_source)}')
print()

def sort_key(item):
    src, rows = item
    return min(r['start_yr'] for r in rows)

for src, rows in sorted(by_source.items(), key=sort_key):
    min_yr = min(r['start_yr'] for r in rows)
    resolutions = set(r['res'] for r in rows)
    dates_sample = rows[0]['dates']
    print(f'  [{min_yr}]  {src}  |  {len(rows)} pages  |  res: {resolutions}  |  e.g. {dates_sample}')
