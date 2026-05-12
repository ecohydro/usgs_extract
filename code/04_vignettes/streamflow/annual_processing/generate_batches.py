import csv, os, sys, math
sys.stdout.reconfigure(encoding='utf-8')

META_PATH = r'c:\Users\aeliz\Dropbox\usgs_extract\data\metadata\main_metadata.csv'
DIGITIZED_ROOT = r'c:\Users\aeliz\Dropbox\usgs_extract\data\digitized'
OUTPUT_DIR = r'c:\Users\aeliz\Dropbox\usgs_extract\data\analysis\streamflow\annual'
BATCH_SIZE = 150

ANNUAL_KEYWORDS = ['annual', 'yearly', 'year']

os.makedirs(OUTPUT_DIR, exist_ok=True)

candidates = []
with open(META_PATH, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('water_type', '').strip().lower() != 'stream discharge':
            continue
        res = row.get('temporal_resolution', '').strip().lower()
        if not any(k in res for k in ANNUAL_KEYWORDS):
            continue
        doc_id = row['id']
        page = row['page_number']
        page_dir = os.path.join(DIGITIZED_ROOT, doc_id, f'page_{page}')
        if not os.path.isdir(page_dir):
            continue
        csvs = [f for f in os.listdir(page_dir) if f.endswith('.csv') and '_metadata' not in f]
        if not csvs:
            continue
        lat = row.get('actual_latitude', '').strip() or row.get('inferred_latitude', '').strip()
        lon = row.get('actual_longitude', '').strip() or row.get('inferred_longitude', '').strip()
        candidates.append({
            'doc_id': doc_id,
            'page_number': page,
            'watersource_name': row.get('watersource_name', '').strip(),
            'temporal_resolution': row.get('temporal_resolution', '').strip(),
            'dates_of_recording': row.get('dates_of_recording', '').strip(),
            'units_of_measurement': row.get('units_of_measurement', '').strip(),
            'latitude': lat,
            'longitude': lon,
            'csv_files': ','.join(csvs),
        })

FIELDNAMES = [
    'doc_id', 'page_number', 'watersource_name', 'temporal_resolution',
    'dates_of_recording', 'units_of_measurement', 'latitude', 'longitude', 'csv_files',
]

n_batches = math.ceil(len(candidates) / BATCH_SIZE)
for i in range(n_batches):
    batch = candidates[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    batch_path = os.path.join(OUTPUT_DIR, f'batch_{i+1:03d}.csv')
    with open(batch_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(batch)

print(f"Total candidates:  {len(candidates)}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Batches created:   {n_batches}")
print(f"Output dir:        {OUTPUT_DIR}")
