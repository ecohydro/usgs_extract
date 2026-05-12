import csv, os, sys, math
sys.stdout.reconfigure(encoding='utf-8')

CANDIDATE_ROOTS = [
    '/Volumes/AHILTON_2/usgs_extract_data/digitized',       # Anna's Mac (external drive)
    r'c:\Users\aeliz\Dropbox\usgs_extract\data\digitized',  # Annette's Windows
]
DIGITIZED_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.isdir(p)), None)
if DIGITIZED_ROOT is None:
    sys.exit('ERROR: digitized data root not found — is the external drive plugged in?')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
META_PATH = os.path.join(REPO_ROOT, 'data', 'metadata', 'main_metadata.csv')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'data', 'analysis', 'streamflow', 'annual')
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
        json_path = os.path.join(DIGITIZED_ROOT, doc_id, f'page_{page}', f'{doc_id}_page_{page}.json')
        if not os.path.isfile(json_path):
            continue
        candidates.append({
            'doc_id': doc_id,
            'page_number': page,
            'watersource_name': row.get('watersource_name', '').strip(),
            'actual_latitude': row.get('actual_latitude', '').strip(),
            'actual_longitude': row.get('actual_longitude', '').strip(),
            'inferred_latitude': row.get('inferred_latitude', '').strip(),
            'inferred_longitude': row.get('inferred_longitude', '').strip(),
            'temporal_resolution': row.get('temporal_resolution', '').strip(),
            'dates_of_recording': row.get('dates_of_recording', '').strip(),
            'units_of_measurement': row.get('units_of_measurement', '').strip(),
        })

FIELDNAMES = [
    'doc_id', 'page_number', 'watersource_name',
    'actual_latitude', 'actual_longitude',
    'inferred_latitude', 'inferred_longitude',
    'temporal_resolution', 'dates_of_recording', 'units_of_measurement',
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
print(f"Digitized root:    {DIGITIZED_ROOT}")
