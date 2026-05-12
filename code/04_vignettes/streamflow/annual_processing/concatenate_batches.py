import csv, os, sys, glob
sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ANNUAL_DIR = os.path.join(REPO_ROOT, 'data', 'analysis', 'streamflow', 'annual')

OUTPUTS = {
    'annual_streamflow': {
        'pattern': 'annual_streamflow_batch_*.csv',
        'out': 'annual_streamflow.csv',
    },
    'monthly_streamflow': {
        'pattern': 'monthly_streamflow_batch_*.csv',
        'out': 'monthly_streamflow.csv',
    },
    'extraction_log': {
        'pattern': 'extraction_log_batch_*.csv',
        'out': 'extraction_log.csv',
    },
}

for label, cfg in OUTPUTS.items():
    batch_files = sorted(glob.glob(os.path.join(ANNUAL_DIR, cfg['pattern'])))
    if not batch_files:
        print(f"{label}: no batch files found, skipping")
        continue

    out_path = os.path.join(ANNUAL_DIR, cfg['out'])
    total_rows = 0
    header_written = False

    with open(out_path, 'w', newline='', encoding='utf-8') as outf:
        writer = None
        for batch_file in batch_files:
            with open(batch_file, newline='', encoding='utf-8') as inf:
                reader = csv.DictReader(inf)
                if writer is None:
                    writer = csv.DictWriter(outf, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1

    print(f"{label}: {len(batch_files)} batch files → {total_rows} rows → {cfg['out']}")

print("Done.")
