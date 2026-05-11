import argparse
import json
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Reducto JSON output to CSV, one file per table."
    )
    parser.add_argument("--json-dir", required=True, help="Directory of Reducto JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to save output CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(f for f in os.listdir(args.json_dir) if f.endswith(".json"))
    print(f"Found {len(json_files)} JSON files")

    for fname in json_files:
        json_path = os.path.join(args.json_dir, fname)
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        chunks = data.get("result", {}).get("chunks", [])
        if not chunks:
            print(f"No chunks: {fname}")
            continue

        base = os.path.splitext(fname)[0]
        table_counter = 0

        for chunk in chunks:
            for block in chunk.get("blocks", []):
                if block.get("type") == "Table":
                    html = block.get("content", "")
                    if not html:
                        continue
                    try:
                        for df in pd.read_html(html):
                            table_counter += 1
                            csv_path = os.path.join(args.output_dir, f"{base}_table{table_counter}.csv")
                            df.to_csv(csv_path, index=False)
                            print(f"Saved: {csv_path}")
                    except Exception as e:
                        print(f"Skipping table in {fname}: {e}")

        if table_counter == 0:
            print(f"No tables: {fname}")
