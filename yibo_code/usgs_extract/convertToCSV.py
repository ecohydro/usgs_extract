import os
import json
import pandas as pd

# Paths
input_root = "/home/waves/data/usgs_extract/ReductJson"
output_root = "/home/waves/data/usgs_extract/ReductCSVs"

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(input_root):
    for filename in filenames:
        if filename.endswith(".json"):
            json_path = os.path.join(dirpath, filename)
            
            # Load JSON
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            chunks = data.get("result", {}).get("chunks", [])
            if not chunks:
                print(f"No chunks found in {json_path}")
                continue

            table_counter = 0

            # Determine relative folder to maintain structure
            relative_folder = os.path.relpath(dirpath, input_root)
            output_dir = os.path.join(output_root, relative_folder)
            os.makedirs(output_dir, exist_ok=True)

            for content in chunks:
                for block in content.get("blocks", []):
                    if block.get("type") == "Table":
                        html_table = block.get("content", "")
                        if html_table:
                            try:
                                dfs = pd.read_html(html_table)
                                for i, df in enumerate(dfs):
                                    table_counter += 1
                                    base_name = os.path.splitext(filename)[0]
                                    csv_filename = f"{base_name}_table{table_counter}.csv"
                                    csv_path = os.path.join(output_dir, csv_filename)
                                    df.to_csv(csv_path, index=False)
                                    print(f"✅ Saved: {csv_path}")
                            except Exception as e:
                                print(f"Skipping table in {json_path} due to error: {e}")

            if table_counter == 0:
                print(f"No tables saved from {json_path}")
