import os
import json
import pandas as pd

# Paths
input_root = "/home/waves/data/usgs_extract/ReductJson"

# Store records
records = []

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(input_root):
    for filename in filenames:
        if filename.endswith(".json"):
            json_path = os.path.join(dirpath, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            chunks = data.get("result", {}).get("chunks", [])
            table_count = 0
            for content in chunks:
                for block in content.get("blocks", []):
                    if block.get("type") == "Table":
                        table_count += 1

            if filename[0] == "r":
                id = "SantaBarbaraDoc"
                page_num = "FULL"
            else:
                id = filename.split('_')[0]
                page_num = filename.split('_')[-1][:-5]         # JSON filename without extension

            records.append({
                "id": id,
                "page_num": page_num,
                "number_of_tables": table_count
            })

# Create DataFrame
df_counts = pd.DataFrame(records)
df_counts.to_csv("/home/waves/data/usgs_extract/metadataResults/table_counts_by_page.csv", index=False)
