import argparse
import json
import os
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify which PDFs from the publication list were downloaded.")
    parser.add_argument("--input-csv", required=True, help="CSV with 'Publication ID' column")
    parser.add_argument("--download-dir", required=True, help="Directory containing downloaded PDFs")
    parser.add_argument("--output-json", default="missing_ids.json", help="Path to save list of missing IDs")
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    expected_ids = set(data["Publication ID"].astype(int))

    downloaded_ids = set()
    for f in os.listdir(args.download_dir):
        if f.endswith(".pdf"):
            try:
                downloaded_ids.add(int(f[:-4]))
            except ValueError:
                pass

    missing = sorted(expected_ids - downloaded_ids)
    print(f"Expected:  {len(expected_ids)}")
    print(f"Downloaded:{len(downloaded_ids)}")
    print(f"Missing:   {len(missing)}  ({1 - len(downloaded_ids) / len(expected_ids):.1%} of total)")

    if missing:
        with open(args.output_json, "w") as f:
            json.dump(missing, f, indent=2)
        print(f"Missing IDs written to {args.output_json}")
