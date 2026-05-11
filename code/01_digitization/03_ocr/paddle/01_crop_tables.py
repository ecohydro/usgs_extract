import argparse
import ast
import os

import pandas as pd
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop detected table regions from page PNGs using Table Transformer detection results."
    )
    parser.add_argument("--png-dir", required=True, help="Directory of full-page PNGs")
    parser.add_argument("--detections-csv", required=True,
                        help="Detection results CSV from 02_table_detection/01_detect.py")
    parser.add_argument("--output-dir", required=True, help="Directory to save cropped table PNGs")
    parser.add_argument("--padding", type=int, default=20,
                        help="Padding in pixels around each bounding box (default: 20)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.detections_csv)
    print(f"Loaded {len(df)} detections across {df['filename'].nunique()} pages")

    for filename, group in df.groupby("filename"):
        png_path = os.path.join(args.png_dir, filename)
        if not os.path.exists(png_path):
            print(f"PNG not found: {png_path}")
            continue

        img = Image.open(png_path).convert("RGB")
        img_w, img_h = img.size
        base = os.path.splitext(filename)[0]

        for i, (_, row) in enumerate(group.iterrows(), start=1):
            bbox = ast.literal_eval(row["bbox"])
            x1 = max(0, bbox[0] - args.padding)
            y1 = max(0, bbox[1] - args.padding)
            x2 = min(img_w, bbox[2] + args.padding)
            y2 = min(img_h, bbox[3] + args.padding)

            label = "table_rotated" if "rotated" in str(row["label"]) else "table"
            out_path = os.path.join(args.output_dir, f"{base}_{label}_{i}.png")
            img.crop((x1, y1, x2, y2)).save(out_path)
            print(f"Saved: {out_path}")
