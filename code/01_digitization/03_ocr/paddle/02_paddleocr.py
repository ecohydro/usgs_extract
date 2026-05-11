import argparse
import os

import cv2
from paddleocr import PPStructure, save_structure_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR PPStructure on cropped table PNGs from 01_crop_tables.py."
    )
    parser.add_argument("--input-dir", required=True, help="Directory of cropped table PNGs")
    parser.add_argument("--output-dir", required=True, help="Directory to save PaddleOCR output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    table_engine = PPStructure(layout=False, image_orientation=False, recovery=False, lang="en")

    png_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".png"))
    print(f"Found {len(png_files)} cropped table images")

    for fname in png_files:
        img_path = os.path.join(args.input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {fname}")
            continue
        try:
            result = table_engine(img)
            save_structure_res(result, args.output_dir, os.path.splitext(fname)[0])
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
