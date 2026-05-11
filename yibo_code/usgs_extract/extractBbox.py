import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image
import pandas as pd
import os

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def crop_and_save_tables_with_padding(img, det_tables, out_dir, padding=10):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_width, img_height = img.size

    for i, det_table in enumerate(det_tables):
        bbox = det_table['bbox']
        # Adjust bounding box with padding
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(img_width, bbox[2] + padding)
        y2 = min(img_height, bbox[3] + padding)
        
        cropped_img = img.crop((x1, y1, x2, y2))

        if det_table['label'] == 'table':
            label = 'table'
        elif det_table['label'] == 'table rotated':
            label = 'table_rotated'
        else:
            continue

        out_path = os.path.join(out_dir, f"{label}_{i}.png")
        cropped_img.save(out_path)

    return None


def extractTable(data, path, out):
    try:
        objects = eval(data.loc[data['file_path'] == path, 'score'].values[0])
    except Exception as e:
        objects = data.loc[data['file_path'] == path, 'score'].values[0]
        
    image = Image.open(path).convert("RGB")
    crop_and_save_tables_with_padding(image, objects, out, 20)

    return 


if __name__ == "__main__":
    output_path = "/home/waves/data/usgs_extract/cali_tables"
    images_path = "/home/waves/data/usgs_extract/cali_images"
    table_data = pd.read_csv("/home/waves/data/usgs_extract/csvs/PaddleTestSet.csv")
    bbox_data = pd.read_csv("/home/waves/data/usgs_extract/csvs/cali_results_1970_clean.csv")
    image_paths = table_data["pathWithTable"]
    for path in image_paths:
        file_name = path.split("/")[-1]
        outpath = os.path.join(output_path, file_name)
        try:
            extractTable(bbox_data, path, outpath)
        except Exception as e:
            print(f"An error occurred: {e}")