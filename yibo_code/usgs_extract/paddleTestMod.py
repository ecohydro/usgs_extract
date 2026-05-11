import os
import cv2
from paddleocr import PPStructure, save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

def paddlerun(pages_paths):
    for path in pages_paths:
        images = os.listdir(path)
        for image in images:
            try:
                name = os.path.basename(path).split('.')[0] + "_" + image[:-4]
                img_path = os.path.join(path, image)
                img = cv2.imread(img_path)
                result = table_engine(img)
                save_structure_res(result, save_folder, name)

            # Load Documents

                # h, w, _ = img.shape
                # res = sorted_layout_boxes(result, w)
                # convert_info_docx(img, res, save_folder, name)
            except Exception as e:
                print(f"Found Exception {e}")

if __name__ == "__main__":

    table_engine = PPStructure(layout=False, image_orientation=False, recovery=False, lang='en')
    save_folder = "/home/waves/data/usgs_extract/paddleOutput"
    tables_path = '/home/waves/data/usgs_extract/cali_tables'
    pages = os.listdir(tables_path)
    pages_paths = []

    try:
        print("Attempting to list all tables")
        for page in pages:
            if page[-4:] == ".png":
                pages_paths.append(os.path.join(tables_path, page))
        print("all table paths are stored in mem.")
    except Exception as e:
        print(f"{e}")

    paddlerun(pages_paths)