# AUTOGENERATED! DO NOT EDIT! File to edit: ../01_utilities.ipynb.

# %% auto 0
__all__ = ['convert_page', 'image_path_from_page']

# %% ../01_utilities.ipynb 3
import os
from pdf2image import convert_from_path
#from torchvision import transforms

# %% ../01_utilities.ipynb 4
# Function to handle the conversion of a single page
def convert_page(args):
    pdf_path, output_folder, page_num = args
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300, fmt='png')
    if images:
        image_path = os.path.join(output_folder, f'{base_filename}_page_{page_num}.png')
        return image_path, images[0]
    return None

def image_path_from_page(pdf_path, output_folder, page_num):
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(output_folder, f'{base_filename}_page_{page_num}.png')
