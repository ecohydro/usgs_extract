import os
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoModelForObjectDetection
import torch
from PIL import Image
from torchvision import transforms
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

cali_csv = "/home/waves/data/usgs_extract/csvs/cali.csv"
pdf_dir = "/home/waves/data/usgs_extract/raw_pdf"
output_path = "/home/waves/data/usgs_extract/validation_test/images"
table_path = "/home/waves/data/usgs_extract/validation_test/tablesBbox"

# def fig2img(fig):
#     """Convert a Matplotlib figure to a PIL Image and return it"""
#     import io
#     buf = io.BytesIO()
#     fig.savefig(buf)
#     buf.seek(0)
#     img = Image.open(buf)
#     return img


# def visualize_detected_tables(img, det_tables, out_path=None):
#     plt.imshow(img, interpolation="lanczos")
#     fig = plt.gcf()
#     fig.set_size_inches(20, 20)
#     ax = plt.gca()

#     for det_table in det_tables:
#         bbox = det_table['bbox']

#         if det_table['label'] == 'table':
#             facecolor = (1, 0, 0.45)
#             edgecolor = (1, 0, 0.45)
#             alpha = 0.3
#             linewidth = 2
#             hatch='//////'
#         elif det_table['label'] == 'table rotated':
#             facecolor = (0.95, 0.6, 0.1)
#             edgecolor = (0.95, 0.6, 0.1)
#             alpha = 0.3
#             linewidth = 2
#             hatch='//////'
#         else:
#             continue

#         rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
#                                     edgecolor='none',facecolor=facecolor, alpha=0.1)
#         ax.add_patch(rect)
#         rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
#                                     edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
#         ax.add_patch(rect)
#         rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
#                                     edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
#         ax.add_patch(rect)

#     plt.xticks([], [])
#     plt.yticks([], [])

#     legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
#                                 label='Table', hatch='//////', alpha=0.3),
#                         Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
#                                 label='Table (rotated)', hatch='//////', alpha=0.3)]
#     plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
#                     fontsize=10, ncol=2)
#     plt.gcf().set_size_inches(10, 10)
#     plt.axis('off')

#     if out_path is not None:
#       plt.savefig(out_path, bbox_inches='tight', dpi=150)

#     return fig

model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

class MaxResize(object):
    def __init__(self, max_size = 800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width,height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

class ToGrayscale(object):
    def __call__(self, img):
        # Convert PIL image to grayscale
        return img.convert("L")

class ApplyThresholding(object):
    def __call__(self, img):
        # Convert PIL image to OpenCV format
        img_cv = np.array(img)
        # Apply binary + Otsu thresholding
        _, img_thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(img_thresh)

class ApplyDilation(object):
    def __init__(self, kernel_size=3):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, img):
        # Convert PIL image to OpenCV format
        img_cv = np.array(img)
        # Apply dilation
        img_dilated = cv2.dilate(img_cv, self.kernel, iterations=1)
        return Image.fromarray(img_dilated)


class EnhanceContrast(object):
    def __call__(self, img):
        img_cv = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_cv)
        return Image.fromarray(img_enhanced)

class EdgeDetection(object):
    def __call__(self, img):
        img_cv = np.array(img)
        edges = cv2.Canny(img_cv, 100, 200)
        return Image.fromarray(edges)

class GrayscaleTo3Channel(object):
    def __call__(self, img):
        # Convert single-channel image to 3-channel by duplicating the grayscale channel
        img_array = np.array(img)
        img_3ch = np.stack([img_array]*3, axis=-1)
        return Image.fromarray(img_3ch)

detection_transform = transforms.Compose([
    ToGrayscale(),
    ApplyThresholding(),
    # ApplyDilation(kernel_size=3),  # Adjust kernel_size as needed
    GrayscaleTo3Channel(),
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])  # Same mean and std for each channel
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# update id2label to include "no object"
id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def runModel(files):
    file_path = []
    score_list = []
    for path in files:
        image = Image.open(path).convert("RGB")
        pixelval = detection_transform(image).unsqueeze(0)
        pixelval = pixelval.to(device)
        with torch.no_grad():
            outputs = model(pixelval)
        objects = outputs_to_objects(outputs, image.size, id2label)
        file_path.append(path)
        score_list.append(objects)
    data = pd.DataFrame({"file_path":file_path, "score": score_list})
    data.to_csv(os.path.join(table_path, "results.csv"), index=False)
    return data
    
if __name__ == "__main__":
    try:
        files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".png")]
        runModel(files)
    except Exception as e:
        print(f"An error occurred: {e}")
