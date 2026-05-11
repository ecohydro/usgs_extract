import argparse
import os
import csv

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForObjectDetection


class MaxResize:
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        scale = self.max_size / max(width, height)
        return image.resize((int(round(scale * width)), int(round(scale * height))))


class GrayscaleTo3Channel:
    def __call__(self, img):
        arr = np.array(img)
        return Image.fromarray(np.stack([arr] * 3, axis=-1))


detection_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),
    transforms.Lambda(lambda img: Image.fromarray(
        cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    )),
    GrayscaleTo3Channel(),
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229]),
])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h),
                        (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=1)


def outputs_to_objects(outputs, img_size, id2label, threshold):
    img_w, img_h = img_size
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = m.indices.detach().cpu().numpy()[0]
    pred_scores = m.values.detach().cpu().numpy()[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    scaled = box_cxcywh_to_xyxy(pred_bboxes) * torch.tensor(
        [img_w, img_h, img_w, img_h], dtype=torch.float32
    )

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, scaled.tolist()):
        class_label = id2label[int(label)]
        if class_label != "no object" and score >= threshold:
            objects.append({
                "label": class_label,
                "score": float(score),
                "bbox": [round(v, 2) for v in bbox],
            })
    return objects


def parse_filename(fname):
    """Extract doc_id and page_num from '{doc_id}_page_{N}.png'."""
    stem = os.path.splitext(fname)[0]
    parts = stem.rsplit("_page_", 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return stem, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Table Transformer on PNGs and write detection results to CSV."
    )
    parser.add_argument("--png-dir", required=True, help="Directory of input PNGs")
    parser.add_argument("--output-csv", required=True, help="Path to write detection results")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Confidence threshold (default: 0.8)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Table Transformer model...")
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )
    model.to(device)
    model.eval()

    id2label = dict(model.config.id2label)
    id2label[len(id2label)] = "no object"

    png_files = sorted(f for f in os.listdir(args.png_dir) if f.endswith(".png"))
    print(f"Found {len(png_files)} PNGs")

    already_done = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv, newline="") as f:
            for row in csv.DictReader(f):
                already_done.add(row["filename"])
        print(f"Skipping {len(already_done)} already-processed files")

    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as out_f:
        writer = csv.writer(out_f)
        if write_header:
            writer.writerow(["filename", "doc_id", "page_num", "label", "score", "bbox"])

        for fname in png_files:
            if fname in already_done:
                continue
            path = os.path.join(args.png_dir, fname)
            doc_id, page_num = parse_filename(fname)

            image = Image.open(path).convert("RGB")
            tensor = detection_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)

            detections = outputs_to_objects(outputs, image.size, id2label, args.threshold)
            for det in detections:
                writer.writerow([fname, doc_id, page_num, det["label"], det["score"], det["bbox"]])

            status = f"{len(detections)} table(s)" if detections else "no tables"
            print(f"{fname}: {status}")
