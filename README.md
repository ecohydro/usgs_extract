# usgs_extract

# Object Detection Pipeline

## Functions

### `MaxResize`

A custom image transformation class that resizes an image such that its largest dimension does not exceed a specified maximum size, preserving the aspect ratio.

**Attributes**:
- `max_size` (int): The maximum allowed size for the largest dimension of the image.

**Methods**:
- `__init__(self, max_size=800)`: Initializes the transformation with a given maximum size.
- `__call__(self, image)`: Resizes the input image to ensure its largest dimension does not exceed `max_size`.

**Usage**:
```python
resize_transform = MaxResize(800)
resized_image = resize_transform(image)
```

### `detection_transform`

A composed transformation that applies `MaxResize`, converts the image to a tensor, and normalizes it using predefined mean and standard deviation values.

**Composition**:
- `MaxResize(800)`
- `transforms.ToTensor()`
- `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

**Usage**:
```python
detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### `box_cxcywh_to_xyxy(x)`

Converts bounding boxes from center x, center y, width, height format to x_min, y_min, x_max, y_max format.

**Parameters**:
- `x` (torch.Tensor): Tensor containing bounding boxes in (cx, cy, w, h) format.

**Returns**:
- `torch.Tensor`: Tensor containing bounding boxes in (x_min, y_min, x_max, y_max) format.

**Usage**:
```python
bbox_xyxy = box_cxcywh_to_xyxy(bbox_cxcywh)
```

### `rescale_bboxes(out_bbox, size)`

Rescales bounding boxes to the original image size.

**Parameters**:
- `out_bbox` (torch.Tensor): Tensor containing bounding boxes in (cx, cy, w, h) format.
- `size` (tuple): The original size of the image as (width, height).

**Returns**:
- `torch.Tensor`: Tensor containing rescaled bounding boxes in (x_min, y_min, x_max, y_max) format.

**Usage**:
```python
rescaled_bboxes = rescale_bboxes(bboxes, image.size)
```

### `outputs_to_objects(outputs, img_size, id2label)`

Processes the model outputs to extract detected objects, their labels, scores, and bounding boxes.

**Parameters**:
- `outputs` (dict): The output from the object detection model.
- `img_size` (tuple): The original size of the image as (width, height).
- `id2label` (dict): Dictionary mapping label ids to label names.

**Returns**:
- `list`: A list of dictionaries, each containing the label, score, and bounding box of a detected object.

**Usage**:
```python
objects = outputs_to_objects(outputs, image.size, id2label)
```

### `runModel(files, directory_path)`

Processes a list of image files to detect objects and saves the results in a CSV file.

**Parameters**:
- `files` (list): A list of file paths to the images.
- `directory_path` (str): The directory path to save the results CSV file.

**Returns**:
- `pd.DataFrame`: A DataFrame containing the file paths and the detected objects with their scores.

**Usage**:
```python
files = ["path/to/image1.jpg", "path/to/image2.jpg"]
results = runModel(files, "path/to/save/results")
print(results)
```

---

This README provides a detailed guide to understanding and utilizing the functions in the script for object detection and processing.
