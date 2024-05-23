# AUTOGENERATED! DO NOT EDIT! File to edit: ../04_model.ipynb.

# %% auto 0
__all__ = ['model']

# %% ../04_model.ipynb 3
from transformers import AutoModelForObjectDetection
import torch


# %% ../04_model.ipynb 6
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")

# %% ../04_model.ipynb 7
## put functions and classes here
