
from collections import Counter
import logging
import time
import json
import numpy as np
import os
from collections.abc import Sequence
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViTForImageClassification
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_bboxes(detections_json, output_file):
    confidence_threshold = 0.2
    output = {}
    with open(detections_json, "r") as file:
        data = json.load(file)
        for image_data in data["images"]:

            bboxes = []
            image_path = image_data["file"]

            if "failure" in image_data:
                continue

            for n, bbox in enumerate(image_data["detections"]):
                conf = bbox["conf"]
                if conf > confidence_threshold:
                    bboxes.append((bbox["bbox"],conf))
            output[image_path] = bboxes
    # Save to a JSON file
    with open(output_file, 'w') as file:
        json.dump(output, file)
    return output

def get_crop(img: Image.Image, bbox_norm, square_crop: bool):
    """
    Crops an image.

    Args:
        img (Image.Image): PIL Image object.
        bbox_norm (list/tuple): Normalized coordinates [xmin, ymin, width, height].
        square_crop (bool): Whether to crop bounding boxes as a square.

    Returns:
        Image.Image: Cropped image.
    """
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)

    if square_crop:
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_size))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_size))
        box_w = box_size
        box_h = box_size

    if box_w == 0 or box_h == 0:
        return None

    crop = img.crop((xmin, ymin, xmin + box_w, ymin + box_h))

    if square_crop:
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    return crop


def get_transform():
    """
    Get the transformation pipeline.

    Returns:
        transforms.Compose: Transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def preprocess_image(image_path, bbox):
    full_image = Image.open(image_path)
    crop_image = get_crop(full_image, bbox, square_crop=True)
    if crop_image is None:
        raise ValueError("Invalid crop dimensions")

    return get_transform()(crop_image)


def initialize_model(num_classes=None, checkpoint_path=None):
    """
    Initialize the Vision Transformer model.

    Args:
        label_dict (dict): Dictionary mapping labels to numerical values.

    Returns:
        ViTForImageClassification: Initialized model.
    """
    if checkpoint_path:
        # ensures that the checkpoint can be loaded regardless of whether it was saved on a GPU or CPU.
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) 
        model_state_dict = checkpoint.get('model_state_dict') or checkpoint
        num_classes_from_checkpoint = model_state_dict['classifier.weight'].size(0)
        if num_classes and num_classes!=num_classes_from_checkpoint:
            logger.warning("MISMATCH between num_classes and checkpoint")
        else:
            num_classes=num_classes_from_checkpoint
    if not num_classes and not checkpoint:
        raise ValueError("You must provide at least one of num_classes and/or checkpoint_path")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.config.hidden_size
    model.classifier = nn.Linear(num_features, num_classes)
    if checkpoint_path:
        model.load_state_dict(model_state_dict)
    return model