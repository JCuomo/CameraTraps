# -*- coding: utf-8 -*-
"""
# Fine tuning VIT freezing **weights**

trains a model given 2 json files:
- <image_name>_detections.json: containing the output of megadectector with bounding boxes
- <images_name>_labels.json: containing the image name and the corresponding label that should be unique to animal occurences (i.e. same for all bboxes)
"""

import json
import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from utils import initialize_model, preprocess_image

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class ImagesDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path_idx, bbox = self.images[idx]
        image_name = image_path_idx[:image_path_idx.rfind("_")]
        return preprocess_image(image_name, bbox)

def create_dataloader(images, batch_size=32):
    """
    Create a DataLoader.

    Args:
        images (list): List of image paths and bounding boxes.
        batch_size (int): Batch size.

    Returns:
        DataLoader: DataLoader object.
    """
    dataset = ImagesDataset(images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

def _classify_image(model, image):
    logits = model(image.unsqueeze(0)).logits
    probabilities = F.softmax(logits, dim=1)
    predicted_prob, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), predicted_prob.item()

def classify_dataloader(model, dataloader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            batch_pred = [_classify_image(model, image) for image in images]
            all_predictions.extend(batch_pred)
    return np.array(all_predictions)


if __name__ == "__main__":

    # Load model from checkpoint
    checkpoint_path = "/home/jcuomo/CameraTraps/output/classification/step3/checkpoint_epoch18.pth"
    model = initialize_model(checkpoint_path=checkpoint_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    bboxes_file = "/home/jcuomo/CameraTraps/output/detection/test_bboxes.json"
    # Load from the JSON file
    with open(bboxes_file, 'r') as file:
        images = json.load(file)

    # Create dataloader
    dataloader = create_dataloader(images, batch_size=128)
    # Make predictions and plot confusion matrix
    predictions = classify_dataloader(model, dataloader, device)
    print(predictions)
