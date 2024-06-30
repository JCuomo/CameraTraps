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
import warnings

# Define a function to filter out the specific warning
def ignore_tensor_warning(message, category, filename, lineno, file=None, line=None):
    return "To copy construct from a tensor" not in str(message)

# Register the filter function to ignore the warning
warnings.showwarning = ignore_tensor_warning
# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImagesDataset(Dataset):

    def __init__(self, images_dict):
        """
        Input is a dictionary like: {image_path:[[bbox1,conf1],...,[bboxn,confn]]}
        e.g.:
        '/home/jcuomo/CameraTraps/images/unlabeled/test/CH09__P30410142__L1__Carrion__0329.JPG': [[[0.4996, 0.2906, 0.06953, 0.15], 0.934]],
        """
        self.image_list = list(images_dict.items())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path, bboxes_confs = self.image_list[idx]
        samples = []
        for bbox, conf in bboxes_confs:
            sample = {
                'image': preprocess_image(image_path, bbox),
                'image_path': image_path,
                'bbox': bbox,
                'bbox_confidence': conf
            }
            samples.append(sample)
        return samples
    
def collate_fn(batch):
    """
    Custom collate function to batch images and their corresponding bounding boxes.
    """
    batch = batch[0]
    images = [torch.tensor(sample['image']) for sample in batch]
    bboxes = [sample['bbox'] for sample in batch]
    image_path = [sample['image_path'] for sample in batch]
    confidences = [sample['bbox_confidence'] for sample in batch]
    
    return {
        'images': torch.stack(images),
        'bboxes': bboxes,
        'image_path': image_path,
        'bbox_confidence': confidences
    }

def create_dataloader(images_dict):
    """
    Create a DataLoader.

    Args:
        images_dict (dict): Dictionary where keys are image paths and values are lists of bounding boxes.
        batch_size (int): Batch size.

    Returns:
        DataLoader: DataLoader object.
    """
    dataset = ImagesDataset(images_dict)
    # Note: batch_size=1 because batches are define within ImagesDataset as a batch corresponds to all the bboxes of 1 image
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_fn)

def classify_dataloader(model, dataloader, label_mapping, device):
    model.eval()
    output= {}
    with torch.no_grad():
        for batch in dataloader:
            batch_output = []
            images = batch['images'].to(device)
            image_paths = batch['image_path'] # all items should be the same
            bboxes = batch['bboxes']
            bbox_confidences = batch['bbox_confidence']
            logits = model(images).logits
            probabilities = F.softmax(logits, dim=1)
            predicted_probs, predicted_classes = torch.max(probabilities, 1)
            for image_path, pred_class, pred_class_prob, bbox, bbox_prob in zip(image_paths, predicted_classes, predicted_probs, bboxes, bbox_confidences):
                batch_output.append({
                    'pred_class': label_mapping.get(pred_class.cpu().item()),
                    'pred_class_prob': pred_class_prob.cpu().item(),
                    'bbox': bbox,
                    'bbox_prob': bbox_prob
                })
            output[image_path]=batch_output
    return output


if __name__ == "__main__":

    # Load model from checkpoint
    checkpoint_path = "/home/jcuomo/CameraTraps/output/classification/step3/checkpoint_epoch18.pth"
    model = initialize_model(checkpoint_path=checkpoint_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    bboxes_file = "/home/jcuomo/CameraTraps/output/detection/test_bboxes.json"
    with open(bboxes_file, 'r') as file:
        images_bboxes = json.load(file)

    label_mapping_file = "/home/jcuomo/CameraTraps/output/classification/step3/label_mapping.json"
    with open(label_mapping_file, 'r') as file:
        label_mapping = json.load(file)
    # json dumps as string so we need to converted to integers
    label_mapping = {int(k): v for k, v in label_mapping.items()}

    # Create dataloader
    dataloader = create_dataloader(images_bboxes)
    # Make predictions and plot confusion matrix
    predictions = classify_dataloader(model, dataloader, label_mapping, device)
                                         
    print(predictions)

