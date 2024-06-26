# -*- coding: utf-8 -*-
"""
# Fine tuning VIT freezing **weights**

trains a model given 2 json files:
- <image_name>_detections.json: containing the output of megadectector with bounding boxes
- <images_name>_labels.json: containing the image name and the corresponding label that should be unique to animal occurences (i.e. same for all bboxes)
"""

import logging
import time
from sklearn.model_selection import train_test_split
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Sequence
from PIL import Image, ImageOps
from transformers import ViTForImageClassification
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def get_or_create_splits(splits_dir, detections_json=None, labels_json=None):
    """

    """
    # Split files paths
    train_split_file = os.path.join(splits_dir,"train_split.json")
    val_split_file = os.path.join(splits_dir,"val_split.json")
    test_split_file = os.path.join(splits_dir,"test_split.json")


    # Function to save splits
    def save_splits(filename, splits):
        with open(filename, 'w') as f:
            json.dump(splits, f)

    # Function to load splits
    def load_splits(filename):
        with open(filename, 'r') as f:
            splits = json.load(f)
        return splits

    # Check if splits already exist
    if all(os.path.exists(f) for f in [train_split_file, val_split_file, test_split_file]):
        logger.info("Loading existing splits...")
        train_split = load_splits(train_split_file)
        val_split = load_splits(val_split_file)
        test_split = load_splits(test_split_file)
        
        images_train = train_split["images"]
        labels_train = train_split["labels"]
        images_val = val_split["images"]
        labels_val = val_split["labels"]
        images_test = test_split["images"]
        labels_test = test_split["labels"]
    else:
        # File paths
        if not detections_json or not labels_json:
            logger.info("No splits found and no jsons specificed")
            return 

        confidence_threshold = 0.2
        images_info = []
        labels_info = []

        with open(labels_json, "r") as labels_file:
            gt_labels = json.load(labels_file)
            with open(detections_json, "r") as file:
                data = json.load(file)
                for image_data in data["images"]:
                    image_path = image_data["file"]

                    # Check if any bbox has confidence over the threshold
                    has_bbox_with_high_confidence = False
                    if "failure" in image_data:
                        continue
                    n = 0
                    for bbox in image_data["detections"]:
                        # if bbox["category"] != "1":
                        #     continue
                        if bbox["conf"] > confidence_threshold:
                            images_info.append(
                                (f"{image_data['file']}_{n}", bbox["bbox"])
                            )
                            labels_info.append(
                                (f"{image_data['file']}_{n}", gt_labels[image_path])
                            )
        logger.info("Dataset size:", len(images_info))
        logger.info("Labels count:", len(labels_info))

        logger.info("Creating new splits...")
        images_train, images_valtest, labels_train, labels_valtest = train_test_split(
            images_info, labels_info, test_size=0.2, random_state=42
        )
        images_val, images_test, labels_val, labels_test = train_test_split(
            images_valtest, labels_valtest, test_size=0.5, random_state=42
        )

        # Save splits to disk
        save_splits(train_split_file, {"images": images_train, "labels": labels_train})
        save_splits(val_split_file, {"images": images_val, "labels": labels_val})
        save_splits(test_split_file, {"images": images_test, "labels": labels_test})

    # Convert labels to numerical values
    # sorting the labels is critical for getting the same dict every time
    labels_names = sorted(list(set([animal for path, animal in labels_train + labels_val + labels_test])))
    label_dict = {
        class_name: label for label, class_name in enumerate(labels_names)
    }
    reversed_label_dict = {
        label_number: class_name for class_name, label_number in label_dict.items()
    }
    return images_train,labels_train,images_val,labels_val,images_test,labels_test,label_dict,reversed_label_dict


def get_crop(img: Image.Image, bbox_norm, square_crop: bool):
    """
    Crops an image
    Args:
        img: PIL.Image.Image object, already loaded
        bbox_norm: list or tuple of float, [xmin, ymin, width, height] all in
            normalized coordinates
        square_crop: bool, whether to crop bounding boxes as a square

    Returns: PIL.Image.Image object, cropped image
    """

    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)

    if square_crop:
        # Ensure the crop box is square and within image boundaries
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_size))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_size))
        box_w = box_size
        box_h = box_size

    if box_w == 0 or box_h == 0:
        return None

    # Image.crop() takes box=[left, upper, right, lower]
    crop = img.crop((xmin, ymin, xmin + box_w, ymin + box_h))

    if square_crop:
        # Pad to square using 0s if necessary
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    return crop

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, label_dict, transform=None):
        self.images = images
        self.labels = labels  # torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.label_dict = label_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path_idx, bbox = self.images[idx]
        image_name = image_path_idx[: image_path_idx.rfind("_")]
        try:
            full_image = Image.open(image_name)
            crop_image = get_crop(full_image, bbox, square_crop=True)
        except Exception as e:
            logger.debug(f"Error loading or cropping image: {image_name}, {e}")
            return None, None
        
        image_path_idx_label, label = self.labels[idx]
        if image_path_idx_label != image_path_idx:
            logger.debug("MISMATCH!!!!!", image_path_idx_label, image_path_idx, idx)

        if self.transform:
            crop_image = self.transform(crop_image)

        return crop_image, self.label_dict[label]


# Define your transform
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# Initialize the model
def initialize_model(label_dict):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.config.hidden_size
    num_classes = len(label_dict.keys())
    model.classifier = nn.Linear(num_features, num_classes)
    return model

# Create datasets and dataloaders
def create_dataloader(images, labels, label_dict, transform, batch_size=32, shuffle=False):
    dataset = CustomDataset(images, labels, transform=transform, label_dict=label_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    return dataloader

# Define your optimizer
def get_optimizer(model):
    return torch.optim.Adam(model.classifier.parameters(), lr=0.001)


# Training loop
def train_model(model, train_dataloader, val_dataloader, optimizer, device, output_dir, num_epochs=20, checkpoint_interval=1):
    model.to(device)
    model.classifier.to(device)
    start_time = time.time()
    for epoch in range(num_epochs):
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        model.train()
        total_loss = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch + 1) % 20 == 0:
                logger.info(f"    Batch [{batch + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Time: {time.time()-start_time}")
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(epoch, batch, model, optimizer, loss, output_dir)
        validate_model(model, val_dataloader, device)
    save_model(model, os.path.join(output_dir,"final_model.pth"))

def save_checkpoint(epoch, batch, model, optimizer, loss, output_dir):
    checkpoint_path = os.path.join(output_dir,f"checkpoint_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'step': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)

# Validation loop
def validate_model(model, val_dataloader, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f"Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

    val_predictions, val_labels = make_predictions(model, val_dataloader, device)
    accuracy = (val_predictions == val_labels).mean()
    logger.info(f"Accuracy:{accuracy}")
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Make predictions on the entire dataset
def make_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.numpy()
            all_predictions.extend(predicted_class)
            all_labels.extend(labels)
    return np.array(all_predictions), np.array(all_labels)

# Plot confusion matrix
def plot_confusion_matrix(all_labels, all_predictions, outputdir):
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(outputdir,"confusion_matrix.png"))
    plt.show()
