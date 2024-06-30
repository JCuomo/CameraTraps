# -*- coding: utf-8 -*-
"""
# Fine tuning VIT freezing **weights**

trains a model given 2 json files:
- <image_name>_detections.json: containing the output of megadectector with bounding boxes
- <images_name>_labels.json: containing the image name and the corresponding label that should be unique to animal occurences (i.e. same for all bboxes)
"""

from collections import Counter
import logging
import time
import json
import numpy as np
import os
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from CameraTraps.classification.utils import preprocess_image


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_or_create_splits(splits_dir, detections_json=None, labels_json=None):
    """
    Get or create train/validation/test splits.

    Args:
        splits_dir (str): Directory to save/load splits.
        detections_json (str): Path to detections json file.
        labels_json (str): Path to labels json file.

    Returns:
        Tuple: train, validation, and test splits along with label dictionaries.
    """
    # Split files paths
    train_split_file = os.path.join(splits_dir, "train_split.json")
    val_split_file = os.path.join(splits_dir, "val_split.json")
    test_split_file = os.path.join(splits_dir, "test_split.json")

    def save_splits(filename, splits):
        with open(filename, 'w') as f:
            json.dump(splits, f)

    def load_splits(filename):
        with open(filename, 'r') as f:
            splits = json.load(f)
        return splits

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
        if not detections_json or not labels_json:
            logger.error("No splits found and no jsons specified")
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

                    if "failure" in image_data:
                        continue

                    for n, bbox in enumerate(image_data["detections"]):
                        if bbox["conf"] > confidence_threshold:
                            images_info.append((f"{image_data['file']}_{n}", bbox["bbox"]))
                            labels_info.append((f"{image_data['file']}_{n}", gt_labels[image_path]))
        
        logger.info(f"Dataset size: {len(images_info)}")
        logger.info(f"Labels count: {len(labels_info)}")

        logger.info("Creating new splits...")
        images_train, images_valtest, labels_train, labels_valtest = train_test_split(
            images_info, labels_info, test_size=0.2, random_state=42
        )
        images_val, images_test, labels_val, labels_test = train_test_split(
            images_valtest, labels_valtest, test_size=0.5, random_state=42
        )

        save_splits(train_split_file, {"images": images_train, "labels": labels_train})
        save_splits(val_split_file, {"images": images_val, "labels": labels_val})
        save_splits(test_split_file, {"images": images_test, "labels": labels_test})

    labels_names = sorted(list(set(label for _, label in labels_train + labels_val + labels_test)))
    label_dict = {class_name: label for label, class_name in enumerate(labels_names)}
    reversed_label_dict = {label_number: class_name for class_name, label_number in label_dict.items()}
    with open(os.path.join(splits_dir,'label_mapping.json'), 'w') as f:
        json.dump(reversed_label_dict, f)
    return images_train, labels_train, images_val, labels_val, images_test, labels_test, label_dict, reversed_label_dict




class TrainingDataset(Dataset):
    def __init__(self, images, labels, label_dict):
        self.images = images
        self.labels = labels
        self.label_dict = label_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path_idx, bbox = self.images[idx]
        image_name = image_path_idx[:image_path_idx.rfind("_")]
        try:
            cropped_image = preprocess_image(image_name, bbox)
        except Exception as e:
            logger.error(f"Error loading or cropping image: {image_name}, {e}")
            return None, None
        
        image_path_idx_label, label = self.labels[idx]
        if image_path_idx_label != image_path_idx:
            logger.debug("MISMATCH!!!!!", image_path_idx_label, image_path_idx, idx)

        return cropped_image, self.label_dict[label]




def create_training_dataloader(images, labels, label_dict, batch_size=32, shuffle=False):
    """
    Create a DataLoader.

    Args:
        images (list): List of image paths and bounding boxes.
        labels (list): List of labels.
        label_dict (dict): Dictionary mapping labels to numerical values.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader object.
    """
    dataset = TrainingDataset(images, labels, label_dict)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)

def augment_minority_classes(images, labels, label_dict, augmentations=None, augmentation_factor=2):
    """
    Augment minority classes in the dataset.

    Args:
        images (list): List of image paths and bounding boxes.
        labels (list): List of labels corresponding to the images.
        label_dict (dict): Dictionary mapping class names to numerical labels.
        augmentations (list): List of torchvision transforms for augmentation.
        augmentation_factor (int): Number of times to augment each minority class sample.

    Returns:
        Tuple: Augmented images and labels.
    """
    if augmentations is None:
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1)
        ]
    
    label_counts = Counter(label for _, label in labels)
    max_count = max(label_counts.values())

    augmented_images = images.copy()
    augmented_labels = labels.copy()

    for label, count in label_counts.items():
        if count < max_count:
            shortfall = max_count - count
            samples_needed = shortfall * augmentation_factor
            samples = [(img, lbl) for img, lbl in zip(images, labels) if lbl == label]
            for img, lbl in samples:
                for _ in range(samples_needed // len(samples)):
                    image_path_idx, bbox = img
                    image_name = image_path_idx[: image_path_idx.rfind("_")]
                    full_image = Image.open(image_name)
                    crop_image = get_crop(full_image, bbox, square_crop=True)

                    for augmentation in augmentations:
                        augmented_image = augmentation(crop_image)
                        augmented_image_path_idx = f"{image_name}_{len(augmented_images)}"
                        augmented_images.append((augmented_image_path_idx, bbox))
                        augmented_labels.append((augmented_image_path_idx, lbl))

    return augmented_images, augmented_labels


def get_optimizer(model):
    """
    Get the optimizer.

    Args:
        model (nn.Module): Model.

    Returns:
        torch.optim.Optimizer: Optimizer.
    """
    return torch.optim.Adam(model.classifier.parameters(), lr=0.001)


def train_model(model, train_dataloader, val_dataloader, optimizer, device, output_dir, num_epochs=20, checkpoint_interval=1):
    """
    Train the model.

    Args:
        model (nn.Module): Model.
        train_dataloader (DataLoader): Training DataLoader.
        val_dataloader (DataLoader): Validation DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        output_dir (str): Directory to save model checkpoints.
        num_epochs (int): Number of epochs.
        checkpoint_interval (int): Interval to save checkpoints.

    Returns:
        None
    """
    model.to(device)
    best_val_loss = float('inf')  
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
                logger.info(f"    Batch [{batch + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Time: {time.time()-start_time:.2f}s")
        if (epoch + 1) % checkpoint_interval == 0:
            val_loss: float = validate_model(model, val_dataloader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss    
                save_model(model, os.path.join(output_dir, f"best_model_e{epoch}.pth"))
            save_checkpoint(epoch, model, optimizer, loss, output_dir)


def save_model(model, path):
    """
    Save the model.

    Args:
        model (nn.Module): Model.
        path (str): Path to save the model.

    Returns:
        None
    """
    torch.save(model.state_dict(), path)

def save_checkpoint(epoch, model, optimizer, loss, output_dir):
    """
    Save model checkpoint.

    Args:
        epoch (int): Current epoch.
        model (nn.Module): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss (torch.Tensor): Loss.
        output_dir (str): Directory to save checkpoint.

    Returns:
        None
    """
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)


def validate_model(model, val_dataloader, device):
    """
    Validate the model.

    Args:
        model (nn.Module): Model.
        val_dataloader (DataLoader): Validation DataLoader.
        device (torch.device): Device to validate on.

    Returns:
        None
    """
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
    average_loss: float | torch.Tensor = val_loss / len(val_dataloader)
    logger.info(f"Validation Loss: {average_loss:.4f}, Validation Accuracy: {100 * correct / total:.2f}%")
    return average_loss


def make_predictions(model, dataloader, device):
    """
    Make predictions on the dataset.

    Args:
        model (nn.Module): Model.
        dataloader (DataLoader): DataLoader.
        device (torch.device): Device.

    Returns:
        Tuple: Predictions and labels.
    """
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


def plot_confusion_matrix(all_labels, all_predictions, output_dir):
    """
    Plot confusion matrix.

    Args:
        all_labels (np.ndarray): Ground truth labels.
        all_predictions (np.ndarray): Predicted labels.
        output_dir (str): Directory to save the plot.

    Returns:
        None
    """
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.show()
