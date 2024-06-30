
import logging

import os
from PIL import Image, ImageOps

from transformers import ViTForImageClassification
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import warnings

# Define a function to filter out the specific warning
def ignore_tensor_warning(message, category, filename, lineno, file=None, line=None):
    return "To copy construct from a tensor" not in str(message)

# Register the filter function to ignore the warning
warnings.showwarning = ignore_tensor_warning
# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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



class InferenceDataset(Dataset):

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
    if not batch:
        return {}
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

def create_inference_dataloader(images_dict):
    """
    Create a DataLoader.

    Args:
        images_dict (dict): Dictionary where keys are image paths and values are lists of bounding boxes.
        batch_size (int): Batch size.

    Returns:
        DataLoader: DataLoader object.
    """
    dataset = InferenceDataset(images_dict)
    # Note: batch_size=1 because batches are define within ImagesDataset as a batch corresponds to all the bboxes of 1 image
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_fn)

def classify_dataloader(model, dataloader, label_mapping, device):
    model.eval()
    output= {}
    with torch.no_grad():
        for batch in dataloader:
            if not batch: continue
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

