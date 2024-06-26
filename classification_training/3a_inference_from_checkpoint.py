import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from transformers import ViTForImageClassification
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import os


test_split_file = "/home/jcuomo/CameraTraps/output/classification/step3/val_split.json"

# Function to load splits
def load_splits(filename):
    with open(filename, 'r') as f:
        splits = json.load(f)
    return splits

# Check if splits already exist
if os.path.exists(test_split_file):
    print("Loading existing splits...")
    test_split = load_splits(test_split_file)
    images_info = test_split["images"]
    labels_info = test_split["labels"]
else:
    # Paths to JSON files
    detectios_json = "/home/jcuomo/CamaraTrampa/training_data_Abril24/all_detections.json"
    labels_json = "/home/jcuomo/CamaraTrampa/training_data_Abril24/all_labels.json"

    # Confidence threshold
    confidence_threshold = 0.2

    # Load JSON data
    images_info = []
    labels_info = []
    with open(labels_json, "r") as labels_file:
        gt_labels = json.load(labels_file)
        with open(detectios_json, "r") as file:
            data = json.load(file)
            for image_data in data["images"]:
                image_path = image_data["file"]
                if "failure" in image_data:
                    continue
                n = 0
                for bbox in image_data["detections"]:
                    if bbox["category"] != "1":
                        continue
                    if bbox["conf"] > confidence_threshold:
                        images_info.append(
                            (f"{image_data['file']}_{n}", bbox["bbox"])
                        )
                        labels_info.append(
                            (f"{image_data['file']}_{n}", gt_labels[image_path])
                        )
print("Dataset size:", len(images_info))
print("Labels count:", len(labels_info))

# Function to get cropped image
def get_crop(img: Image.Image, bbox_norm, square_crop: bool):
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)

    if square_crop:
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    if box_w == 0 or box_h == 0:
        return None

    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if square_crop and (box_w != box_h):
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    return crop

# Create label dictionary
labels_names = set([animal for path, animal in labels_info])
label_dict = {class_name: label for label, class_name in enumerate(labels_names)}
reversed_label_dict = {label_number: class_name for class_name, label_number in label_dict.items()}
print(reversed_label_dict)
# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path_idx, bbox = self.images[idx]
        image_name = image_path_idx[: image_path_idx.rfind("_")]
        full_image = Image.open(image_name)
        crop_image = get_crop(img=full_image, bbox_norm=bbox, square_crop=True)

        image_path_idx_label, label = self.labels[idx]
        if image_path_idx_label != image_path_idx:
            print("MISMATCH!!!!!", image_path_idx_label, image_path_idx, idx)

        if self.transform:
            crop_image = self.transform(crop_image)

        return crop_image, label_dict[label]

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize the model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
num_features = model.config.hidden_size
num_classes = 7
model.classifier = torch.nn.Linear(num_features, num_classes)

# Load checkpoint if it exists
checkpoint_path = "/home/jcuomo/CameraTraps/output/classification/step3/checkpoint_epoch14.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# Create test dataset and dataloader
test_dataset = CustomDataset(images_info, labels_info, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inference
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        outputs = model(images)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.numpy()
        all_predictions.extend(predicted_class)
        all_labels.extend(labels)

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate accuracy
accuracy = (all_predictions == all_labels).mean()
print("Accuracy:", accuracy)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
