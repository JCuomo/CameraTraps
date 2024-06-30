
# Define function to load the model from checkpoint
import logging
import os

import torch

from classification.utils import initialize_model
from classification_training.training_utils import create_training_dataloader, get_or_create_splits, get_transform, make_predictions, plot_confusion_matrix
from transformers import ViTForImageClassification


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    # Paths and parameters
    output_dir = "/home/jcuomo/CameraTraps/output/classification/step3"
    checkpoint_path = os.path.join(output_dir, "checkpoint_epoch0.pth")
    splits_dir = output_dir

    # Load splits
    _, _, images_val, labels_val, images_test, labels_test, label_dict, _ = get_or_create_splits(splits_dir)

    # Load model from checkpoint
    num_classes = len(label_dict.keys())
    model = initialize_model(num_classes,checkpoint_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create test dataloader
    transform = get_transform()
    test_dataloader = create_training_dataloader(images_test, labels_test, label_dict, transform, batch_size=128, shuffle=False)

    # Make predictions and plot confusion matrix
    test_predictions, test_labels = make_predictions(model, test_dataloader, device)
    accuracy = (test_predictions == test_labels).mean()
    logger.info(f"Accuracy: {accuracy:.4f}")
    plot_confusion_matrix(test_labels, test_predictions, output_dir)