
# Define function to load the model from checkpoint
import logging
import os

import torch

from CameraTraps.classification_training.fine_tuning_vit import create_dataloader, get_or_create_splits, get_transform, make_predictions, plot_confusion_matrix
from transformers import ViTForImageClassification


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_model_from_checkpoint(checkpoint_path, num_classes):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    num_features = model.config.hidden_size
    model.classifier = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    return model




if __name__ == "__main__":
    # Paths and parameters
    output_dir = "/home/jcuomo/CameraTraps/output/classification/step3"
    checkpoint_path = os.path.join(output_dir, "checkpoint_epoch0.pth")
    splits_dir = output_dir

    # Load splits
    _, _, images_val, labels_val, images_test, labels_test, label_dict, _ = get_or_create_splits(splits_dir)

    # Load model from checkpoint
    num_classes = len(label_dict.keys())
    model = load_model_from_checkpoint(checkpoint_path, num_classes)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create test dataloader
    transform = get_transform()
    test_dataloader = create_dataloader(images_test, labels_test, label_dict, transform, batch_size=128, shuffle=False)

    # Make predictions and plot confusion matrix
    test_predictions, test_labels = make_predictions(model, test_dataloader, device)
    accuracy = (test_predictions == test_labels).mean()
    logger.info(f"Accuracy: {accuracy:.4f}")
    plot_confusion_matrix(test_labels, test_predictions, output_dir)