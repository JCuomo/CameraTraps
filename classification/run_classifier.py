# -*- coding: utf-8 -*-
"""
# Fine tuning VIT freezing **weights**

trains a model given 2 json files:
- <image_name>_detections.json: containing the output of megadectector with bounding boxes
- <images_name>_labels.json: containing the image name and the corresponding label that should be unique to animal occurences (i.e. same for all bboxes)
"""


import json
from detection.utils import get_bboxes_thr
import torch
from classification.utils import classify_dataloader, create_inference_dataloader, initialize_model


if __name__ == "__main__":

    # Load model from checkpoint: fine-tune required
    checkpoint_path = "/home/jcuomo/CameraTraps/output/classification/step3/final_model.pth"
    model = initialize_model(checkpoint_path=checkpoint_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # label_mapping created during fine-tuning
    label_mapping_file = "/home/jcuomo/CameraTraps/output/classification/step3/label_mapping.json"
    with open(label_mapping_file, 'r') as file:
        label_mapping = json.load(file)
    label_mapping = {int(k): v for k, v in label_mapping.items()}

    # Load bboxes: detections required
    bboxes_file = "/home/jcuomo/CameraTraps/output/detection_training/detections_all.json"
    with open(bboxes_file, 'r') as file:
        images_bboxes = json.load(file)
    # images_bboxes = get_bboxes_thr(bboxes_file)
        

    # Run classifier
    dataloader = create_inference_dataloader(images_bboxes)
    predictions = classify_dataloader(model, dataloader, label_mapping, device)
    predictions_file = "/home/jcuomo/CameraTraps/output/classification/classifications.json"
    with open(predictions_file, 'w') as file:
        json.dump(predictions, file)
    print(predictions)

