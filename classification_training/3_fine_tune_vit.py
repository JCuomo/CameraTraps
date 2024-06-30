# -*- coding: utf-8 -*-
"""
# Fine tuning VIT freezing **weights**

trains a model given 2 json files:
- <image_name>_detections.json: containing the output of megadectector with bounding boxes
- <images_name>_labels.json: containing the image name and the corresponding label that should be unique to animal occurences (i.e. same for all bboxes)
"""

import logging


from CameraTraps.classification.utils import get_transform,initialize_model
from CameraTraps.classification_training.training_utils import augment_minority_classes, create_dataloader, get_optimizer, get_or_create_splits,  make_predictions, plot_confusion_matrix, train_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # step1: create the train|val|test splits
    output_dir = "/home/jcuomo/CameraTraps/output/classification/step3"
    detections_json = "/home/jcuomo/CameraTraps/output/classification/step2/all_detections.json"
    labels_json = "/home/jcuomo/CameraTraps/output/classification/step2/all_labels.json"
    images_train,labels_train,images_val,labels_val,images_test,labels_test,label_dict,reversed_label_dict = get_or_create_splits(output_dir, detections_json=detections_json, labels_json=labels_json)

    logger.info(f"Labels:{label_dict}")
    
    # step2: finetune the model
    batch_size = 128
    num_epochs = 30

    checkpoint_interval = 1
    transform = get_transform()
    model = initialize_model(len(label_dict))
    device = "cuda:0"
    optimizer = get_optimizer(model)

    images_train, labels_train = augment_minority_classes(images_train, labels_train, label_dict)
    train_dataloader = create_dataloader(images_train, labels_train, label_dict, transform, batch_size=batch_size, shuffle=True)
    val_dataloader = create_dataloader(images_val, labels_val, label_dict, transform, batch_size=batch_size, shuffle=False)

    train_model(model, train_dataloader, val_dataloader, optimizer, device, output_dir,num_epochs=num_epochs, checkpoint_interval=checkpoint_interval)

    # step3: check the model
    test_dataloader = create_dataloader(images_test, labels_test, label_dict, transform, batch_size=batch_size, shuffle=False)
    test_predictions, test_labels= make_predictions(model, test_dataloader, device)
    accuracy = (test_predictions == test_labels).mean()
    logger.info(f"Accuracy:{accuracy}")
    plot_confusion_matrix(test_labels, test_predictions, output_dir)
