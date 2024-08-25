import json

def count_bboxes(detections_file):
    """
    Returns a dictionary mapping image paths to the number of bounding boxes detected.
    
    Args:
    detections_file (str): Path to the JSON file containing detections.

    Returns:
    dict: A dictionary where the keys are image paths and the values are the number of bounding boxes.
    """
    with open(detections_file, "r") as f:
        detections = json.load(f)
    
    bbox_counts = {}
    for detection in detections:
        image_path = detection["image_path"]
        num_bboxes = len(detection["boxes"])
        bbox_counts[image_path] = num_bboxes
    
    return bbox_counts
