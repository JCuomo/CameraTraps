
import json


def get_bboxes(detections_json):
    confidence_threshold = 0.2
    output = {}
    with open(detections_json, "r") as file:
        data = json.load(file)
        for image_data in data["images"]:

            bboxes = []
            image_path = image_data["file"]

            if "failure" in image_data:
                continue

            for n, bbox in enumerate(image_data["detections"]):
                conf = bbox["conf"]
                if conf > confidence_threshold:
                    bboxes.append((bbox["bbox"],conf))
            output[image_path] = bboxes
    return output

def get_bboxes_thr(detections_json):
    output = {}
    with open(detections_json, "r") as file:
        data = json.load(file)
        for image_data in data["images"]:

            bboxes = []
            image_path = image_data["file"]

            if "failure" in image_data:
                continue

            for n, bbox in enumerate(image_data["detections"]):
                bboxes.append((bbox["bbox"],bbox["conf"]))
            output[image_path] = bboxes
    return output

def count_detections(detections_json):
    confidence_threshold = 0.2
    output = {}
    with open(detections_json, "r") as file:
        data = json.load(file)
        for image_data in data["images"]:
            count = 0
            image_path = image_data["file"]

            if "failure" in image_data:
                continue

            for n, bbox in enumerate(image_data["detections"]):
                conf = bbox["conf"]
                if conf > confidence_threshold:
                    count += 1
            output[image_path] = count
    return output
