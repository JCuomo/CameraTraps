


import os

import numpy as np
from utils import count_bboxes
from CameraTraps.metadate.utils import get_counts_dir
# RMSE
def rmse(counts1, counts2):
    differences = [(counts1[image] - counts2[image]) ** 2 for image in counts1 if image in counts2]
    return (sum(differences) / len(differences)) ** 0.5

# Correlation
def correlation(counts1, counts2):
    common_images = [image for image in counts1 if image in counts2]
    x = [counts1[image] for image in common_images]
    y = [counts2[image] for image in common_images]
    return np.corrcoef(x, y)[0, 1]

# Mean Absolute Difference
def mean_absolute_difference(counts1, counts2):
    differences = [abs(counts1[image] - counts2[image]) for image in counts1 if image in counts2]
    return sum(differences) / len(differences)

images_dir = "/home/jcuomo/images/CH08__P40406151__V2"
output_base_dir = "/home/jcuomo/GroundingDINO/output"
detections_file = os.path.join(output_base_dir, "detections.json")

n_counts = get_counts_dir(images_dir)
n_detections = count_bboxes(detections_file)

print("Correlation:", correlation(n_counts, n_detections))
print("Mean Absolute Difference:", mean_absolute_difference(n_counts, n_detections))
print("RMSE:", rmse(n_counts, n_detections))
