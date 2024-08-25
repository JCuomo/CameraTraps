

import os
import threading

from CameraTraps.detection.utils import count_detections
from CameraTraps.metadate.utils import get_counts_dir

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Example counts from two algorithms
counts1 = {'image1': 5, 'image2': 3, 'image3': 8}
counts2 = {'image1': 6, 'image2': 2, 'image3': 7}

# Correlation
def correlation(counts1, counts2):
    common_images = [image for image in counts1 if image in counts2]
    x = [counts1[image] for image in common_images]
    y = [counts2[image] for image in common_images]
    return np.corrcoef(x, y)[0, 1]

# Bland-Altman Plot
def bland_altman_plot(counts1, counts2):
    common_images = [image for image in counts1 if image in counts2]
    x = [counts1[image] for image in common_images]
    y = [counts2[image] for image in common_images]
    mean_counts = [(a + b) / 2 for a, b in zip(x, y)]
    diff_counts = [a - b for a, b in zip(x, y)]
    mean_diff = np.mean(diff_counts)
    std_diff = np.std(diff_counts)
    
    plt.scatter(mean_counts, diff_counts)
    plt.axhline(mean_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--')
    plt.xlabel('Mean of Counts')
    plt.ylabel('Difference of Counts')
    plt.title('Bland-Altman Plot')
    plt.show()
    plt.savefig("altman.png", dpi=300)


# Mean Absolute Difference
def mean_absolute_difference(counts1, counts2):
    differences = [abs(counts1[image] - counts2[image]) for image in counts1 if image in counts2]
    return sum(differences) / len(differences)

# RMSE
def rmse(counts1, counts2):
    differences = [(counts1[image] - counts2[image]) ** 2 for image in counts1 if image in counts2]
    return (sum(differences) / len(differences)) ** 0.5

# Scatter Plot
def scatter_plot(counts1, counts2):
    common_images = [image for image in counts1 if image in counts2]
    x = [counts1[image] for image in common_images]
    y = [counts2[image] for image in common_images]
    plt.scatter(x, y)
    plt.xlabel('Algorithm 1 Counts')
    plt.ylabel('Algorithm 2 Counts')
    plt.title('Scatter Plot of Algorithm Counts')
    plt.show()
    plt.savefig("scatter.png", dpi=300)


if __name__ == "__main__":
    images_dir = "/home/jcuomo/images/CH08__P40406151__V2"
    detections_file = "/home/jcuomo/CameraTraps/output/detection_training/detections.json"
    
    n_detections = count_detections(detections_file)
    n_counts = get_counts_dir(images_dir)
    # Example usage
    print("Correlation:", correlation(n_counts, n_detections))
    bland_altman_plot(n_counts, n_detections)
    print("Mean Absolute Difference:", mean_absolute_difference(n_counts, n_detections))
    print("RMSE:", rmse(n_counts, n_detections))
    scatter_plot(n_counts, n_detections)        