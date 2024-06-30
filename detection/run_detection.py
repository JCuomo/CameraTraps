import json
import logging
import os
import subprocess
import time

from utils import get_bboxes


if __name__ == "__main__":
    megadetector_script = "/home/jcuomo/CameraTraps/downloads/MegaDetector/megadetector/detection/run_detector_batch.py"
    model = "/home/jcuomo/CameraTraps/downloads/md_v5a.0.0.pt"
    images_dir = "/home/jcuomo/CameraTraps/images/unlabeled/test"
    output_base_dir = "/home/jcuomo/CameraTraps/output/detection"
    detections_file = os.path.join(output_base_dir, "test_detections.json")

    # Step1: run Megadector to detect animals
    subprocess.run(
        [
            "python",
            megadetector_script,
            model,
            images_dir,
            detections_file,
            "--include_exif_data",
        ]
    )

    # Step2: save bounding boxes of each animal per image
    bboxes_file = os.path.join(output_base_dir, "test_bboxes.json")
    bboxes = get_bboxes(detections_file)
    with open(bboxes_file, 'w') as file:
        json.dump(bboxes, file)

