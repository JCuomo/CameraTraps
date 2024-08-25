import json
import logging
import os
import subprocess
import time



if __name__ == "__main__":
    megadetector_script = "/home/jcuomo/CameraTraps/downloads/MegaDetector/megadetector/detection/run_detector_batch.py"
    model = "/home/jcuomo/CameraTraps/downloads/md_v5a.0.0.pt"
    images_dir = "/home/jcuomo/images/CH08__P40406151__V2"
    output_base_dir = "/home/jcuomo/CameraTraps/output/detection_training"
    detections_file = os.path.join(output_base_dir, "detections.json")

    # Step1: run Megadector to detect animals
    subprocess.run(
        [
            "python",
            megadetector_script,
            model,
            images_dir,
            detections_file,
            "--recursive",
            "--include_exif_data",
        ]
    )
