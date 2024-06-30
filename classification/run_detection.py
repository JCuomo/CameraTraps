import json
import logging
import os
import subprocess
import time

from utils import get_bboxes


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    megadetector_script = "/home/jcuomo/CameraTraps/downloads/MegaDetector/megadetector/detection/run_detector_batch.py"
    model = "/home/jcuomo/CameraTraps/downloads/md_v5a.0.0.pt"
    images_dir = "/home/jcuomo/CameraTraps/images/unlabeled/test"
    output_base_dir = "/home/jcuomo/CameraTraps/output/detection"
    detections_file = os.path.join(output_base_dir, "test_detections.json")

    # subprocess.run(
    #     [
    #         "python",
    #         megadetector_script,
    #         model,
    #         images_dir,
    #         detections_file,
    #         "--include_exif_data",
    #     ]
    # )
    bboxes_file = os.path.join(output_base_dir, "test_bboxes.json")

    get_bboxes(detections_file, bboxes_file)



    # st: float = time.time()
    # run_megadetector_on_dir(images_json_dir, megadetector_script, model, output_dir)
    # print("Megadetector:",time.time()-st)
    # # js="/home/jcuomo/CameraTraps/output/classification/step1_output/CAR__090113__01__Carrion_images.json"
    # # run_megadetector_on_file(js, megadetector_script, model)
    # merge_detections(images_json_dir)
