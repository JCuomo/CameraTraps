import logging
import time

from detection.megadetector_utils import merge_detections, run_megadetector_on_dir

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    megadetector_script = "/home/jcuomo/CameraTraps/downloads/MegaDetector/megadetector/detection/run_detector_batch.py"
    model = "/home/jcuomo/CameraTraps/downloads/md_v5a.0.0.pt"
    images_json_dir = "/home/jcuomo/CameraTraps/output/classification/step1_output"
    output_dir = "/home/jcuomo/CameraTraps/output/classification/step2"
    st: float = time.time()
    run_megadetector_on_dir(images_json_dir, megadetector_script, model, output_dir)
    print("Megadetector:",time.time()-st)
    # js="/home/jcuomo/CameraTraps/output/classification/step1_output/CAR__090113__01__Carrion_images.json"
    # run_megadetector_on_file(js, megadetector_script, model)
    merge_detections(images_json_dir)
