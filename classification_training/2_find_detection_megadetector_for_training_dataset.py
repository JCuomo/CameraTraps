import json
import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


def run_megadetector_on_dir(dir, megadetector_script, model, output_dir):
    """
    Run megadector on all "_images.json" in "dir" and create the corresponding "_detections.json" files
    """
    for file in os.listdir(dir):
        if not file.endswith("_images.json"):
            continue
        run_megadetector_on_file(os.path.join(dir,file), megadetector_script, model, output_dir)


def run_megadetector_on_file(file, megadetector_script, model, output_dir):
    """
    Run megadector on a "_images.json" file and create the corresponding "_detections.json" file
    """
    output_file_path = os.path.join(output_dir,os.path.basename(file.replace('images.json', '_detections.json'))) 
    logger.debug(file, output_file_path)
    subprocess.run(
        [
            "python",
            megadetector_script,
            model,
            file,
            output_file_path,
            "--include_exif_data",
        ]
    )


def merge_detections(dir):
    """
    Merge "_detections.json" into "all_detections.json"
    """
    merged_detections = {"images": []}
    for filename in os.listdir(dir):
        if filename.endswith("_detections.json"):
            logger.debug("Merging", filename)
            with open(os.path.join(dir, filename), "r") as file:
                detections_data = json.load(file)
                merged_detections["images"].extend(detections_data["images"])
    # Save the merged detections to a new JSON file
    output_file = os.path.join(dir, "all_detections.json")
    with open(output_file, "w") as file:
        json.dump(merged_detections, file, indent=4)


if __name__ == "__main__":
    megadetector_script = "/home/jcuomo/CameraTraps/downloads/MegaDetector/megadetector/detection/run_detector_batch.py"
    model = "/home/jcuomo/CameraTraps/downloads/md_v5a.0.0.pt"
    images_json_dir = "/home/jcuomo/CameraTraps/output/classification/step1_output"
    output_dir = "/home/jcuomo/CameraTraps/output/classification/step2"
    st = time.time()
    run_megadetector_on_dir(images_json_dir, megadetector_script, model, output_dir)
    print("Megadetector:",time.time()-st)
    # js="/home/jcuomo/CameraTraps/output/classification/step1_output/CAR__090113__01__Carrion_images.json"
    # run_megadetector_on_file(js, megadetector_script, model)
    merge_detections(images_json_dir)
