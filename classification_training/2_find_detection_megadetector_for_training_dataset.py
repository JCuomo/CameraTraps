import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


def run_megadetector_on_dir(dir, megadetector_script, model):
    """
    Run megadector on all "_images.json" in "dir" and create the corresponding "_detections.json" files
    """
    for file in os.listdir(dir):
        if not file.endswith("_images.json"):
            continue
        run_megadetector_on_file(file, megadetector_script, model)


def run_megadetector_on_file(file, megadetector_script, model):
    """
    Run megadector on a "_images.json" file and create the corresponding "_detections.json" file
    """
    output_file_path = file.replace('images.json', '_detections.json')
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
    megadetector_script = "/home/jcuomo/CamaraTrampa/MegaDetector/megadetector/detection/run_detector_batch.py"
    model = "md_v5a.0.0.pt"
    images_json_dir = "/home/jcuomo/CamaraTrampa/step1_output"
    #run_megadetector_on_dir(images_json_dir, megadetector_script, model)
    run_megadetector_on_file("/home/jcuomo/CamaraTrampa/step1_output/CH07__P70510151__V1_images.json", megadetector_script, model)
    merge_detections(images_json_dir)
