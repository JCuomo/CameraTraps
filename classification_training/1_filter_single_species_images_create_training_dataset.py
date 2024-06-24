"""1. filter single species images - create training_dataset.ipynb

Given a labeled (metadata) images directory, it will output two json files:
- <image_name>_images.json: containing a list of images with only 1 species
- <image_name>_labels.json: containing a list of images and the corresponding specie

it takes ~0.15s per image
"""

import os
import json
import subprocess
import logging
import threading
import time
import numpy as np
import concurrent.futures

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def read_metadata(image_path):
    command = ["exiftool", "-j", image_path]
    output = subprocess.check_output(command, universal_newlines=True)
    metadata = json.loads(output)[0]
    return metadata


def log_failed_image(image_path: str, reason: str, file):
    logger.debug(f"image:{image_path} || {reason}")
    if file:
        error_data = {"image_path": image_path, "reason": reason}
        json.dump(error_data, file)
        file.write("\n")

def get_species(image_path, species, species_count, lock):
    metadata = read_metadata(image_path)
    if isinstance(metadata.get("HierarchicalSubject"), list):
        for item in metadata["HierarchicalSubject"]:
            if "|" not in item:
                continue
            tag, value = item.split("|")
            with lock:
                if tag.startswith("A__Species"):
                    species.add(value)
                if tag.startswith("B__No"):
                    species_count.add(tag)


def list_species_in_images_threads(images_dir, output_dir):
    """
    Check metadata of images recursively found in images_dir and writes:
     - species.txt with a list of unique species in A__Species
     - species_count.txt with a list of unique species in B__No
    """
    species = set()  # from A__Species|...
    species_count = set()  # from B_No|...
    output_file_species = os.path.join(output_dir, "species.txt")
    output_file_species_count = os.path.join(output_dir, "species_count.txt")
    image_files = []
    start_time = time.time()
    for root, dirs, files in os.walk(images_dir):
        for image_file in files:
            if image_file.lower().endswith(".jpg"):
                image_files.append(os.path.join(root, image_file))

    lock = threading.Lock()
    num_cores = os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_cores
    ) as executor:
        futures = [
            executor.submit(
                get_species, image_file, species, species_count, lock
            )
            for image_file in image_files
        ]
        concurrent.futures.wait(futures)

    with open(output_file_species, "w") as species_file:
        json.dump(list(species), species_file)

    with open(output_file_species_count, "w") as species_count_file:
        json.dump(list(species_count), species_count_file)
    logger.info(f"Time to complete:{time.time()-start_time}")
    print(f"Time to complete:{time.time()-start_time}")


def get_counts(image_path, failed_file=None, indet_file=None):
    """
    Returns:
        a dictionary with the tagged species:count in an image.
        an empty dictionary if the metadata is malformed/wrong.
    """
    item = None
    try:
        metadata = read_metadata(image_path)
        species_count = {}  # from B__No...|N
        species = []  # from A__Species|...
        items = metadata.get("HierarchicalSubject")
        if isinstance(items, list):
            for item in metadata["HierarchicalSubject"]:
                if item.startswith("A__By"):
                    continue
                if "|" not in item:
                    log_failed_image(
                        image_path,
                        f"Tag {item} doesn't have '|'",
                        failed_file,
                    )
                    return {}  # discard malformed metadata
                tag, value = item.split("|")
                if tag.startswith("A__Species"):
                    species.append(value)
                    if "indet" in value.lower():
                        log_failed_image(
                            image_path,
                            f"Specie: {value}",
                            indet_file,
                        )
                        return {}
                if tag.startswith("B__No"):
                    species_count[tag] = value
    except Exception as e:
        reason = item or e
        log_failed_image(
            image_path,
            f"Unkwonk failure:{reason}",
            failed_file,
        )
        return {}
    # verify mismatch between metadata
    if len(species) != len(species_count.keys()):
        # after initial check, made a secondary check for specific species than don't require count specification
        species_exceptions: list[str] = [
            "Lycalopex culpaeus",
            "Puma concolor",
            "Personnel",
            "Lycalopex griseus",
            "Cathartes aura",
            "Phalcoboenus megalopterus"
            "Vultur gryphus"
            "Geranoaetus polyosoma"
            "Lepus europaeus"
        ]
        for specie in species_exceptions:
            if specie in species and len(species)==1 and not species_count:
                species_count["B__No"+specie] = 1
        if len(species) != len(species_count.keys()):
            log_failed_image(
                image_path,
                f"Species identified:{species} vs Species counted:{species_count.keys()}",
                failed_file,
            )
            return {}

    return species_count

def get_folder_names_from_files(directory):
    """
    Extracts folder names from file names in the given directory.
    Assumes files are named like <folder_name>_images.json and <folder_name>_labels.json.
    """
    folder_names = set()
    for file_name in os.listdir(directory):
        if file_name.endswith('_images.json') or file_name.endswith('_labels.json'):
            folder_name = file_name[:-12]
            folder_names.add(folder_name)
    return folder_names

def filter_single_species_images(images_dir, output_dir):
    """
    Filter images with only 1 species (any number of individuals of the same species)
    """
    time_per_image = []
    if not os.path.exists(images_dir):
        logger.error(f"Directory {images_dir} doesn't exist")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    already_processed = get_folder_names_from_files(output_dir)
    malformed_metadata_path = os.path.join(output_dir, "malformed_metadata.json")
    indet_metadata_path = os.path.join(output_dir, "indet_metadata.json")
    n_images = 0
    with open(malformed_metadata_path, "a") as malformed_metadata_file, \
         open(indet_metadata_path, "a") as indet_metadata_file:

        for root, _, files in os.walk(images_dir):
            if os.path.basename(root) in already_processed: 
                logger.info(f"Skipping {root}, already processed")
                continue
            useful_training_image = []
            labels = {}
            start_root_path = os.path.split(root)[-1]

            for file in files:
                if not file.lower().endswith(".jpg"):
                    continue
                n_images += 1
                start_time = time.time()
                full_image_path = os.path.join(root, file)
                counts = get_counts(full_image_path, malformed_metadata_file, indet_metadata_file)

                if not counts or len(counts) > 1:
                    logger.debug(f"Discard {file}: {counts}")
                    continue

                useful_training_image.append(full_image_path)
                labels[full_image_path] = list(counts.keys())[0].split("No")[1]
                logger.debug(f"Accept {file}: {counts}")
                time_per_image.append(time.time() - start_time)

            if useful_training_image:
                output_file_images = os.path.join(output_dir, f"{start_root_path}_images.json")
                output_file_labels = os.path.join(output_dir, f"{start_root_path}_labels.json")

                with open(output_file_images, "w") as fw_images:
                    json.dump(useful_training_image, fw_images)

                with open(output_file_labels, "w") as fw_labels:
                    json.dump(labels, fw_labels)

            logger.info(f"Processed folder {root}: {len(files)} files")

    if time_per_image:
        time_per_image = np.array(time_per_image)
        mean_time = time_per_image.mean()
        std_time = time_per_image.std()
        logger.debug(f"Average time taken to process an image: {mean_time} +- {std_time}")
        print(f"Average time taken to process an image: {mean_time} +- {std_time}")

    logger.info(f"Processed {n_images} images")


def merge_labels(labels_dir):
    """
    Merge labels files into a single files.

    :param labels_dir: directory where 1 o more *_labels.json files are

    Returns:
        creates a single json file with all the *_labels.json files content merged.
    """
    merged_labels = {}
    for filename in os.listdir(labels_dir):
        if filename.endswith("_labels.json"):
            with open(os.path.join(labels_dir, filename), "r") as file:
                labels_data = json.load(file)
                merged_labels.update(labels_data)
    # Save the merged detections to a new JSON file
    output_file = os.path.join(labels_dir, "all_labels.json")
    with open(output_file, "w") as file:
        json.dump(merged_labels, file, indent=4)


if __name__ == "__main__":
    images_dir = "/home/jcuomo/CameraTraps/images/labeled"
    output_dir = "/home/jcuomo/CameraTraps/output/classification/step1_output"
    # Step 1: find what species are in the photos and 
    # manually make sure there is consistency in the naming
    # estimated time in 12 core: 0.025sec per image
    # list_species_in_images_threads(images_dir, output_dir)

    # Step 2: filter images with only one specie
    # st = time.time()
    # filter_single_species_images(images_dir, output_dir)
    # print("single thread:",time.time()-st)

    # Step 3: merge *_labels.json to all_labels.json
    merge_labels(output_dir)
  