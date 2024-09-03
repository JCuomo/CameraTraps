
import contextlib
import json
import logging
import os
import subprocess
import threading

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import concurrent.futures

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


def get_species(image_path, species, species_count, lock=None):
    # Use a no-op context manager if lock is not provided
    if lock is None:
        lock = contextlib.nullcontext()

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




def get_counts_dir(images_dir):
    """
    Check metadata of images recursively found in images_dir and returns a dict(image_path,count)
    """

    image_files = []
    output_dict ={}
    for root, dirs, files in os.walk(images_dir):
        for image_file in files:
            if image_file.lower().endswith(".jpg"):
                image_files.append(os.path.join(root, image_file))

    lock = threading.Lock()
    num_cores = os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_cores
    ) as executor:
        future_to_image_file = {
            executor.submit(get_counts, image_file): image_file
            for image_file in image_files
        }
        # concurrent.futures.wait(futures)
        for future in concurrent.futures.as_completed(future_to_image_file):
            image_file = future_to_image_file[future]
            try:
                species_count = future.result()
                output_dict[image_file] = sum(int(c) for c in species_count.values())
            except Exception as e:
                print(f"Exception for {image_file}: {e}")
        return output_dict

