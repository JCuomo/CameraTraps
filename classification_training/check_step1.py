import os

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

def get_folder_names_from_directories(directory):
    """
    Recursively extracts folder names from the directory tree.
    """
    folder_names = set()
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            folder_names.add(dir_name)
    return folder_names

def compare_folder_names(dir_with_files, dir_with_folders):
    """
    Compares folder names derived from files in one directory with actual folder names in another directory.
    """
    folder_names_from_files = get_folder_names_from_files(dir_with_files)
    folder_names_from_directories = get_folder_names_from_directories(dir_with_folders)

    if folder_names_from_files == folder_names_from_directories:
        print("The folder names match.")
    else:
        print("The folder names do not match.")
        print("Folders in dir_with_files not in dir_with_folders:", folder_names_from_files - folder_names_from_directories)
        print("Folders in dir_with_folders not in dir_with_files:", folder_names_from_directories - folder_names_from_files)

# Example usage:
dir_with_files = '/home/jcuomo/CameraTraps/output/classification/step1'
dir_with_folders = '/home/jcuomo/CameraTraps/images/labeled'


compare_folder_names(dir_with_files, dir_with_folders)
