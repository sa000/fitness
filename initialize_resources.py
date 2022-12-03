import os
from globals import RESOURCES_ROOT, POSTAUGMENTATION_PATH
import sys

def make_resources(excercise: str):
    """
    Create a folder for the excercise in the resources folder

    """
    resource_folder = os.path.join(RESOURCES_ROOT, excercise)
    make_folder(resource_folder)
    make_folder(os.path.join(resource_folder, "plots"))
    make_folder(os.path.join(RESOURCES_ROOT, "unseen"))

    make_folder(os.path.join(RESOURCES_ROOT, "unseen", excercise))

    return


# loop through folders in directory


def make_folder(path: str):
    """
    Create a folder if it doesn't exist
    """
    if not os.path.exists(path):
        print(f"Creating folder for {path}")
        os.makedirs(path)
    return


def initialize_postaugmentation_folders(excercise: str):
    """
    For an excercise , create a folder for each dataset type and position if it doesn't exist

    """
    excercise_path = os.path.join(POSTAUGMENTATION_PATH, excercise)
    make_folder(excercise_path)
    for dataset in ["train", "test"]:
        dataset_path = os.path.join(excercise_path, dataset)
        make_folder(dataset_path)
        for position in ["start", "end"]:
            position_path = os.path.join(dataset_path, position)
            make_folder(position_path)


if __name__ == "__main__":
    try:
        excercise = sys.argv[1]
    except:
        exercise='squat' #default
    make_resources(excercise)
    initialize_postaugmentation_folders(excercise)
