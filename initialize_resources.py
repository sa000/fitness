import os
import sys
import configparser

from globals import POSTAUGMENTATION_PATH, RESOURCES_ROOT, BUCKET_NAME
import boto3

config = configparser.ConfigParser()
config.read("config.ini")
s3 = boto3.client("s3")


def initialize_s3_resources():
    """
    Create a bucket called fitness with the following structure
    fitness
        images:
            processed:
            raw:
            video_frames:
        resources:
    """
    # create the bucket if not already exist
    folder_structure = {
        "images":  ['processed', 'raw', 'video_frames'],
        "resources": [],
        'models': []
    }
    response = s3.create_bucket(
        Bucket=BUCKET_NAME
    )
    #create an empty image folder
    for folder in folder_structure.keys():
        s3.put_object(Bucket=BUCKET_NAME, Key=f"{folder}/")
        for sub_folder in folder_structure[folder]:
            s3.put_object(Bucket=BUCKET_NAME, Key=f"{folder}/{sub_folder}/")

def initialize_excercise_folders_s3(excercise: str):
    '''
    Create a folder for the excercise in the resources folder in s3
    '''
    for folder in ['plots']:
        s3.put_object(Bucket=BUCKET_NAME, Key=f"resources/{excercise}/{folder}/")
    positions = ['start', 'end']
    for folder in ['processed', 'raw']:
        s3.put_object(Bucket=BUCKET_NAME, Key=f"images/{folder}/{excercise}/")
        for position in positions:
            s3.put_object(Bucket=BUCKET_NAME, Key=f"images/{folder}/{excercise}/{position}/")

    print(f'Created folders for excercise in s3 for {excercise}')
def empty_s3_bucket():
    '''
    Empty and delete all buckets in s3
    '''
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(BUCKET_NAME)
    bucket.objects.all().delete()
    bucket.delete()
# empty_s3_bucket()
initialize_s3_resources()
# initialize_excercise_folders_s3('bicep_curl')
# def make_resources(excercise: str):
#     """
#     Create a folder for the excercise in the resources folder

#     """
#     resource_folder = os.path.join(RESOURCES_ROOT, excercise)
#     make_folder(resource_folder)
#     make_folder(os.path.join(resource_folder, "plots"))


#     return


# loop through folders in directory


# def make_folder(path: str):
#     """
#     Create a folder if it doesn't exist
#     """
#     if not os.path.exists(path):
#         print(f"Creating folder for {path}")
#         os.makedirs(path)
#     return


# def initialize_postaugmentation_folders(excercise: str):
#     """
#     For an excercise , create a folder for each dataset type and position if it doesn't exist

#     """
#     excercise_path = os.path.join(POSTAUGMENTATION_PATH, excercise)
#     make_folder(excercise_path)
#     for dataset in ["train", "test"]:
#         dataset_path = os.path.join(excercise_path, dataset)
#         make_folder(dataset_path)
#         for position in ["start", "end"]:
#             position_path = os.path.join(dataset_path, position)
#             make_folder(position_path)


# if __name__ == "__main__":
#     try:
#         excercise = sys.argv[1]
#     except:
#         exercise='squat' #default
#     make_resources(excercise)
#     initialize_postaugmentation_folders(excercise)
