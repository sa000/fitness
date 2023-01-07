import os
import sys
import configparser

from globals import POSTAUGMENTATION_PATH, RESOURCES_ROOT, BUCKET_NAME
import boto3

config = configparser.ConfigParser()
#print current directory
# config.read("./config.ini")
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
    dataset_types = ['train', 'test']
    for folder in ['processed', 'raw']:
        s3.put_object(Bucket=BUCKET_NAME, Key=f"images/{folder}/{excercise}/")
        if folder == 'raw':
            continue
        for dataset_type in dataset_types:
            s3.put_object(Bucket=BUCKET_NAME, Key=f"images/{folder}/{excercise}/{dataset_type}/")
            for position in positions:
                print(folder, dataset_type, position)
                s3.put_object(Bucket=BUCKET_NAME, Key=f"images/{folder}/{excercise}/{dataset_type}/{position}/")

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

if __name__ == "__main__":
    try:
        excercise = sys.argv[1]
    except:
        exercise='kb_around_the_world' #default
    initialize_s3_resources()
    initialize_excercise_folders_s3(excercise)
