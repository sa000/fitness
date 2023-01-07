import os
import random
import uuid
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.utils import (img_to_array, load_img,  # type: ignore
                                    save_img)
from tqdm import tqdm

from globals import POSTAUGMENTATION_PATH, RAW_IMAGES, BUCKET_NAME
import boto3 
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

def get_augmented_images(img: tf.Tensor):
    """
    Augment an image through a series of transformations

    args:
    img: image to augment
    """
    augmented_images = []
    flipped = tf.image.flip_left_right(img_to_array(img))
    augmented_images = augmented_images + [flipped]
    return augmented_images


def save_image_s3(image: tf.Tensor, excercise: str, class_name: str):
    """
    Save images to the appropriate folder

    args:

    images: image to save
    excercise: the excercise the images are for
    class_name: the class the images belong to

    """
    TRAINING_SIZE = .75
    dataset_type = "train" if random.random() < TRAINING_SIZE else "test"
    uuid_name = str(uuid.uuid4())
    image_name = f"{uuid_name}.png"
    image_path = os.path.join(
        POSTAUGMENTATION_PATH, excercise, dataset_type, class_name, image_name
    )
    save_img(image_path, image)


def perform_augmentation(excercise: str):
    """
    Perform augmentation on all images in the raw images folder

    args:
        excercise: the excercise to perform augmentation on
    """
    print(f"Performing augmentation for {excercise}")
    for position in ["start", "end"]:
        image_file_path = os.path.join(RAW_IMAGES, excercise, position)
        objects = s3_client.list_objects(Bucket=BUCKET_NAME, Prefix=image_file_path)['Contents']
        for object in objects[0:2]:
            image_path = object['Key']
            if image_path[-1] == "/":
                continue
            body = s3_resource.Object(BUCKET_NAME, image_path).get()['Body'].read()
            img = load_img(BytesIO(body))
            print(f"Augmenting {image_path} for {excercise} {position}")
            augmented_images = get_augmented_images(img)
            for image in [img, *augmented_images]:
                save_image_s3(image, excercise, position)


if __name__ == "__main__":
    import sys
    try:
        excercise = "kb_around_the_world"
    except:
        raise Exception("Please provide an excercise name")
    perform_augmentation(excercise)
