import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from io import BytesIO
from data import BodyPart
from helpers.helpers import detect
from tensorflow.keras.utils import img_to_array, load_img, save_img  # type: ignore

LANDMARK_THRESHOLD = 0.00001
from globals import POSTAUGMENTATION_PATH, RAW_IMAGES, BUCKET_NAME

import globals as g
import boto3

s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_headers_for_model() -> list:
    """
    Get the headers for the Features file
    """
    bodypart_cols = [
        [bodypart.name + "_x", bodypart.name + "_y", bodypart.name + "_score"]
        for bodypart in BodyPart
    ]
    headers = []
    for bodypart_col in bodypart_cols:
        headers += bodypart_col
    headers = headers + ["class_name", "class_no"]
    return headers


HEADERS = get_headers_for_model()


def get_columns_to_drop(excercise: str):
    """
    Get the columns to drop from the dataframe. Aim to help model learn faster with only relevant features
    """
    not_revelant_body_parts = []
    if excercise == "kb_around_the_world":
        not_revelant_body_parts = ["KNEE", "ANKLE"]
    if excercise == "kb_situp":
        not_revelant_body_parts = ["KNEE", "EYE", "ANKLE"]
    # for each not relevant body part, remove the LEFT, RIGHT and SCORE columns from the BODYPART columns
    columns_to_drop = []
    for body_part in not_revelant_body_parts:
        for column in [body_part + "_x", body_part + "_y", body_part + "_score"]:
            if body_part == "NOSE":
                columns_to_drop.append(column)
            else:
                columns_to_drop.append("LEFT_" + column)
                columns_to_drop.append("RIGHT_" + column)
    return columns_to_drop


def create_feature_image(image: np.ndarray):
    """Creates a feature image for a given image.

    Args:
        image: The image to create a feature image for.
    Returns:
        A feature image.
    """
    image_height, image_width, channel = image.shape
    if channel == 4:
        image = image[::3]
    coordinates = get_keypoints(image)
    if coordinates == []:
        return []
    return coordinates


def generate_feature_file(excercise: str):
    """
     Generates a csv file with the features for the given excercise.

    Args:
        excercise: The excercise to create a model for.
    """

    class_names = ["start", "end"]
    resource_path = os.path.join(g.RESOURCES_ROOT, excercise)
    for dataset_type in ["train", "test"]:
        print(f"Generating feature file for {excercise} {dataset_type} dataset")
        path = os.path.join(g.POSTAUGMENTATION_PATH, excercise, dataset_type)

        objects = s3_client.list_objects(Bucket=BUCKET_NAME, Prefix=path)["Contents"]
        observations, debugging_observations = [], []
        for object in tqdm(
            objects[0:100], desc=f"Generating features for {excercise} {dataset_type}"
        ):
            if object["Key"][-1] == "/":
                continue
            _, _, _, dataset_type, class_name, file_path = object["Key"].split("/")
            class_no = class_names.index(class_name)
            target_info = [class_name, class_no]
            body = s3_resource.Object(BUCKET_NAME, object["Key"]).get()["Body"].read()
            image = tf.image.decode_image(body)
            coordinates = create_feature_image(image)
            observation = coordinates + target_info
            observations.append(observation)
        df = pd.DataFrame(observations, columns=HEADERS)
        # columns_to_drop = get_columns_to_drop(excercise)
        # df.drop(columns=columns_to_drop, inplace=True, axis=1)
        # df.to_csv(
        #     f"{g.RESOURCES_ROOT}/{excercise}/{dataset_type}_{excercise}.csv",
        #     index=False,
        # )
        csv_string = df.to_csv(index=False)
        # write the dataframe to s3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=f"{resource_path}/{dataset_type}_{excercise}.csv",
            Body=csv_string,
        )
        print(f"Saved features for {excercise} {dataset_type} to s3")


def get_keypoints(image: np.ndarray) -> list:
    """
    Get the keypoints from the image using movement
    return:
        coordinates: 1x51 array of keypoints. 17 keypoints x 3 (x,y,score)
    """
    person = detect(image)

    # Go through each tuple in person.keypoints. if there is a score less than .1, use the RIGHT or LEFT version of the body part
    # if there is no RIGHT or LEFT version, then skip the image
    for keypoint in person.keypoints:
        x_coord, y_coord, score = (
            keypoint.coordinate.x,
            keypoint.coordinate.y,
            keypoint.score,
        )

    min_landmark_score = min([keypoint.score for keypoint in person.keypoints])

    if min_landmark_score <= LANDMARK_THRESHOLD:
        print(
            f"Skipping image because one of the landmarks has a score of {min_landmark_score}"
        )
        return []
    pose_landmarks = np.array(
        [
            [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
            for keypoint in person.keypoints
        ],
        dtype=np.float32,
    )
    coordinates = pose_landmarks.flatten().tolist()
    return coordinates


if __name__ == "__main__":
    import sys

    try:
        excercise = sys.argv[1]
    except:
        excercise = "kb_around_the_world"
    print(excercise)
    generate_feature_file(excercise)
