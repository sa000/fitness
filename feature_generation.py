
from tqdm import tqdm
from helpers.helpers import  detect
import numpy as np 
import pandas as pd
from data import BodyPart
import sys
import os
import tensorflow as tf

LANDMARK_THRESHOLD = 0.3
import globals as g

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_headers_for_model() -> list:
    '''
    Get the headers for the Features file
    '''
    bodypart_cols = [[bodypart.name + '_x', bodypart.name + '_y', 
                  bodypart.name + '_score'] for bodypart in BodyPart] 
    headers = []
    for bodypart_col in bodypart_cols:
      headers += bodypart_col
    headers= headers + ['class_name', 'class_no']
    return headers
HEADERS = get_headers_for_model()

def create_features(excercise: str, dataset_type: str):
    """Creates a model for a given excercise.

    Args:
        excercise: The excercise to create a model for.

    Returns:
        A compiled model.
    """
    print(f"Creating feature set for {excercise} in the {dataset_type} dataset")
    
    images_in_train_folder = os.path.join(g.IMAGES_ROOT, excercise, dataset_type)
    image_folder = os.path.join(g.IMAGES_ROOT, excercise, dataset_type)
    class_names = os.listdir(image_folder)

    observations = []
    for (class_no, class_name) in enumerate(class_names):
            IMAGE_LIMIT = 2
            idx = 0  
            class_path = os.path.join(image_folder, class_name)
            image_files = os.listdir(class_path)
            target_info = [class_name, class_no]
            print(f"Processing {class_name} images for {excercise}")
            for image_file in tqdm(image_files, desc ="Processing images"):
                print(class_name, image_file)
                idx+=1
                if idx==IMAGE_LIMIT:
                    break
                image_path = os.path.join(class_path, image_file)
                image = tf.io.read_file(image_path)
                image = tf.io.decode_jpeg(image)
                image_height, image_width, channel = image.shape
                if channel==4:
                    image = image[::3]
                coordinates = get_keypoints(image)
                if coordinates == []:
                    continue
                observation = coordinates + target_info
                observations.append(observation)
            print(f"Processed {class_name} images for {excercise}")
    df = pd.DataFrame(observations, columns = HEADERS)
    df.to_csv(f'{g.RESOURCES_ROOT}/{excercise}/{dataset_type}_{excercise}.csv', index=False)
    print(f'Created {dataset_type}_{excercise}.csv')
    return df

def get_keypoints(image: np.ndarray) ->list:
    '''
    Get the keypoints from the image using movement
    return:
        coordinates: 1x51 array of keypoints. 17 keypoints x 3 (x,y,score)
    '''
    person = detect(image)
    min_landmark_score = min(
        [keypoint.score for keypoint in person.keypoints])
    if min_landmark_score <=LANDMARK_THRESHOLD:
        print(f"Skipping image because one of the landmarks has a score of {min_landmark_score}")
        return []
    pose_landmarks = np.array(
        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
            for keypoint in person.keypoints],
        dtype=np.float32)
    coordinates = pose_landmarks.flatten().tolist()
    return coordinates
if __name__ == "__main__":
    excercise = 'squat'
    print(excercise)
    create_features(excercise, 'train')
    create_features(excercise, 'test')
