import os
from tensorflow.keras.utils import img_to_array, save_img, load_img  # type: ignore
import tensorflow as tf
import random
import uuid
from globals import RAW_IMAGES, POSTAUGMENTATION_PATH


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


def save_images(images: list, excercise: str, class_name: str):
    """
    Save images to the appropriate folder

    args:

    images: list of images to save
    excercise: the excercise the images are for
    class_name: the class the images belong to

    """
    dataset_type = "train" if random.random() < 0.8 else "test"
    for image in images:
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
        image_files = os.listdir(image_file_path)
        for image_file in image_files:
            print(f"Augmenting {image_file} for {excercise} {position}")
            image_path = os.path.join(image_file_path, image_file)
            img = load_img(image_path)
            processed_images = get_augmented_images(img)
            save_images(processed_images, excercise, position)


if __name__ == "__main__":
    excercise = "squat"
    perform_augmentation(excercise)
