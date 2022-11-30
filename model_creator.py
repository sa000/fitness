import csv
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from natsort import natsorted
import cv2
import itertools
import numpy as np
import pandas as pd
from matplotlib import font_manager
import imageio
from helpers.helpers import draw_prediction_on_image, detect
import imageio
from tqdm import tqdm

import os
import sys
import tempfile
from PIL import Image

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from helpers.landmarks import load_pose_landmarks, landmarks_to_embedding
import io
from PIL import Image, ImageDraw, ImageSequence, ImageFont
from data import BodyPart

class_names = ["start", "end"]
LANDMARK_THRESHOLD = 0.3
RESOURCES_ROOT = "resources"
IMAGES_ROOT = "images/processed"


def make_resource(excercise: str):
    """
    Create a folder for the excercise in the resources folder

    """
    resource_folder = os.path.join(RESOURCES_ROOT, excercise)
    if not os.path.exists(resource_folder):
        os.makedirs(resource_folder)
    return


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


def create_features(excercise: str, dataset_type: str):
    """Creates a model for a given excercise.

    Args:
        excercise: The excercise to create a model for.

    Returns:
        A compiled model.
    """
    print(f"Creating feature set for {excercise} in the {dataset_type} dataset")

    images_in_train_folder = os.path.join(IMAGES_ROOT, excercise, dataset_type)
    image_folder = os.path.join(IMAGES_ROOT, excercise, dataset_type)
    class_names = os.listdir(image_folder)

    observations = []

    for (class_no, class_name) in enumerate(class_names):
        IMAGE_LIMIT = 2
        idx = 0
        class_path = os.path.join(image_folder, class_name)
        image_files = os.listdir(class_path)
        target_info = [class_name, class_no]
        print(f"Processing {class_name} images for {excercise}")
        for image_file in tqdm(image_files, desc="Processing images"):
            print(class_name, image_file)
            idx += 1
            if idx == IMAGE_LIMIT:
                break
            image_path = os.path.join(class_path, image_file)
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image)
            image_height, image_width, channel = image.shape
            if channel == 4:
                image = image[::3]
            coordinates = get_keypoints(image)
            if coordinates == []:
                continue
            observation = coordinates + target_info
            observations.append(observation)
        print(f"")

    df = pd.DataFrame(observations, columns=HEADERS)
    df.to_csv(
        f"{RESOURCES_ROOT}/{excercise}/{dataset_type}_{excercise}.csv", index=False
    )
    print(f"Created {dataset_type}_{excercise}.csv")
    return df


def get_keypoints(image: np.ndarray) -> list:
    """
    Get the keypoints from the image using movement
    return:
        coordinates: 1x51 array of keypoints. 17 keypoints x 3 (x,y,score)
    """
    person = detect(image)
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


def create_model(class_names: list):
    inputs = tf.keras.Input(shape=(51))
    embedding = landmarks_to_embedding(inputs)

    layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)
    model.summary()
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def run_model(X_train, y_train, X_val, y_val):
    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    earlystopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20)

    # Start training
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, earlystopping],
    )
    return history


def create_plot(excercise: str, history):
    # Visualize the training history to see whether you're overfitting.
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["TRAIN", "VAL"], loc="lower right")
    plt.show()
    # Save the matplot figure
    plt.savefig(f"{excercise}_model_accuracy.png")
    plt.close()


def plot_confusion_matrix(
    cm,
    classes,
    excercise,
    dataset_type,
    normalize=False,
    cmap=plt.cm.Blues,
):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    title = f"Confusion matrix for {excercise} {dataset_type}"
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{excercise}_{dataset_type}_cm.png")
    plt.close()


if True:
    excercise = "squat"
    make_resource(excercise)
    train_df = create_features("squat", "train")


def split_data(df):
    df.drop(columns=["class_name"], inplace=True)
    y = df.pop("class_no")
    y = keras.utils.to_categorical(y)
    X = df.astype("float64")
    return X, y


mode = None
if mode == "create_model":
    print("Creating datasets")

    train_df = create_features("squat", "train")
    test_df = create_features("squat", "test")

    # csv_file = f'train_{excercise}.csv'
    # X, y, class_names, _ = load_pose_landmarks(csv_file)
    X, y = split_data(train_df)
    X_test, y_test = split_data(test_df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # csv_file = f'test_{excercise}.csv'
    # X_test, y_test, _, df_test = load_pose_landmarks(csv_file)

    model = create_model(class_names)

    history = run_model(X_train, y_train, X_val, y_val)
    model.save(f"{excercise}_model.h5")
    create_plot(excercise, history)
    y_pred = model.predict(X_test)

    # Convert the prediction result to cla#ss name
    y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
    y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

    # Plot the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plot_confusion_matrix(cm, class_names, excercise, "test")


def generated_graded_video(predictions):
    """
    Generates a graded video
    """
    # create font to be bla
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 14)
    # make the font smaller

    images = natsorted(os.listdir("tony"))
    # Loop over each frame in the animated image
    index = 0
    frames = []
    start = None
    import PIL

    duration = 20
    excericse_count = 0
    prev_state, current_state = "start", "start"
    print(duration)
    excercise_complete, just_finished = False, False
    transition_state = False
    on_new_rep = False
    excercise_count = 0
    for image_path in images:
        print(image_path)
        # add text to the image in the rop right coner saying the class name
        img = Image.open("tony/" + image_path)
        height, width = img.size
        img = img.resize((height // 3, width // 3))
        prediction = predictions[
            predictions.image_path == image_path
        ].class_name.values[0]
        current_state = prediction
        max_prob = np.max(
            predictions[predictions.image_path == image_path][
                ["class_0", "class_1"]
            ].values
        )
        max_prob = float(int(max_prob * 1000)) / 1000
        print(current_state, prev_state, transition_state)
        if (current_state != prev_state or transition_state) and current_state == "end":
            # Changed states, are we finishing the rep?
            transition_state = True
            if max_prob >= 0.95:
                excercise_count += 1
                transition_state = False

        value_text = f"Prediction: {prediction}: Probability  {max_prob} , \nCurrent Excercise Count{excercise_count}"
        print(value_text)
        draw = ImageDraw.Draw(img)
        draw.text((img.width - 350, 0), value_text, (255, 0, 0), font=font)
        frames.append(img)
        index += 1
        prev_state = current_state
    frames[0].save(
        f"optimized_{duration}.gif",
        format="GIF",
        save_all=True,
        duration=duration,
        append_images=frames[1:],
        quality=40,
        optimize=True,
    )
    duration = 20
    # print(duration)
    # frames[0].save(f'optimized_{duration}.gif', format='GIF', save_all=True, duration=duration, append_images=frames[1:], quality=40, optimize=True)


if mode == "run_on_new_data":
    # Applying it on new data
    model = keras.models.load_model(f"{excercise}_model.h5")
    print("APPLYING ON NEW DATA")
    observations = []
    idx = 0
    for image_path in os.listdir("tony"):
        print(idx)
        row = []
        image = tf.io.read_file(f"tony/{image_path}")
        image = tf.io.decode_jpeg(image)
        features = generate_features(image)
        features = np.array(features).reshape(1, 51)
        prediction = model.predict(features)
        class_no = np.argmax(prediction)
        class_name = class_names[class_no]
        row = [image_path, prediction[0][0], prediction[0][1], class_name, class_no]
        observations.append(row)
        idx += 1

    df = pd.DataFrame(
        observations,
        columns=["image_path", "class_0", "class_1", "class_name", "class_no"],
    )
    df.to_csv("tony_predictions.csv")


if mode == "generate_video":
    predictions = pd.read_csv("tony_predictions.csv")
    generated_graded_video(predictions)
