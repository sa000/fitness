import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from natsort import natsorted
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from feature_generation import create_feature_image, create_features
from globals import RESOURCES_ROOT
from helpers.landmarks import landmarks_to_embedding
from helpers.plot_utils import create_plot, plot_confusion_matrix
import sys

def create_model(class_names: list, num_features: int):
    """
    Create a model with the following layers:
    1. Input layer
    2. Embedding layer
    3. Dense layer
    4. Dropout layer
    5. Dense layer
    6. Dropout layer
    7. Dense layer
    8. Output layer
    """
    inputs = tf.keras.Input(shape=(num_features))
    embedding = landmarks_to_embedding(inputs)

    layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)
    print(model.summary())
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def split_data(dataset_type: str):
    """
    Split the data into X & y data from the feature set csv
    """
    print("Splitting data into X & y")
    df = pd.read_csv(f"{RESOURCES_ROOT}/{excercise}/{dataset_type}_{excercise}.csv")
    df.drop(columns=["class_name"], inplace=True)
    y = df.pop("class_no")
    y = keras.utils.to_categorical(y)
    X = df.astype("float64")
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
):
    """
    Train the model and save the weights
    """
    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = "kb_weights.best.hdf5"
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
        batch_size=4,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, earlystopping],
    )
    return history


def predict_on_unseen_data(excercise: str):
    """
    Predict on unseen data

    args:
    excercise: str: the excercise to predict on
    unseen_folder: str: the folder containing the unseen data

    """
    # Load the best weights
    print('Predicting on unseen data for ', excercise)
    model = keras.models.load_model(f"{RESOURCES_ROOT}/{excercise}/{excercise}_model.h5")
    observations = []
    unseen_folder = f"images/video_frames/{excercise}"
    unseen_images = natsorted(os.listdir(unseen_folder))
    idx = 0
    class_names = ["start", "end"]

    for image_path in tqdm(unseen_images[0:300], desc="Predicting on unseen data"):
        row = []
        image = tf.io.read_file(f"{unseen_folder}/{image_path}")
        image = tf.io.decode_jpeg(image)
        features = create_feature_image(image, excercise)
        num_features = features.shape[1]

        features = np.array(features).reshape(1, num_features)
        prediction = model.predict(features)
        class_no = np.argmax(prediction)
        class_name = class_names[class_no]
        row = [image_path, prediction[0][0], prediction[0][1], class_name, class_no]
        observations.append(row)
        print (f"Predicted class: {class_name} ")
    df = pd.DataFrame(
        observations,
        columns=["image_path", "class_0", "class_1", "class_name", "class_no"],
    )
    df.to_csv(f"{excercise}_video_predictions.csv")
    return df

if __name__ == "__main__":
    try:
        excercise = sys.argv[1]
    except:
        excercise = "kb_situp"
    class_names = ["start", "end"]
    X, y = split_data("train")
    X_test, y_test = split_data("test")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    num_features = X_train.shape[1]
    print(f"Number of features: {num_features}")
    model = create_model(class_names, num_features)
    history = train_model(X_train, y_train, X_val, y_val)
    model.save(f"{RESOURCES_ROOT}/{excercise}/{excercise}_model.h5")
    # Prediction
    y_pred = model.predict(X_test)

    # Convert the prediction result to class name
    y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
    y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

    # Plotting
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    create_plot(excercise, history)
    plot_confusion_matrix(cm, class_names, excercise)
