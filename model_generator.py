import os
import sys

import keras
from tensorflow.keras.utils import load_img, img_to_array  
import io 
import numpy as np
import pandas as pd
import tensorflow as tf
from natsort import natsorted
from io import StringIO
from feature_generation import get_columns_to_drop
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from feature_generation import create_feature_image,get_headers_for_model
from globals import RESOURCES_ROOT
from helpers.landmarks import landmarks_to_embedding
from helpers.plot_utils import create_plot, plot_confusion_matrix

from globals import POSTAUGMENTATION_PATH, RAW_IMAGES, BUCKET_NAME
import boto3 
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

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

    layer = keras.layers.Dense(1000, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(512, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)
    print(model.summary())
    # define the callback
    opt = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]    )

    return model


def split_data(dataset_type: str):
    """
    Split the data into X & y data from the feature set csv
    """
    print("Splitting data into X & y")
    path = os.path.join(RESOURCES_ROOT, excercise, f"{dataset_type}_{excercise}.csv")
    s3_object = s3_client.get_object(Bucket=BUCKET_NAME, Key=path)
    csv_string = s3_object['Body'].read().decode('utf-8')

    # Convert the CSV string to a DataFrame
    df = pd.read_csv(StringIO(csv_string))
    columns = get_columns_to_drop(excercise)+['class_name']
    df.drop(columns=columns, inplace=True)
    y = df.pop("class_no")
    y = keras.utils.to_categorical(y)
    X = df.astype("float64")
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    excercise: str
):
    """
    Train the model and save the weights
    """
    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = f"{excercise}_weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    earlystopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=25)

    # Start training
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=10,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, earlystopping],
    )
    
    return history


def predict_on_unseen_data(excercise: str, video_file: str):
    """
    Predict on unseen data

    args:
    excercise: str: the excercise to predict on
    unseen_folder: str: the folder containing the unseen data

    """
    # Load the best weights
    print('Predicting on unseen data for ', excercise)
    model = keras.models.load_model(f"models/{excercise}_model.h5")
    observations = []
    video_label = video_file.strip('.mp4')
    unseen_folder = f"images/video_frames/{excercise}/{video_label}/unlabeled/"
    objects = s3_client.list_objects(Bucket=BUCKET_NAME, Prefix=unseen_folder)['Contents']
    #load the images from s3 bucket
    idx = 0
    class_names = ["start", "end"]
    columns_to_drop = get_columns_to_drop(excercise)
    HEADERS = get_headers_for_model()[:-2]
    indices_to_keep = [i for i, x in enumerate(HEADERS) if x not in columns_to_drop]

    for object in tqdm(objects, desc="Predicting on unseen data"):
        image_path = object['Key']
        if image_path[-1] == '/':
            continue
        body = s3_resource.Object(BUCKET_NAME, image_path).get()['Body'].read()
        image = tf.io.decode_jpeg(body)
        features = create_feature_image(image)
        features = [x for i, x in enumerate(features) if i in indices_to_keep]
        num_features = len(features)


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
    df.to_csv(f"{video_label}.csv")
    #move file to s3
    s3_client.upload_file(f"{video_label}.csv",
     BUCKET_NAME, 
     f"resources/{excercise}/{video_label}/predictions.csv")
    os.remove(f"{video_label}.csv")
    return df

if __name__ == "__main__":
    try:
        excercise = sys.argv[1]
    except:
        excercise = "kb_around_the_world"
    class_names = ["start", "end"]
    X, y = split_data("train")
    X_test, y_test = split_data("test")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    num_features = X_train.shape[1]
    print(f"Number of features: {num_features}")
    model = create_model(class_names, num_features)
    history = train_model(X_train, y_train, X_val, y_val, excercise)
    model.save(f"models/{excercise}_model.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(f"models/{excercise}_model.tflite", 'wb') as f:
        f.write(tflite_model)

    #upload the file to s3
    s3_client.upload_file(f"models/{excercise}_model.h5", BUCKET_NAME, f"models/{excercise}_model.h5")
    s3_client.upload_file(f"models/{excercise}_model.tflite", BUCKET_NAME, f"models/{excercise}_model.tflite")

    # Prediction
    y_pred = model.predict(X_test)

    # Convert the prediction result to class name
    y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
    y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    create_plot(excercise, history)
    plot_confusion_matrix(cm, class_names, excercise)
    os.remove(f'{excercise}_weights.best.hdf5')
    os.remove(f'{excercise}_cm.png')
    os.remove(f'{excercise}_model_accuracy.png')
