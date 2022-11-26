import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm
from PIL import Image

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from helpers.movenet_processor import MoveNetPreprocessor 
from helpers.landmarks import load_pose_landmarks, landmarks_to_embedding
def create_features(excercise: str, dataset_type: str):
    """Creates a model for a given excercise.

    Args:
        excercise: The excercise to create a model for.

    Returns:
        A compiled model.
    """

    images_in_train_folder = os.path.join(IMAGES_ROOT, excercise, dataset_type)
    images_out_train_folder = f'poses_images_out_{dataset_type}_{excercise}'
    csvs_out_train_path = f'{dataset_type}_{excercise}.csv'
    preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_train_folder,
    images_out_folder=images_out_train_folder,
    csvs_out_path=csvs_out_train_path,
    )

    preprocessor.process(per_pose_class_limit=None)
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
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model 
def run_model(X_train, y_train, X_val, y_val):
    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                patience=20)

    # Start training
    history = model.fit(X_train, y_train,
                        epochs=200,
                        batch_size=16,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, earlystopping])
    return history 

def create_plot(excercise:str, history ):
    # Visualize the training history to see whether you're overfitting.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['TRAIN', 'VAL'], loc='lower right')
    plt.show()
    #Save the matplot figure
    plt.savefig(f'{excercise}_model_accuracy.png')
    plt.close()    

def plot_confusion_matrix(cm, classes,
                            excercise,          
                          normalize=False,
                          cmap=plt.cm.Blues,
                          ):
  """Plots the confusion matrix."""
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
  title = f'Confusion matrix for {excercise}'
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=55)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plt.savefig(f'{excercise}_cm.png')
  plt.close()  
IMAGES_ROOT = 'images/processed'  
excercise = 'squat'
# create_features('squat', 'train')
# create_features('squat', 'test')
# create_features('squat', 'practice')


csv_file = f'train_{excercise}.csv'
X, y, class_names, _ = load_pose_landmarks(csv_file)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                test_size=0.15)
csv_file = f'test_{excercise}.csv'                                          
X_test, y_test, _, df_test = load_pose_landmarks(csv_file)

model = create_model(class_names)
history = run_model(X_train, y_train, X_val, y_val)
create_plot(excercise, history)
y_pred = model.predict(X_test)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plot_confusion_matrix(cm,
                      class_names,
                        excercise)


#Applying it on new data