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

def create_features(excercise: str):
    """Creates a model for a given excercise.

    Args:
        excercise: The excercise to create a model for.

    Returns:
        A compiled model.
    """

    images_in_train_folder = os.path.join(IMAGES_ROOT, excercise, 'train')
    images_out_train_folder = 'poses_images_out_train'
    csvs_out_train_path = 'train_data.csv'
    preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_train_folder,
    images_out_folder=images_out_train_folder,
    csvs_out_path=csvs_out_train_path,
    )

    preprocessor.process(per_pose_class_limit=None)

IMAGES_ROOT = 'images/processed'  

create_features('squat')