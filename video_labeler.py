# import keras

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from helpers.helpers import draw_prediction_on_image, detect


def load_video_into_cv2(excercise):
    """
    Load video to cv2 and same 30 frames per second to folder

    """
    video = cv2.VideoCapture("tony_squats.mp4")
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    count = 0
    while video.isOpened() and count < 100:
        success, frame = video.read()
        print(count)
        if success:
            cv2.imwrite(
                f"images/processed/{excercise}/unseen/unknown/{count}.jpg", frame
            )
            count += 1
        else:
            break


excercise = "squat"

load_video_into_cv2("squat")
