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


# excercise = "squat"

# load_video_into_cv2("squat")

def load_video_for_process(video_file: str, frame_starting: int, frame_interval: int, total_frames: int):
    """
    Load video to cv2 and save every n frame to folder, starting off with a shift  of frame_starting

    """
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    frame_count = 0 
    print(f'Processing video to save for {video_file}')
    output_dir = '/Users/sakib/Desktop/raw'
    num_saved = 0 
    while video.isOpened():
        success, frame = video.read()
        
        if success and (frame_count % frame_interval == 0 ):
            if frame_count > frame_starting:
                print('Saving frame', frame_count)
                num_saved+=1
                cv2.imwrite(
                    f"{output_dir}/{frame_count}.jpg", frame
                )
        if num_saved>total_frames:
            break
        frame_count += 1
video_file ='/Users/sakib/Downloads/10000000_5593358860777945_7796839231421277800_n.mp4'
load_video_for_process(video_file, frame_starting=296, frame_interval=25, total_frames=50)

