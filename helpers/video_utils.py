import os

import cv2
import numpy as np
from matplotlib import font_manager
from natsort import natsorted
from PIL import Image, ImageFont
from tqdm import tqdm
import pandas as pd
from initialize_resources import make_folder
from model_generator import predict_on_unseen_data

project_root = os.path.dirname(os.path.dirname(__file__))

def load_video_into_cv2(
    excercise,
    video_file="tony_squats.mp4",
):
    """
    Load video to cv2 and same 30 frames per second to folder

    """
    print(f'Loading video {video_file} into cv2')
    print(project_root)

    video_label = video_file.split(".")[0]
    video_path = os.path.join(project_root,"videos", video_file)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print('fps',fps)
    count = 0
    make_folder(os.path.join(project_root, "images", 'unseen', excercise, video_label))
    while video.isOpened():
        success, frame = video.read()
        if success:
            print(f'Writing frame {count}')
            cv2.imwrite(f"{project_root}/images/unseen/{excercise}/{video_label}/{count}.jpg", frame)
            count += 1
        else:
            print('Video ended')
            break

def generate_predictions_onframes(excercise, video_file="tony_squats.mp4"):
    '''
    Generate predictions on frames
    excercise: string, name of excercise
    video_file: string, name of video file
    '''
    #load tensorflow model
    model = tf.keras.models.load_model(f"resources/{excercise}/{excercise}_model.h5")
    #load frames
    frame_path = os.path.join("images", "unseen", excercise, video_file.strip('.mp4'))
    frames = natsorted(os.listdir(frame_path))
    #load features

def generated_graded_video(excercise: str,video_file="tony_squats.mp4"):
    """
    Generates a graded video. Using the predictions, it will grade the video and add a label to each frame
    """
    # create font to be bla
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 14)
    # make the font smaller
   
    video_label = 'tony_squats'
    frame_path = os.path.join(project_root, "images", 'unseen', excercise, video_label)
    frames = natsorted(os.listdir(frame_path))

    # Load predictions
    #predictions = predict_on_unseen_data(excercise, frame_path)
    predictions = pd.read_csv("{unseen_folder}_predictions.csv")
    # Loop over each frame in the animated image
    index = 0
    start = None
    duration = 20
    excericse_count = 0
    prev_state, current_state = "start", "start"
    print(duration)
    excercise_complete, just_finished = False, False
    transition_state = False
    on_new_rep = False
    excercise_count = 0
    for frame in tqdm(frames[0:10], desc="Processing Frame"):
        # add text to the image in the rop right coner saying the class name
        img = Image.open(f"{frame_path}/{frame}")
        height, width = img.size
        img = img.resize((height // 3, width // 3))
        prediction = predictions[
            predictions.image_path == frame
        ].class_name.values[0]
        current_state = prediction
        max_prob = np.max(
            predictions[predictions.image_path == frame][
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
if __name__ == 'main':
    excercise = "squat"
    generated_graded_video(excercise)