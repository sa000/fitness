import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import font_manager
from natsort import natsorted
from PIL import Image, ImageFont
from tqdm import tqdm

from initialize_resources import make_folder
from model_generator import predict_on_unseen_data

project_root = os.path.dirname(os.path.dirname(__file__))

def load_video_into_cv2(
    excercise,
):
    """
    Load video to cv2 and same 30 frames per second to folder

    """
    video_file =f"{excercise}.mp4"
    print(f'Loading video {video_file} into cv2')

    video_path = os.path.join(project_root,'resources',excercise, video_file)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print('fps',fps)
    count = 0
    make_folder(os.path.join(project_root, "images", 'video_frames', excercise))
    while video.isOpened():
        success, frame = video.read()
        if success:
            print(f'Writing frame {count}')
            cv2.imwrite(f"{project_root}/images/video_frames/{excercise}/{count}.jpg", frame)
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

def generated_graded_video(excercise: str):
    """
    Generates a graded video. Using the predictions, it will grade the video and add a label to each frame
    """
    # create font to be bla
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 14)
    # make the font smaller
   
    frame_path = os.path.join(project_root, "images", 'video_frames', excercise)
    label_path = os.path.join(project_root, "images", 'video_frames', excercise+'_labeled')
    frames = natsorted(os.listdir(frame_path))

    # Load predictions
    # predictions = predict_on_unseen_data(excercise, frame_path)
    predictions = pd.read_csv(f"{excercise}_video_predictions.csv")
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
    for frame in tqdm(frames, desc="Processing Frame"):
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


        # Increse the excercise count if we are back in the start state
        if current_state == "start" and prev_state == "end":
            excercise_count += 1
            print(excercise_count)
        prev_state = current_state
        print(frame, excercise_count)
        #write the probablity of the prediction on the image 
        img = np.array(img)
        img = cv2.putText(
            img,
            f"{prediction}:{max_prob} Reps: {excercise_count}",
            (15, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        img = Image.fromarray(img)
        img.save(f"{label_path}/{frame}")

def convert_frames_to_gif(
    excercise: str, fps: int = 30
):
    """
    Convert frames to gif
    """
    import imageio
    frame_path = os.path.join(project_root, "images", 'video_frames', excercise+'_labeled')
    # Get the frames and convert them to a gif
    frames = natsorted(os.listdir(frame_path))
    images = []
    for frame in frames:
        images.append(imageio.imread(f"{frame_path}/{frame}"))
    imageio.mimsave(f"resources/{excercise}/{excercise}_labeled.gif", images, fps=fps)

if __name__ == 'main':
    excercise = "squat"
    #generated_graded_video(excercise)
    convert_frames_to_gif(excercise)