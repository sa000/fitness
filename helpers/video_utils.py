from natsort import natsorted

import cv2
from matplotlib import font_manager
from PIL import ImageFont, Image
import os
from tqdm import tqdm
import numpy as np
from initialize_resources import make_folder

def load_video_into_cv2(
    excercise,
    video_file="tony_squats.mp4",
):
    """
    Load video to cv2 and same 30 frames per second to folder

    """
    print(f'Loading video {video_file} into cv2')
    project_root = os.path.dirname(os.path.dirname(__file__))
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


def generated_graded_video(predictions: list, excercise: str,video_file="tony_squats.mp4"):
    """
    Generates a graded video. Using the predictions, it will grade the video and add a label to each frame
    """
    # create font to be bla
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 14)
    # make the font smaller

    frame_path = os.path.join("images", "unseen", excercise, video_file)
    frames = natsorted(os.listdir(frame_path))
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
    for frame_path in tqdm(frames, desc="Processing Frame"):
        print(frame_path)
        # add text to the image in the rop right coner saying the class name
        img = Image.open("{unseen_path}/" + frame_path)
        height, width = img.size
        img = img.resize((height // 3, width // 3))
        prediction = predictions[
            predictions.frame_path == frame_path
        ].class_name.values[0]
        current_state = prediction
        max_prob = np.max(
            predictions[predictions.frame_path == frame_path][
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
