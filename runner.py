import os

from helpers.video_utils import (
    convert_frames_to_gif,
    generated_graded_video,
    load_video_into_cv2,
)
from model_generator import predict_on_unseen_data

excercise = "squat"

os.system(f"python3 model_generator.py {excercise}")

