import os

from helpers.video_utils import (
    convert_frames_to_gif,
    generated_graded_video,
    load_video_into_cv2,
)

excercise = "squat"

os.system(f"python3 model_generator.py {excercise}")

