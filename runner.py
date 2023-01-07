import os

from helpers.video_utils import (
    convert_frames_to_gif,
    generated_graded_video,
    load_video_into_cv2,
)
from model_generator import predict_on_unseen_data

excercise = "kb_around_the_world"
# os.system(f"python3 initialize_resources.py {excercise}")
# os.system(f"python3 image_aug.py {excercise}")
# os.system(f"python3 feature_generation.py {excercise}")
# os.system(f"python3 model_generator.py {excercise}")

video_file = "kb_around_the_world_v2.mp4"
# load_video_into_cv2(excercise, video_file)
# predict_on_unseen_data(excercise,video_file)
# generated_graded_video(excercise,video_file)
convert_frames_to_gif(excercise,video_file)
