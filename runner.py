import os
from video_utils import generated_graded_video,load_video_into_cv2
excercise = 'squat'
#os.system('python3 initialize_resources.py')
#os.system(f"python3 image_aug.py {excercise}")
os.system(f"python3 feature_generation.py {excercise}")  
#os.system(f"python3 model_generator.py {excercise}")
load_video_into_cv2(excercise)

generated_graded_video(excercise)