
from dataclasses import dataclass
import tensorflow as tf
import numpy as np 
from typing import Tuple
from util.helpers import to_gif
INPUT_SIZE = 192

@dataclass
class Video:
    video_path: str

    def __post_init__(self):
        self.input_video = self.get_video(self.video_path)
        self.input_video_config = self.create_config(self.input_video)
        print(f"Finished initializing the video for {self.video_path}")
    def get_video(self, path: str) -> np.ndarray:
        '''
        Gets the video from the path
        '''
        video = tf.io.read_file(path)
        video = tf.image.decode_gif(video)
        print("video loaded", video.shape)
        return video

    #create a parameter config for the vidoe containing num_frames, image_height, image_width and crop region
    def create_config(self, video: np.ndarray) -> dict:
        '''
        Creates a config for the video
        '''
        num_frames, image_height, image_width, _ = video.shape
        crop_region = init_crop_region(image_height, image_width)
        return {
            'num_frames': num_frames,
            'image_height': image_height,
            'image_width': image_width,
            'crop_region': crop_region
        }
    def generate_keypoints(self) -> Tuple[list, list]:
        '''
        '''

        output_images = []
        keypoints = []
        #bar = display(progress(0, num_frames-1), display_id=True)
        #unpacked the config
        num_frames, image_height, image_width, crop_region = self.input_video_config['num_frames'], self.input_video_config['image_height'], self.input_video_config['image_width'], self.input_video_config['crop_region']
        #FOR TESTING. FOR PRODUCTION, REMOVE THIS LINE
        for frame_idx in range(num_frames):
            print(f"Processing frame {frame_idx}")

            keypoints_with_scores = run_inference(
                movenet, self.input_video[frame_idx, :, :, :], crop_region,
                crop_size=[INPUT_SIZE, INPUT_SIZE])
            keypoints.append(keypoints_with_scores)
            output_images.append(draw_prediction_on_image(
                self.input_video[frame_idx, :, :, :].numpy().astype(np.int32),
                keypoints_with_scores, crop_region=None,
                close_figure=True, output_image_height=300))
            crop_region = determine_crop_region(
                keypoints_with_scores, image_height, image_width)
        output = np.stack(output_images, axis=0)
        print(f"Finsihed labeling the video {self.video_path}")
        self.output_video, self.keypoints = output, keypoints

    def save_output(self, output_path: str):
        '''
        Saves the output video to the output path
        '''
        to_gif(self.output_video, fps = 10, filepath = output_path)