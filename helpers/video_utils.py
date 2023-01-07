import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import font_manager
from natsort import natsorted
from tensorflow.keras.utils import img_to_array, load_img  # type: ignore
from PIL import Image, ImageFont
import io
from tqdm import tqdm
import boto3
from globals import BUCKET_NAME
from io import BytesIO
import awswrangler as wr

s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")
project_root = os.path.dirname(os.path.dirname(__file__))


def load_video_into_cv2(excercise, video_file):
    """
    Load video to cv2 and same 30 frames per second to folder

    """
    print(f"Loading video {video_file} into cv2")
    video_label = video_file.strip(".mp4")
    video_path = os.path.join("resources", excercise, video_file)
    s3_client.download_file(BUCKET_NAME, video_path, video_file)
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    print("fps", fps)
    count = 0
    success = True  
    while video.isOpened() and success:
        success, frame = video.read()
        image_path = os.path.join(
            "images",
            "video_frames",
            excercise,
            video_label,
            "unlabeled",
            f"{count}.jpg",
        )
        if success:
            _, image_data = cv2.imencode(".jpg", frame)
            print(f"Writing frame {count}")
            s3_client.put_object(
                Bucket=BUCKET_NAME, Key=image_path, Body=image_data.tostring()
            )
        count += 1
    os.remove(video_file)


def is_new_excercise(excercise: str, current_state: str, prev_state: str):
    if current_state == "start" and prev_state == "end":
        return True
    return False


def generated_graded_video(excercise: str, video_file: str):
    """
    Generates a graded video. Using the predictions, it will grade the video and add a label to each frame
    """
    # create font to be bla
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 14)
    # make the font smaller
    video_label = video_file.strip(".mp4")

    frame_path = os.path.join(
        "images", "video_frames", excercise, video_label, "unlabeled"
    )
    print(frame_path)
    label_path = os.path.join(
        "images", "video_frames", excercise, video_label, "labeled"
    )

    objects = s3_client.list_objects(Bucket=BUCKET_NAME, Prefix=frame_path)["Contents"]

    # frames = natsorted(os.listdir(frame_path))

    # Load predictions
    # predictions = pd.read_csv(f"{excercise}_{video_label}_video_predictions.csv")
    predictions = wr.s3.read_csv(
        f"s3://{BUCKET_NAME}/resources/{excercise}/{video_label}/predictions.csv"
    )
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
    for object in tqdm(
        sorted(objects, key=lambda x: int(x["Key"].split("/")[-1].strip(".jpg"))),
        desc="Processing Frame",
    ):
        if object["Key"] == "/":
            continue
        frame_path = object["Key"]
        print(frame_path)
        body = s3_resource.Object(BUCKET_NAME, frame_path).get()["Body"].read()
        img = load_img(io.BytesIO(body))

        # add text to the image in the rop right coner saying the class name
        # img = Image.open(f"{frame_path}/{frame}")
        height, width = img.size
        img = img.resize((height // 3, width // 3))
        prediction = predictions[
            predictions.image_path == frame_path
        ].class_name.values[0]
        current_state = prediction
        max_prob = np.max(
            predictions[predictions.image_path == frame_path][
                ["class_0", "class_1"]
            ].values
        )
        max_prob = float(int(max_prob * 1000)) / 1000
        should_increase_rep = is_new_excercise(excercise, current_state, prev_state)
        if should_increase_rep:
            if excercise == "kb_around_the_world":
                excercise_count += 0.5
            else:
                excercise_count += 1
            print(excercise_count)
        prev_state = current_state
        # write the probablity of the prediction on the image
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
        buf = io.BytesIO()
        img.save(buf, format="png")
        buf.seek(0)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=label_path + "/" + frame_path.split("/")[-1],
            Body=buf,
        )


def convert_frames_to_gif(excercise: str, video_file: str, fps: int = 30):
    """
    Convert frames to gif
    """
    import imageio

    video_label = video_file.strip(".mp4")
    label_path = os.path.join(
        "images", "video_frames", excercise, video_label, "labeled"
    )
    # Get the frames and convert them to a gif
    objects = s3_client.list_objects(Bucket=BUCKET_NAME, Prefix=label_path)["Contents"]

    images = []
    for object in tqdm(
        sorted(objects, key=lambda x: int(x["Key"].split("/")[-1].strip(".jpg"))),
        desc="Processing Frame",
    ):
        if object["Key"] == "/":
            continue
        frame_path = object["Key"]
        body = s3_resource.Object(BUCKET_NAME, frame_path).get()["Body"].read()
        img = load_img(io.BytesIO(body))
        images.append(img)
    imageio.mimsave(f"{video_label}_labeled.gif", images, fps=fps)
    # supload
    s3_client.upload_file(
        f"{video_label}_labeled.gif",
        BUCKET_NAME,
        f"images/video_frames/{excercise}/{video_label}/labeled.gif",
    )
    os.remove(f"{video_label}_labeled.gif")


if __name__ == "main":
    excercise = "kb_around_the_world"
    video_file = "kb_around_the_world_v2"
    load_video_into_cv2(excercise, video_file)

    # generated_graded_video(excercise,video_file)
    # convert_frames_to_gif(excercise,video_file)
