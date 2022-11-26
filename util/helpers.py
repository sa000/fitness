from tensorflow_docs.vis import embed
import imageio
import numpy as np
from matplotlib import font_manager
import io
from PIL import Image, ImageDraw, ImageSequence,ImageFont


def to_gif(images, filepath, fps=10):
    """Converts image sequence (4D numpy array) to gif."""
    imageio.mimsave(filepath, images, fps=fps)
    print("Gif saved to {}".format(filepath))
    return embed.embed_file(filepath)

def generate_videos_side_by_side(video1: np.ndarray, video2: np.ndarray, filepath):
    sidebyside_output = np.dstack((video1, video2))
    to_gif(sidebyside_output, filepath)

def generated_graded_video(im: Image, grades: list, filename: str):
    '''
    Generates a graded video
    '''
    font = font_manager.FontProperties(family='sans-serif', weight='bold')
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 48)

    frames = []
    # Loop over each frame in the animated image
    index = 0 
    
    for frame in ImageSequence.Iterator(im):
        # Draw the text on the frame
        d = ImageDraw.Draw(frame)
        d.text((100,10), str(grades[index]), font=font)

        del d

        # However, 'frame' is still the animated image with many frames
        # It has simply been seeked to a later frame
        # For our list of frames, we only want the current frame

        # Saving the image without 'save_all' will turn it into a single frame image, and we can then re-open it
        # To be efficient, we will save it to a stream, rather than to file
        b = io.BytesIO()
        frame.save(b, format="GIF")
        frame = Image.open(b)

        # Then append the single frame image to a list of frames
        index +=1
        frames.append(frame)
    # Save the frames as a new image
    frames[0].save(f'output/graded_{filename}.gif', save_all=True, append_images=frames[1:])
    print('with text')

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 