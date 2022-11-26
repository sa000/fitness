import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.utils import img_to_array, save_img, load_img # type: ignore
import tensorflow as tf
import random 
import uuid
#hey
RAW_IMAGES = 'images/raw'
POSTAUGMENTATION_PATH = 'images/processed'
DATASET_TYPES = ['train', 'test']
POSITIONS = ['start', 'end']

#with tensorflow use the avx2 fma if aaviablle
if tf.test.gpu_device_name():
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


#loop through folders in directory
def initialize_folder(excercise):
    if not os.path.exists(os.path.join(POSTAUGMENTATION_PATH, excercise)):
        os.makedirs(os.path.join(POSTAUGMENTATION_PATH, excercise))
    for dataset in DATASET_TYPES: 
        if not os.path.exists(os.path.join(POSTAUGMENTATION_PATH, excercise, dataset)):
            os.makedirs(os.path.join(POSTAUGMENTATION_PATH, excercise, dataset))
        for position in POSITIONS:
            if not os.path.exists(os.path.join(POSTAUGMENTATION_PATH, excercise, dataset, position)):
                os.makedirs(os.path.join(POSTAUGMENTATION_PATH, excercise, dataset, position))

def augment_image(raw_image, excercise, position):
    flipped = tf.image.flip_left_right(img_to_array(raw_image)) 
    images = [raw_image, flipped]
    dataset_type = 'train' if random.random()<.8 else 'test'
    for image in images:
        #select a random number between 0 and 1
        uuid_name = str(uuid.uuid4())
        image_name = f'{uuid_name}.png'
        save_img(os.path.join(POSTAUGMENTATION_PATH, excercise, dataset_type, position, image_name), image)
    #FLip and save image

excercise = 'squat'
initialize_folder(excercise)
for position in POSITIONS:
    for image in os.listdir(os.path.join(RAW_IMAGES, excercise, position)):
        print(f'Processing {image} for {excercise} {position}')
        img = load_img(os.path.join(RAW_IMAGES, excercise, position, image))
        augment_image(img, excercise, position)
