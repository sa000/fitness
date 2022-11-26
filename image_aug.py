import os
from tensorflow.keras.utils import img_to_array, save_img, load_img # type: ignore
import tensorflow as tf
import random 

RAW_IMAGES = '/Users/sakib/Desktop/excercises/raw'
POSTAUGMENTATION_PATH = '/Users/sakib/Desktop/excercises/postprocessing_excercises'
#loop through folders in directory
for excercise in os.listdir(RAW_IMAGES):
    #loop through images in folder
    if excercise=='.DS_Store':
        continue
    for position in os.listdir(os.path.join(RAW_IMAGES, excercise)):
        im_number = 0 
        if position=='.DS_Store':
            continue
        excercise_position = f'{excercise}_{position}'
        print(f'Augmentating Data for {excercise_position}')
        #Make a folder if it doesn't exist in postprocessing
        for folder in ['train', 'test']:
            if not os.path.exists(f'{POSTAUGMENTATION_PATH}/{folder}/{excercise_position}'):
                os.makedirs(f'{POSTAUGMENTATION_PATH}/{folder}/{excercise_position}')
        #load images in position folder
        
        for image in os.listdir(os.path.join(RAW_IMAGES, excercise, position)):
            # #load image
            print(image)
            img = load_img(os.path.join(RAW_IMAGES, excercise, position, image))
            flipped = tf.image.flip_left_right(img_to_array(img)) 
            images = [img, flipped]
            for im in images:
                im_number+=1
                image_name = f'image_{im_number}.png'
                #select a random number between 0 and 1
                folder = 'train' if random.random()<.8 else 'test'
                save_img(os.path.join(POSTAUGMENTATION_PATH, folder, excercise_position, image_name), im)
            #FLip and save image
            
