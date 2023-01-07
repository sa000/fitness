# def get_columns_to_drop(excercise: str):
#     '''
#     Get the columns to drop from the dataframe. Aim to help model learn faster with only relevant features
#     '''
#     if excercise=='kb_around_world':
#         not_revelant_body_parts = ['NOSE', 'EYE', 'EAR', 'KNEE', 'ANKLE']
#     #for each not relevant body part, remove the LEFT, RIGHT and SCORE columns from the BODYPART columns
#     columns_to_drop = []
#     for body_part in not_revelant_body_parts:
#         print(body_part)
#         for column in [body_part+'_x', body_part+'_y', body_part+'_score']:
#             columns_to_drop.append(column)
#     return columns_to_drop

# print(get_columns_to_drop('kb_around_world'))

import boto3
from tensorflow.keras.utils import (img_to_array, load_img,  # type: ignore
                                    save_img)
from tqdm import tqdm
# Create a boto3 client for the S3 service
s3 = boto3.client('s3')

# List all objects in the directory
response = s3.list_objects(Bucket='fitness-sa-tx', Prefix='images/raw/kb_around_the_world/start/')

# Get a list of all the objects in the directory
objects = response['Contents']

# Iterate over the objects and print their key (the full path to the object in the bucket)
for obj in objects[0:10]:
    if obj['Key'][-1] == '/':
        continue
    a = load_img(obj['Key'])
    print(a)
    break

