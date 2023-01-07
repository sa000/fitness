# import keras 
import os
# import tensorflow as tf
excercises = ['kb_situp']

from globals import RESOURCES_ROOT

# for excercise in excercises:
#     #load the model
#     print(excercise)
#     model = keras.models.load_model(f"{RESOURCES_ROOT}/{excercise}/{excercise}_model.h5", compile=False)
#     model.compile()
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_model = converter.convert()

#     print('Model size: %dKB' % (len(tflite_model) / 1024))

#     with open(f"{RESOURCES_ROOT}/{excercise}/{excercise}_model.tflite", 'wb') as f:
#         f.write(tflite_model)

# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/

import boto3
from botocore.exceptions import ClientError


def get_secret():

    secret_name = "access_key"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    print(secret)
    # Your code goes here.
get_secret()