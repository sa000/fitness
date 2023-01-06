import keras 
import os
import tensorflow as tf
excercises = ['kb_situp']

from globals import RESOURCES_ROOT

for excercise in excercises:
    #load the model
    print(excercise)
    model = keras.models.load_model(f"{RESOURCES_ROOT}/{excercise}/{excercise}_model.h5", compile=False)
    model.compile()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    print('Model size: %dKB' % (len(tflite_model) / 1024))

    with open(f"{RESOURCES_ROOT}/{excercise}/{excercise}_model.tflite", 'wb') as f:
        f.write(tflite_model)