from keras.preprocessing import image
import os
import cv2
import glob
import numpy as np
import pandas as pd
from keras.models import model_from_json
from django.core.files.storage import default_storage
from django.conf import settings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_model_and_pretrained_weights():
    """
    loading our CNN model.The parameters and structure of model
    is present in that file

    """
    json_file = open('./uploadimage/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./uploadimage/model.h5")
    return loaded_model


def predict(file):
    file_name = default_storage.save(file.name, file)
    file_url = default_storage.url(file_name)
    test_image = image.load_img('/home/mohan/venv/imagetag'+file_url, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    loaded_model = load_model_and_pretrained_weights()
    result = loaded_model.predict(test_image)
    classes_array = ['OTHER', 'animal', 'cartoon', 'chevron', 'floral',
                     'geometry', 'houndstooth', 'ikat', 'letter_numb',
                     'plain', 'polka dot', 'scales', 'skull', 'squares',
                     'stars', 'stripes', 'tribal']
    return classes_array[np.where(result[0] == 1)[0][0]]
