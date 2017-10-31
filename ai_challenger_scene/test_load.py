# -*- coding: utf-8 -*-
import sys
import pickle
import cv2
import numpy as np

from keras import backend as K
from keras.utils import np_utils

IMAGE_SIZE = 299
DATA_URL_SCENE_VALIDATION = "/home/yan/Desktop/QlabChallengerRepo/test_pickle/ai_challenger_scene_validation_content_resize.npz"


def load_batch(fpath):
    """Internal utility for parsing AI Challenger Scene data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    d = np.load(fpath)    
	
    data = d['arr_0']
    labels = d['arr_1']

    data = data.reshape(data.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
    return data, labels

data,labels = load_batch(DATA_URL_SCENE_VALIDATION)
print(labels)
