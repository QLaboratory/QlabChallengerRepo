# -*- coding: utf-8 -*-
import sys
import pickle
import cv2
import numpy as np

from keras import backend as K
from keras.utils import np_utils



# DATA_URL_SCENE_TRAIN = "D:\\QlabChallengerRepo\\dataset\\ai_challenger_scene_train_direct_resize"
# DATA_URL_SCENE_VALIDATION = "D:\\QlabChallengerRepo\\dataset\\ai_challenger_scene_validation_direct_resize"
# DATA_URL_SCENE_TEST = "D:\\QlabChallengerRepo\\dataset\\ai_challenger_scene_test"
DATA_URL_SCENE_TRAIN = "/home/yan/Desktop/QlabChallengerRepo/dataset/ai_challenger_scene_train_direct_resize"
DATA_URL_SCENE_VALIDATION = "/home/yan/Desktop/QlabChallengerRepo/dataset/ai_challenger_scene_validation_direct_resize"
DATA_URL_SCENE_TEST = "/home/yan/Desktop/QlabChallengerRepo/dataset/ai_challenger_scene_test"

nb_train_samples = 53879 # 53879 training samples
nb_valid_samples = 7120 # 100 validation samples
num_classes = 80


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing AI Challenger Scene data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 224, 224)
    return data, labels


def load_data():
    """Loads AI Challenger Scene dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    num_train_samples = 53879

    x_train = np.zeros((num_train_samples, 3, 224, 224), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    x_train, y_train = load_batch(DATA_URL_SCENE_TRAIN)
    x_test, y_test = load_batch(DATA_URL_SCENE_VALIDATION)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_scene_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows, img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples, :, :, :]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows, img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples, :, :, :]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid
