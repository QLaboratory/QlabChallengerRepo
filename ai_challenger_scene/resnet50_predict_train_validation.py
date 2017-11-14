# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
import warnings
import gc
import sys
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import json


from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint

SCENE_TEST_DATA_FOLDER_PATH = "/home/yan/Desktop/QlabChallengerRepo/dataset_224/scene_validation_content_resize"
PREDICT_MODEL = "/home/yan/Desktop/QlabChallengerRepo/ai_challenger_scene/trained_BEST_model/RESNET_50_MODEL_WEIGHTS.06-0.77963.hdf5"

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(img_rows, img_cols, color_type=1, num_classes=None):
     # Determine proper input shape
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')


    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

     # load weights
    model.load_weights(PREDICT_MODEL)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def GetJpgList(p):
    if p == "":
        return []
    # p = p.replace("/", "\\")
    if p[-1] != "/":
        p = p + "/"
    file_list = os.listdir(p)
    jpg_list = []
    for i in file_list:
        if os.path.isfile(p + i):
            name, suffix = os.path.splitext(p + i)
            if ('.jpg' == suffix):
                jpg_list.append(i)
    return jpg_list


if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 80
    batch_size = 8
    nb_epoch = 1

    # Load Scene data. Please implement your own load_data() module for your own dataset
    if os.path.exists(SCENE_TEST_DATA_FOLDER_PATH):
        test_data_files = GetJpgList(SCENE_TEST_DATA_FOLDER_PATH)
    else:
        print('Test data folder can not find ...')

    model = ResNet50(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

    # Make predictions
    predict_json = []
    count = 1
    totalnum = str(len(test_data_files))
    # predict_annotation_dic_temp = {}
    # predict_annotation_dic_temp['image_id'] = "1.jpg"
    # predict_annotation_dic_temp['label_id'] = [1, 2, 3]
    # predict_json.append(predict_annotation_dic_temp)
    # predict_annotation_dic_temp = {}
    # predict_annotation_dic_temp['image_id'] = "2.jpg"
    # predict_annotation_dic_temp['label_id'] = [2, 3, 4]
    # predict_json.append(predict_annotation_dic_temp)

    for i in test_data_files:
        im = Image.open(os.path.join(SCENE_TEST_DATA_FOLDER_PATH, i))
        im_array = np.array(im).reshape(1, img_rows, img_cols, channel)
        predictions_valid = model.predict(im_array, verbose=0)

        predict_annotation_dic_temp = {}
        predict_annotation_dic_temp['image_id'] = i
        predict_label_id = predictions_valid[0]
        predict_annotation_dic_temp['label_id'] = predict_label_id.tolist()
        if (count % 100 == 0):
            print(str(count) + "/" + totalnum)
        # print(predict_annotation_dic_temp)
        # print(predict_label_id)
        count += 1
        predict_json.append(predict_annotation_dic_temp)

    predict_json_file_path = open("/home/yan/Desktop/ResNet50" + "_predict_validation.json", "w")

    json.dump(predict_json, predict_json_file_path)

    gc.collect()
