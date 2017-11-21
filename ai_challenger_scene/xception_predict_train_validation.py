# -*- coding: utf-8 -*-
'''Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790.
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.

# Reference:

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

'''
from __future__ import print_function
from __future__ import absolute_import
import gc
import numpy as np
from PIL import Image
import json
import warnings
import numpy as np
import os
import warnings
from datetime import datetime
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint

SCENE_MODEL_SAVE_PATH = "/home/yan/Desktop/QlabChallengerRepo/ai_challenger_scene/Xception"
SCENE_TEST_DATA_FOLDER_PATH = "/home/yan/Desktop/QlabChallengerRepo/dataset_299/scene_train_content_resize"
PREDICT_MODEL = "/home/yan/Desktop/QlabChallengerRepo/ai_challenger_scene/predict_loaded_models/XCEPTION_MODEL_WEIGHTS.04-0.81376.h5"

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

def Xception(img_rows, img_cols, color_type=1, num_classes=None):
    
     # Determine proper input shape
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

   
    x_newfc = GlobalAveragePooling2D(name='avg_pool')(x)
    x_newfc = Dense(num_classes, activation='softmax', name='predictions')(x_newfc)
    # Create model.
    model = Model(img_input, x_newfc, name='xception')

    # load weights
    model.load_weights('Xception/XCEPTION_MODEL_WEIGHTS.04-0.81376.h5')
    
    # Learning rate is changed to 0.001
    sgd = SGD(lr=5e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':

    img_rows, img_cols = 299, 299 # Resolution of inputs
    channel = 3
    num_classes = 80
    batch_size = 8
    nb_epoch = 1

    # Load Scene data. Please implement your own load_data() module for your own dataset
    if os.path.exists(SCENE_TEST_DATA_FOLDER_PATH):
        test_data_files = GetJpgList(SCENE_TEST_DATA_FOLDER_PATH)
    else:
        print('Test data folder can not find ...')

    # Load our model
    model = Xception(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

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

    predict_json_file_path = open("/home/yan/Desktop/Xception" + "_predict_train.json", "w")

    json.dump(predict_json, predict_json_file_path)

    gc.collect()
