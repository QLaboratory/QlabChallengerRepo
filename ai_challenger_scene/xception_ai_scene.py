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

import warnings
import numpy as np
import os
import gc
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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from scipy import misc  
import numpy
import numpy.random
import scipy.ndimage
import scipy.misc

SCENE_MODEL_SAVE_PATH = "/home/yan/Desktop/QlabChallengerRepo/ai_challenger_scene/Xception"

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
    model.load_weights('Xception/XCEPTION_MODEL_WEIGHTS.01-0.74157.h5')
    
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def random_crop_image(image):
      height, width = image.shape[:2]
      random_array = numpy.random.random(size=4);
      w = int((width*0.5)*(1+random_array[0]*0.5))
      h = int((height*0.5)*(1+random_array[1]*0.5))
      x = int(random_array[2]*(width-w))
      y = int(random_array[3]*(height-h))
      
      image_crop = image[y:h+y,x:w+x,0:3]
      image_crop = misc.imresize(image_crop,image.shape)

      image_crop = image_crop/255.
      image_crop = image_crop - 0.5
      image_crop = image_crop*2.0
      return image_crop

if __name__ == '__main__':

    img_rows, img_cols = 299, 299 # Resolution of inputs
    channel = 3
    num_classes = 80
    batch_size = 8
    nb_epoch = 2
    nb_train_samples = 53880
    nb_validation_samples = 7120

    # Load our model
    model = Xception(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

    #classes
    our_class = []
    for i in range(num_classes):
        our_class.append(str(i))

    # data arguement
    train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                preprocessing_function=random_crop_image,
                fill_mode='reflect')
    test_datagen = ImageDataGenerator(
                preprocessing_function=random_crop_image,
                fill_mode='reflect')

    train_generator = train_datagen.flow_from_directory(
                '/home/yan/Desktop/QlabChallengerRepo/dataset/train/',
                target_size=(img_rows,img_cols),
                batch_size=batch_size,
                classes=our_class)
    validation_generator = test_datagen.flow_from_directory(
                '/home/yan/Desktop/QlabChallengerRepo/dataset/valid/',
                target_size=(img_rows,img_cols),
                batch_size=batch_size,
                classes=our_class)
    #print(train_generator.class_indices)
    #print(validation_generator.class_indices)

    # Callback
    checkpointer = ModelCheckpoint(filepath='/home/yan/Desktop/QlabChallengerRepo/ai_challenger_scene/Xception/XCEPTION_MODEL_WEIGHTS.{epoch:02d}-{val_acc:.5f}.h5',
                                   monitor='val_acc',
                                   verbose=1,
                                   save_weights_only= True,
                                   save_best_only=False)

    # Start Fine-tuning
    model.fit_generator(train_generator,
              steps_per_epoch=nb_train_samples//batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              callbacks=[checkpointer],
              validation_data=validation_generator,
              validation_steps=nb_validation_samples//batch_size)

    gc.collect()
