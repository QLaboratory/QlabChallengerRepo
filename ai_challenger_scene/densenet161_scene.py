# -*- coding: utf-8 -*-
import os
import gc

from datetime import datetime
from keras.models import load_model
from sklearn.metrics import log_loss
from load_scene import load_scene_data

SCENE_MODEL_SAVE_PATH = "/home/yan/Desktop/QlabChallengerRepo/ai_challenger_scene/imagenet_models"


if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 80
    # batch_size = 1
    batch_size = 8
    # nb_epoch = 10
    nb_epoch = 1

    # Load Scene data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_scene_data(img_rows, img_cols)

    # Load our model
    LAST_SAVED_MODEL = "MODEL"
    LAST_SAVED_MODEL_PATH = os.path.join(SCENE_MODEL_SAVE_PATH, LAST_SAVED_MODEL)
    model = load_model(LAST_SAVED_MODEL)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    CURRENT_TIME = "MODEL_"+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+".h5"
    CURRENT_SCENE_MODEL_SAVE_PATH = os.path.join(SCENE_MODEL_SAVE_PATH, CURRENT_TIME)

    model.save_weights(CURRENT_SCENE_MODEL_SAVE_PATH)

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print("score: ", score)

    gc.collect()
