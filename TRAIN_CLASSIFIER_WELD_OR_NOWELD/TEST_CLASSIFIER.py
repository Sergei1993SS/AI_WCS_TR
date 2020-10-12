from tools import load_data
from tools import constants
from models import models
import tensorflow as tf
import cv2 as cv
import os
import numpy as np

def run():
    print('Start load data info')
    list_NO_weld = [constants.CLASSIFIER_BINARY_PATH_NO_WELD + file for file in os.listdir(constants.CLASSIFIER_BINARY_PATH_NO_WELD)]
    list_YES_weld = load_data.make_list_yes_weld(constants.CLASIIFIER_MODE_LOAD)
    train_pos, train_neg, validation_pos, validation_neg = load_data.split_dataset_classifier_weld(list_NO_weld, list_YES_weld, constants.CLASSIFIER_BINARY_SPLIT_SIZE, constants.CLASSIFIER_BINARY_NP_SEED)

    model = tf.keras.models.load_model(constants.CLASSIFIER_BINARY_SAVE_PATH + '/classifier_weld_95_89.h5')
    model.summary()

    cv.namedWindow('valid_pos', cv.WINDOW_NORMAL)
    for file in validation_pos:
        img = cv.imread(file)
        img = cv.resize(img, dsize=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1]))
        pred = model.predict(np.expand_dims(img, axis=0))
        print(pred[0][0])
        if pred[0][0]<0.5:
            cv.imshow('valid_pos', img)
            cv.waitKey()


if __name__ == '__main__':
    run()