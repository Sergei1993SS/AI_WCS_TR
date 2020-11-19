import tensorflow as tf
from tools import constants
from tools import statistics
from tools import load_data
import cv2 as cv
import numpy as np
import os
from models import models, metrics



def run():

    tf.keras.backend.clear_session()

    jsons = statistics.get_jsons()
    images, labels, counter = load_data.get_marking_balanced_dataset_cast(jsons)
    print(counter)
    glass_images_train, glass_labels_train, glass_images_validation, glass_labels_validation, \
    burn_and_fistula_images_train, burn_and_fistula_labels_train, burn_and_fistula_images_validation, burn_and_fistula_labels_validation, \
    metal_spray_images_train, metal_spray_labels_train, metal_spray_images_validation, metal_spray_labels_validation, \
    crater_images_train, crater_labels_train, crater_images_validation, crater_labels_validation, \
    shell_images_train, shell_labels_train, shell_images_validation, shell_labels_validation, \
    background_images_train, background_labels_train, background_images_validation, background_labels_validation = load_data.split_balanced_defects_cast(
        images, labels, constants.CLASSIFIER_MULTI_LABEL_SPLIT, constants.CLASSIFIER_MULTI_LABEL_RANDOM_SEED)


    classifier_model = tf.keras.models.load_model(constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/classifier_defects0.871.h5', compile=False)

    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = tf.losses.BinaryCrossentropy()

    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.f1], run_eagerly=False)

    validation_images = []
    validation_images.extend(glass_images_validation)
    validation_images.extend(burn_and_fistula_images_validation)
    validation_images.extend(metal_spray_images_validation)
    validation_images.extend(crater_images_validation)
    validation_images.extend(shell_images_validation)
    validation_images.extend(background_images_validation)

    validation_labels = []
    validation_labels.extend(glass_labels_validation)
    validation_labels.extend(burn_and_fistula_labels_validation)
    validation_labels.extend(metal_spray_labels_validation)
    validation_labels.extend(crater_labels_validation)
    validation_labels.extend(shell_labels_validation)
    validation_labels.extend(background_labels_validation)

    cv.namedWindow('test', cv.WINDOW_NORMAL)

    for idx in range(len(validation_images)):
        image = cv.imread(validation_images[idx])
        cv.imshow('test', image)
        image = cv.resize(image, dsize=(constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1]))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image/255.0
        image = np.transpose(image, axes=[1, 0 ,2])
        print(image.shape)
        image = np.expand_dims(image, axis=0)

        res = classifier_model.predict(image)
        res = np.squeeze(res)

        print('Network: {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{}'.format(
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[0], np.round(res[0], decimals=2),
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[1], np.round(res[1], decimals=2),
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[2], np.round(res[2], decimals=2),
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[3], np.round(res[3], decimals=2),
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[4], np.round(res[4], decimals=2),
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[5], np.round(res[5], decimals=2),
            #constants.CLASSIFIER_MULTI_LABEL_CLASSES[6], np.round(res[6], decimals=2),
        ))

        print('True lb: {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{}'.format(
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[0], validation_labels[idx][0],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[1], validation_labels[idx][1],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[2], validation_labels[idx][2],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[3], validation_labels[idx][3],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[4], validation_labels[idx][4],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[5], validation_labels[idx][5],
            #constants.CLASSIFIER_MULTI_LABEL_CLASSES[6], validation_labels[idx][6],
        ))

        print()
        print()
        cv.waitKey()






if __name__ == '__main__':
    run()