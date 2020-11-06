import tensorflow as tf
from tools import constants
from tools import statistics
from tools import load_data
import cv2 as cv
import numpy as np

def run():
    jsons = statistics.get_jsons()
    images, labels, counter = load_data.get_marking(jsons)
    print(counter)
    train_images, train_labels, validation_images, validation_labels = load_data.split_strat_defects(images,
                                                                                                     labels,
                                                                                                     split_size=constants.CLASSIFIER_MULTI_LABEL_SPLIT,
                                                                                                     seed=constants.CLASSIFIER_MULTI_LABEL_RANDOM_SEED)


    classifier_model = tf.keras.models.load_model(
        constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/classifier_defects.h5', compile=False)

    cv.namedWindow('test', cv.WINDOW_NORMAL)

    for idx in range(len(validation_images)):
        image = cv.imread(validation_images[idx])
        cv.imshow('test', image)
        image = cv.resize(image, dsize=(constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1]))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image/255.0
        image = np.expand_dims(image, axis=0)

        res = classifier_model.predict(image)
        res = np.squeeze(res)
        res = np.round(res)

        print('Network: {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{}'.format(
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[0], res[0],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[1], res[1],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[2], res[2],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[3], res[3],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[4], res[4],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[5], res[5],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[6], res[6],
        ))

        print('True lb: {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{} -- {}:{}'.format(
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[0], validation_labels[idx][0],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[1], validation_labels[idx][1],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[2], validation_labels[idx][2],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[3], validation_labels[idx][3],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[4], validation_labels[idx][4],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[5], validation_labels[idx][5],
            constants.CLASSIFIER_MULTI_LABEL_CLASSES[6], validation_labels[idx][6],
        ))

        print()
        print()
        cv.waitKey()






if __name__ == '__main__':
    run()