import os
import tensorflow as tf
from tools import constants
import cv2 as cv
import numpy as np

dict_defects = {0: "Шлак", 1: "Прожог или Поры или свищ", 2 :"Брызги металла", 3: "Кратер или раковина", 4: "Шов не обнаружен(осутствует-зачищен-сильно загрязнен)"}
path_stat = '//media//sergei//0B8209AF0B8209AF//Статистика дефектов//07.04.2021_3_14'
path_image = '//media//sergei//0B8209AF0B8209AF//07.04//(2290)07.04.2021_3_14__875.00seams'
threshold_defect = 0.65

def run():
    list_images = os.listdir(path_image)
    #list_images = list_images[1196:-1]
    model_weld = tf.keras.models.load_model(constants.CLASSIFIER_BINARY_SAVE_PATH + '//8.02classifier_weld0.989.h5',
                                            compile=False)
    model_weld.summary()

    model_defect = tf.keras.models.load_model(
        constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/21.03classifier_defects0.904.h5', compile=False)
    model_defect.summary()
    counter = 0

    for image_name in list_images:
        counter = counter+1
        img_origin = cv.imread(path_image + '//' + image_name)
        img_weld = cv.cvtColor(img_origin, cv.COLOR_BGR2RGB)

        img_weld = cv.resize(img_weld, (constants.CLASSIFIER_BINARY_IMG_SIZE[1], constants.CLASSIFIER_BINARY_IMG_SIZE[0]))

        img_weld = img_weld/255.0
        img_weld = np.expand_dims(img_weld, axis=0)


        pred_weld = model_weld.predict(img_weld)[0][0]
        if pred_weld > 0.3:
            img_defects = cv.cvtColor(img_origin, cv.COLOR_BGR2RGB)
            img_defects = img_defects/255.0
            img_defects = np.expand_dims(img_defects, axis=0)
            pred_defects = model_defect.predict(img_defects)[0]
            if pred_defects[4] < 0.6:
                defects = pred_defects[0:3]
                for i in range(len(defects)):
                    if defects[i]>0.65:

                        cv.imwrite(path_stat + '//' + dict_defects[i] + '//' + image_name, cv.resize(img_origin, (
                        constants.CLASSIFIER_BINARY_IMG_SIZE[1], constants.CLASSIFIER_BINARY_IMG_SIZE[0])))
                        print("Write {}".format(dict_defects[i]))

        else:
            cv.imwrite(path_stat +'//'+dict_defects[4]+ '//' + image_name, cv.resize(img_origin, (constants.CLASSIFIER_BINARY_IMG_SIZE[1], constants.CLASSIFIER_BINARY_IMG_SIZE[0])))
            print("Write {}".format(dict_defects[4]))



if __name__ == '__main__':
    run()