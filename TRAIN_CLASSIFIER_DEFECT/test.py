import time

import tensorflow as tf
from tools import constants
from tools import statistics
from tools import load_data
import cv2 as cv
import os
from models import models, metrics
import json
from typing import Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


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


    classifier_model = tf.keras.models.load_model(constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/classifier_defects0.896.h5', compile=False)

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


def class_in_masks(masks, class_name):
    for mask in masks:
        if mask['class_name'] == class_name:
            return True

    return False

def image_in_IMAGES(IMAGES, image_name):
    for img in IMAGES:
        if image_name == img['ImageName']:
            return True
    return False

def xz():
    # run()
    path = '/home/sergei/DataSet/875_set10/IMAGES/JPEG/'
    images = os.listdir(path)
    # images = [path + '/' + image for image in images]
    # print(images)

    model_defect = tf.keras.models.load_model(
        constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/precision_classifier_defects0.793.h5', compile=False)
    model_defect.summary()

    model_weld = tf.keras.models.load_model(constants.CLASSIFIER_BINARY_SAVE_PATH + '/classifier_weld0.989.h5',
                                            compile=False)
    model_weld.summary()
    cv.namedWindow('valid_pos', cv.WINDOW_NORMAL)

    dict_true = {'glass': 0, 'burn_and_fistula_pores_and_inclusions': 0, 'metal_spray': 0, 'crater': 0, 'shell': 0,
                 'no_weld': 0, 'yes_weld': 0}
    dict_false = {'glass': 0, 'burn_and_fistula_pores_and_inclusions': 0, 'metal_spray': 0, 'crater': 0, 'shell': 0,
                  'no_weld': 0, 'yes_weld': 0}
    dict_pass = {'glass': 0, 'burn_and_fistula_pores_and_inclusions': 0, 'metal_spray': 0, 'crater': 0, 'shell': 0}
    json_data = {}

    with open("/home/sergei/DataSet/875_set4/JSON/JSON_annotation.json", 'r') as j:
        json_data = json.load(j)
        print(json_data['Images'])
    cv.namedWindow('test', cv.WINDOW_NORMAL)
    for image in images:
        if not image_in_IMAGES(json_data['Images'], image):
            img_out = cv.imread(path + image)
            img = cv.resize(img_out, dsize=(
                constants.CLASSIFIER_BINARY_IMG_SIZE[1], constants.CLASSIFIER_BINARY_IMG_SIZE[0]))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img / 255.0
            pred = model_weld.predict(np.expand_dims(img, axis=0))
            pred = np.round(pred[0][0])
            if pred == 0:
                dict_true['no_weld'] = dict_true['no_weld'] + 1
            else:
                dict_false['yes_weld'] = dict_false['yes_weld'] + 1
    print(dict_true)
    print(dict_false)

    for image in json_data['Images']:
        img_out = cv.imread(path + image['ImageName'])
        img = cv.resize(img_out, dsize=(
            constants.CLASSIFIER_BINARY_IMG_SIZE[1], constants.CLASSIFIER_BINARY_IMG_SIZE[0]))
        cv.imshow("test", img_out)
        cv.waitKey()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img / 255.0
        pred = model_weld.predict(np.expand_dims(img, axis=0))
        pred = np.round(pred[0][0])
        print('weld {}'.format(pred))
        if pred == 0:
            if class_in_masks(image["Masks"], 'weld'):
                dict_false['yes_weld'] = dict_false['yes_weld'] + 1
            else:
                dict_true['no_weld'] = dict_true['no_weld'] + 1
        else:
            if class_in_masks(image["Masks"], 'weld'):
                dict_true['yes_weld'] = dict_true['yes_weld'] + 1

                img = cv.resize(img_out, dsize=(
                    constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0]))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = img / 255.0
                pred = model_defect.predict(np.expand_dims(img, axis=0))
                pred = np.round(pred[0])
                print('defects {}'.format(pred))
                #####################################################
                if pred[0] == 0:
                    if class_in_masks(image["Masks"], 'glass'):
                        dict_false['glass'] = dict_false['glass'] + 1
                else:
                    if class_in_masks(image["Masks"], 'glass'):
                        dict_true['glass'] = dict_true['glass'] + 1
                    else:
                        dict_pass['glass'] = dict_pass['glass'] + 1
                #####################################################

                #####################################################
                if pred[1] == 0:
                    if class_in_masks(image["Masks"], 'burn_and_fistula') or class_in_masks(image["Masks"],
                                                                                            'pores_and_inclusions'):
                        dict_false['burn_and_fistula_pores_and_inclusions'] = dict_false[
                                                                                  'burn_and_fistula_pores_and_inclusions'] + 1
                else:
                    print('yes burn_and_fistula')
                    if class_in_masks(image["Masks"], 'burn_and_fistula') or class_in_masks(image["Masks"],
                                                                                            'pores_and_inclusions'):
                        dict_true['burn_and_fistula_pores_and_inclusions'] = dict_true[
                                                                                 'burn_and_fistula_pores_and_inclusions'] + 1
                    else:
                        dict_pass['burn_and_fistula_pores_and_inclusions'] = dict_pass[
                                                                                 'burn_and_fistula_pores_and_inclusions'] + 1
                #####################################################

                if pred[2] == 0:
                    if class_in_masks(image["Masks"], 'metal_spray'):
                        dict_false['metal_spray'] = dict_false['metal_spray'] + 1

                else:
                    if class_in_masks(image["Masks"], 'metal_spray'):
                        dict_true['metal_spray'] = dict_true['metal_spray'] + 1
                    else:
                        dict_pass['metal_spray'] = dict_pass['metal_spray'] + 1

                if pred[3] == 0:
                    if class_in_masks(image["Masks"], 'crater'):
                        dict_false['crater'] = dict_false['crater'] + 1
                else:
                    if class_in_masks(image["Masks"], 'crater'):
                        dict_true['crater'] = dict_true['crater'] + 1
                    else:
                        dict_pass['crater'] = dict_pass['crater'] + 1

                if pred[4] == 0:
                    if class_in_masks(image["Masks"], 'shell'):
                        dict_false['shell'] = dict_false['shell'] + 1
                else:
                    if class_in_masks(image["Masks"], 'shell'):
                        dict_true['shell'] = dict_true['shell'] + 1
                    else:
                        dict_pass['shell'] = dict_pass['shell'] + 1

    print(dict_true)
    print(dict_false)
    print(dict_pass)


# define decorator
def init_parameters(fun, **init_dict):
    """
    help you to set the parameters in one's habits
    """
    def job(*args, **option):
        option.update(init_dict)
        return fun(*args, **option)
    return job


def cv2_img_add_text(img, text, left_corner: Tuple[int, int],
                     text_rgb_color=(255, 0, 0), text_size=24, font='arial.ttf', **option):
    """
    USAGE:
        cv2_img_add_text(img, '中文', (0, 0), text_rgb_color=(0, 255, 0), text_size=12, font='mingliu.ttc')
    """
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img

def run2():
    path = '/home/sergei/DataSet/875_set15/IMAGES/JPEG/'
    images = os.listdir(path)
    images.sort()
    dict_defects = {0:"Шлаковые вкючения", 1:"Поры/прожог/свищ", 2:"Брызги металла", 3:"Кратер/раковина", 4:"Шов в норме"}

    model_defect = tf.keras.models.load_model(
        constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/classifier_defects0.889.h5', compile=False)
    model_defect.summary()
    TEXT_SIZE = LINE_HEIGHT = 100

    cv.namedWindow('Demo', cv.WINDOW_NORMAL)
    cv.resizeWindow('Demo', 1000, 900)
    for image in images:
        img_out = cv.imread(path + image)


        img = cv.resize(img_out, dsize=(
            constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img / 255.0
        pred = model_defect.predict(np.expand_dims(img, axis=0))[0]
        pred[pred<0.6] = 0
        pred = np.round(pred)

        index = 0
        for idx in range(len(pred)):

            if pred[4]>0.6:
                img_out = cv2_img_add_text(img_out, dict_defects[4], (0, 0), text_rgb_color=(0, 255, 0), text_size=TEXT_SIZE)
                break
            if np.max(pred[0:4])<0.6:
                img_out = cv2_img_add_text(img_out, dict_defects[4], (0, 0), text_rgb_color=(0, 255, 0),
                                           text_size=TEXT_SIZE)
                break
            elif pred[idx]>0.6 and pred[4]<0.6:
                img_out = cv2_img_add_text(img_out, dict_defects[idx], (0, LINE_HEIGHT * index), text_rgb_color=(255, 0, 0),
                                           text_size=TEXT_SIZE)
                index = index +1

        cv.imshow("Demo", img_out)
        print('defects {}'.format(pred))
        cv.waitKey(1)
        time.sleep(0.5)




if __name__ == '__main__':
    run2()
    '''model_defect = tf.keras.models.load_model(
        constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/precision_classifier_defects0.776.h5', compile=False)
    model_defect.summary()

    jsons = statistics.get_jsons()
    images, labels, counter = load_data.get_marking_balanced_dataset_cast(jsons)
    print(images)
    cv.namedWindow('test', cv.WINDOW_NORMAL)

    for i in range(len(images['crater_shell'])):
        img_out = cv.imread(images['crater_shell'][i])
        img = cv.resize(img_out, dsize=(
            constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img / 255.0
        pred = model_defect.predict(np.expand_dims(img, axis=0))
        pred = np.squeeze(pred)
        print(pred)
        if pred[4] > 0.5:
            print("Шов без дефектов")
        else:
            for j in range(4):
                if pred[j]>0.7:
                    print(constants.CLASSIFIER_MULTI_LABEL_CLASSES[j])

        print()
        print('_________________________________')
        cv.imshow('test', img_out)
        cv.waitKey()'''
