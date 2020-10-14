'''
Модуль загрузки и предварительной обработки данных и аугментации
Автор: Сергей Сисюкин
e-mail: sergei.sisyukin@gmail.com

Датасет представляет собой шлавную директорию "DataSet", в которой расположены
директории с множестом сетов Set1...Setn
'''
import tensorflow as tf
from tools import constants
import os
import json
import numpy as np
import random
import tensorflow_addons as tfa
import cv2 as cv

resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(constants.CLASSIFIER_BINARY_IMG_SIZE[0],
                                                            constants.CLASSIFIER_BINARY_IMG_SIZE[1]),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / constants.CLASSIFIER_BINARY_NORNALIZE)])

data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),

        ]) #tf.keras.layers.experimental.preprocessing.RandomContrast((0.1, 0.7))

data_augmentation_neg = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),

        ])

'''
Функция проходит по всем джейсонам и выбирает 
изображения на котрых есть шов 
'''
def make_list_yes_weld(mode = 'original'):
    list_DIR_YES_weld = os.listdir(constants.PATH_DATASET)
    list_DIR_YES_weld.remove('Set_no_weld')
    list_DIR_YES_weld = [constants.PATH_DATASET + dir for dir in list_DIR_YES_weld]
    list_YES_weld = []


    for dir in  list_DIR_YES_weld:
        json_file = os.listdir(dir + "/JSON/")

        if len(json_file) == 1:
            with open(dir + "/JSON/" + json_file[0], 'r') as j:
                json_data = json.load(j)

                for image in json_data['Images']:
                    for mask in image["Masks"]:
                        if mask['class_name'] == 'weld':
                            if mode == 'original':
                                path = os.path.splitext(dir + "/IMAGES/original/" + image['ImageName'])[0] + '.npy'
                                st_size = os.stat(path).st_size
                                if st_size >= 30081152:
                                    list_YES_weld.append(path)

                            elif  mode == 'JPEG':
                                path = dir + "/IMAGES/JPEG/" + image['ImageName']
                                if os.path.isfile(path):
                                    list_YES_weld.append(dir + "/IMAGES/JPEG/" + image['ImageName'])

    return list_YES_weld

'''
функция разделяет списки данных на:
train_neg - список тренировочных изображений на которых нет шва
train_pos - список тренировочных изображений на которых есть шов
validation_pos - список валидацилнных изображений на которых есть шов
validation_neg - список валидацилнных изображений на которых нет шва
В соответствии с пропорцией split_size
Делается для того чтобы сбалансровать подачу данных в модель при обучении
'''
def split_dataset_classifier_weld(list_NO_weld, list_YES_weld, split_size=0.8, seed = 1):

    random_obj = random.Random(seed)

    train_neg = random_obj.sample(list_NO_weld, int(len(list_NO_weld)*split_size))
    train_pos = random_obj.sample(list_YES_weld, int(len(list_YES_weld) * split_size))
    print('train_pos: {}, train_neg: {}; all_train: {}'.format(len(train_pos), len(train_neg), len(train_pos) + len(train_neg)))

    validation_neg = [file for file in list_NO_weld if file not in train_neg]
    validation_pos = [file for file in list_YES_weld if file not in train_pos]
    validation = validation_neg + validation_pos
    print('validation_pos: {}, validation_neg: {}; all_validation: {}'.format(len(validation_pos), len(validation_neg), len(validation)))

    return train_pos, train_neg, validation_pos, validation_neg



'''
load npy files
'''
def read_npy_file(file):
    data = np.load(file.numpy())
    return data.astype(dtype=np.float32)

'''
Функция формирования единицы данных со швом для tf.DataSet(.jpg)
'''
def parse_pos_images_jpg_train(filename):

    label = tf.constant([1], dtype=tf.float32)

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, constants.CLASSIFIER_BINARY_IMG_SIZE)

    image = tf.expand_dims(image, 0)
    image = data_augmentation(image)
    image = tf.squeeze(image, axis=0)
    #image = tf.clip_by_value(image, 0.0, 1.0)

    #Augmen
    image = tf.image.random_hue(image, 0.3)
    image = tf.image.random_saturation(image, 0.1, 1.2)
    image = tf.image.random_brightness(image, 0.1) #0.05 [i-0.8, i+0.8]
    image = tf.clip_by_value(image, 0.0, 1.0)




    return image, label

'''
Функция формирования единицы данных со швом для tf.DataSet(.npy)
'''
def parse_pos_images_npy_train(filemame):

    label = tf.constant([1], dtype=tf.float32)
    image = tf.py_function(read_npy_file, [filemame], [tf.float32])
    image = tf.ensure_shape(image, [1, constants.SHAPE_ORIGIN_IMAGE[0], constants.SHAPE_ORIGIN_IMAGE[1], constants.SHAPE_ORIGIN_IMAGE[2]])
    image = resize_and_rescale(image)
    image = data_augmentation(image)
    image = tf.keras.backend.squeeze(image, axis=0)

    noise = tf.random.normal(image.shape, mean=constants.CLASSIFIER_BINARY_AUGMENTATION_NOISE_MEAN,
                             stddev=constants.CLASSIFIER_BINARY_AUGMENTATION_NOISE_STDEV)

    image = image + noise



    return image, label

def parse_pos_images_jpg_validation(filename):

    label = tf.constant([1], dtype=tf.float32)

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, constants.CLASSIFIER_BINARY_IMG_SIZE)

    return image, label

def parse_pos_images_npy_validation(filemame):

    label = tf.constant([1], dtype=tf.float32)
    image = tf.py_function(read_npy_file, [filemame], [tf.float32])
    image = tf.ensure_shape(image, [1, constants.SHAPE_ORIGIN_IMAGE[0], constants.SHAPE_ORIGIN_IMAGE[1], constants.SHAPE_ORIGIN_IMAGE[2]])
    image = resize_and_rescale(image)
    image = tf.keras.backend.squeeze(image, axis=0)
    return image, label


'''
Функция формирования единицы данных со швом для tf.DataSet(.jpg)
'''


def parse_neg_images_jpg_train(filename):
    label = tf.constant([0], dtype=tf.float32)

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, constants.CLASSIFIER_BINARY_IMG_SIZE)

    # Augment
    image = tf.expand_dims(image, 0)
    image = data_augmentation(image)
    image = tf.squeeze(image, axis=0)
    # image = tf.clip_by_value(image, 0.0, 1.0)

    # Augmen
    image = tf.image.random_hue(image, 0.3)
    image = tf.image.random_saturation(image, 0.1, 1.2)
    image = tf.image.random_brightness(image, 0.1)  # 0.05 [i-0.8, i+0.8]
    image = tf.clip_by_value(image, 0.0, 1.0)


    return image, label

'''
Функция формирования единицы данных без шва для tf.DataSet (.npy)
'''

def parse_neg_images_npy_train(filemame):
    label = tf.constant([0], dtype=tf.float32)

    image = tf.py_function(read_npy_file, [filemame], [tf.float32])
    image = tf.ensure_shape(image, [1, constants.SHAPE_ORIGIN_IMAGE[0], constants.SHAPE_ORIGIN_IMAGE[1], constants.SHAPE_ORIGIN_IMAGE[2]])
    image = resize_and_rescale(image)
    image = data_augmentation(image)

    image = tf.keras.backend.squeeze(image, axis=0)
    noise = tf.random.normal(image.shape, mean=constants.CLASSIFIER_BINARY_AUGMENTATION_NOISE_MEAN,
                             stddev=constants.CLASSIFIER_BINARY_AUGMENTATION_NOISE_STDEV)

    image = image + noise



    return image, label


def parse_neg_images_jpg_validation(filename):

    label = tf.constant([0], dtype=tf.float32)

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, constants.CLASSIFIER_BINARY_IMG_SIZE)

    return image, label

def parse_neg_images_npy_validation(filemame):
    label = tf.constant([0], dtype=tf.float32)
    image = tf.py_function(read_npy_file, [filemame], [tf.float32])
    image = tf.ensure_shape(image, [1, constants.SHAPE_ORIGIN_IMAGE[0], constants.SHAPE_ORIGIN_IMAGE[1], constants.SHAPE_ORIGIN_IMAGE[2]])
    image = resize_and_rescale(image)
    image = tf.keras.backend.squeeze(image, axis=0)
    return image, label


'''
load dataset for training classifier_weld "weld or no weld"
'''
def load_data_set_classifier_weld(split_size=0.8, seed = 1):

    print('Start load data info')
    list_NO_weld = [constants.CLASSIFIER_BINARY_PATH_NO_WELD + file for file in os.listdir(constants.CLASSIFIER_BINARY_PATH_NO_WELD)]
    list_YES_weld = make_list_yes_weld(constants.CLASIIFIER_MODE_LOAD)


    train_pos, train_neg, validation_pos, validation_neg = split_dataset_classifier_weld(list_NO_weld, list_YES_weld, split_size, seed)


    '''
    train positive
    '''
    ds_train_pos = tf.data.Dataset.from_tensor_slices(train_pos)
    ds_train_pos = ds_train_pos.shuffle(buffer_size=len(train_pos))
    ds_train_pos = ds_train_pos.map(parse_pos_images_jpg_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    '''cv.namedWindow("img", cv.WINDOW_NORMAL)
    for element in ds_train_pos.as_numpy_iterator():
        print(element[0].max())
        print(element[0].min())
        cv.imshow('img', cv.cvtColor(element[0], cv.COLOR_RGB2BGR))
        cv.waitKey()'''

    '''
    train negative
    '''
    ds_train_neg = tf.data.Dataset.from_tensor_slices(train_neg)
    ds_train_neg = ds_train_neg.shuffle(buffer_size=len(train_neg))
    ds_train_neg = ds_train_neg.map(parse_neg_images_jpg_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    '''cv.namedWindow("img", cv.WINDOW_NORMAL)
    for element in ds_train_neg.as_numpy_iterator():
        print(element[0].max())
        print(element[0].min())
        cv.imshow('img', cv.cvtColor(element[0], cv.COLOR_RGB2BGR))
        cv.waitKey()'''


    '''
    validation positive
    '''
    ds_validation_pos = tf.data.Dataset.from_tensor_slices(validation_pos)
    ds_validation_pos = ds_validation_pos.map(parse_pos_images_jpg_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE)



    '''
    validation negative
    '''
    ds_validation_neg = tf.data.Dataset.from_tensor_slices(validation_neg)
    ds_validation_neg = ds_validation_neg.map(parse_neg_images_jpg_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    resampled_ds_train = tf.data.experimental.sample_from_datasets([ds_train_pos, ds_train_neg], weights=[0.5, 0.5])
    resampled_ds_train = resampled_ds_train.batch(constants.CLASSIFIER_BATCH_SIZE)
    resampled_ds_train.shuffle(buffer_size=len(train_neg)+len(train_pos))
    resampled_ds_train = resampled_ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    resampled_ds_train = resampled_ds_train.repeat()

    resampled_ds_validation = tf.data.experimental.sample_from_datasets([ds_validation_pos, ds_validation_neg])
    resampled_ds_validation = resampled_ds_validation.batch(constants.CLASSIFIER_BATCH_SIZE)
    resampled_ds_validation = resampled_ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

    resampled_steps_per_epoch = np.ceil( len(train_pos) / constants.CLASSIFIER_BATCH_SIZE)

    return resampled_ds_train, resampled_ds_validation, resampled_steps_per_epoch
