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
import numpy as np
import random
from tools import statistics
import json
#import cv2 as cv


data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode='constant'),
        tf.keras.layers.experimental.preprocessing.RandomContrast((0.1, 0.7))
        ]) #

data_augmentation_weld = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #tf.keras.layers.experimental.preprocessing.RandomRotation(0.02, fill_mode='constant'),

        #tf.keras.layers.experimental.preprocessing.RandomContrast((0.5, 1.2))
        ])

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
    image = data_augmentation_weld(image)
    image = tf.squeeze(image, axis=0)

    #Augmen
    #image = tf.image.random_hue(image, 0.15)
    #image = tf.image.random_saturation(image, 0.1, 1.2)
    #image = tf.image.random_contrast(image, 0.5, 1.2)
    noise = tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.05)
    image = tf.image.random_brightness(image, 0.1)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)




    return image, label


def parse_pos_images_jpg_validation(filename):

    label = tf.constant([1], dtype=tf.float32)

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, constants.CLASSIFIER_BINARY_IMG_SIZE)

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

    image = tf.expand_dims(image, 0)
    image = data_augmentation_weld(image)
    image = tf.squeeze(image, axis=0)

    # Augmen
    # image = tf.image.random_hue(image, 0.15)
    # image = tf.image.random_saturation(image, 0.1, 1.2)
    # image = tf.image.random_contrast(image, 0.5, 1.2)
    noise = tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.05)
    image = tf.image.random_brightness(image, 0.1)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)


    return image, label




def parse_neg_images_jpg_validation(filename):

    label = tf.constant([0], dtype=tf.float32)

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, constants.CLASSIFIER_BINARY_IMG_SIZE)

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
    ds_train_pos = ds_train_pos.repeat()
    ds_train_pos = ds_train_pos.shuffle(buffer_size=len(train_pos))
    ds_train_pos = ds_train_pos.map(parse_pos_images_jpg_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)



    '''
    train negative
    '''
    ds_train_neg = tf.data.Dataset.from_tensor_slices(train_neg)
    ds_train_neg = ds_train_neg.repeat()
    ds_train_neg = ds_train_neg.shuffle(buffer_size=len(train_neg))
    ds_train_neg = ds_train_neg.map(parse_neg_images_jpg_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)



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

    resampled_ds_train = tf.data.experimental.sample_from_datasets([ds_train_pos, ds_train_neg], weights=[0.6, 0.4])
    resampled_ds_train = resampled_ds_train.batch(constants.CLASSIFIER_BATCH_SIZE)
    resampled_ds_train = resampled_ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    resampled_ds_validation = tf.data.experimental.sample_from_datasets([ds_validation_pos, ds_validation_neg])
    resampled_ds_validation = resampled_ds_validation.batch(constants.CLASSIFIER_BATCH_SIZE)
    resampled_ds_validation = resampled_ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

    resampled_steps_per_epoch = np.ceil(len(train_pos) / constants.CLASSIFIER_BATCH_SIZE)

    '''cv.namedWindow('test', flags=cv.WINDOW_NORMAL)
    for element in resampled_ds_train.as_numpy_iterator():

            img, label = element
            print(img[0].shape)
            for i in range(len(img)):
                cv.imshow('test', cv.cvtColor(img[i], code=cv.COLOR_RGB2BGR))
                cv.waitKey()'''

    return resampled_ds_train, resampled_ds_validation, resampled_steps_per_epoch

def walk_up_folder(path, depth=1):
    _cur_depth = 1
    while _cur_depth < depth:
        path = os.path.dirname(path)
        _cur_depth += 1
    return path

def make_label(masks):

    label = np.zeros(shape=(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES)))

    defects = [mask['class_name'] for mask in masks if mask['class_name'] in constants.CLASSIFIER_MULTI_LABEL_CLASSES]
    defects = list(set(defects))
    if len(defects) == 0:
        if 'background' in constants.CLASSIFIER_MULTI_LABEL_CLASSES:
            defects.append('background')
        else:
            print('List defects for make label is empty')
            exit()

    for defect in defects:
        label[constants.CLASSIFIER_MULTI_LABEL_CLASSES.index(defect)] = 1.0


    return label, defects


'''
Функция бегает по json-ам собирает пути до изображений, создает лейблы 
и считает изображения с определенным типом дефекта
'''
def get_marking(jsons):

    images = []
    labels = []
    counter = {}

    for defect in constants.CLASSIFIER_MULTI_LABEL_CLASSES:
        counter[defect] = 0

    for file in jsons:
         with open(file, 'r') as j:
            json_data = json.load(j)

            path_set = walk_up_folder(os.path.split(file)[0], 2)
            for image in json_data['Images']:
                path_img = path_set + '/IMAGES/JPEG/' + image['ImageName']

                if os.path.isfile(path_img):
                    if len(image["Masks"]) > 0:
                        label, defects = make_label(image["Masks"])
                        images.append(path_img)
                        labels.append(label)

                        for defect in defects:
                            counter[defect] +=1

                    else:
                        print('Masks images({}) is empty'.format(path_img))

    return images, labels, counter


'''
Стратифицированное разделение данных на тренировочные и валидационные
'''
def split_strat_defects(list_images, labels, split_size, seed):

    dict_defect_idx = {}
    train_images = []
    validation_images = []
    train_labels = []
    validation_labels = []

    random_obj = random.Random(seed)


    for image, label in zip(list_images, labels):
        defects = [constants.CLASSIFIER_MULTI_LABEL_CLASSES[i] for i in range(len(label)) if label[i]==1]
        defects.sort()
        name_defect = '_'.join(defects)
        if name_defect in dict_defect_idx:
            dict_defect_idx[name_defect].append(list_images.index(image))
        else:
            dict_defect_idx[name_defect] = []
            dict_defect_idx[name_defect].append(list_images.index(image))


    for key, value in dict_defect_idx.items():
        print('{} : {}'.format(key, len(value)))
        if len(value)<10:
            if len(value) < 3:
                train_images.extend([list_images[idx] for idx in value])
                train_labels.extend([labels[idx] for idx in value])

            elif len(value) == 3:
                tr_list = random_obj.sample(value, 2)
                val_list = [val for val in value if val not in tr_list]

                train_images.extend([list_images[idx] for idx in tr_list])
                train_labels.extend([labels[idx] for idx in tr_list])

                validation_images.extend([list_images[idx] for idx in val_list])
                validation_labels.extend([labels[idx] for idx in val_list])

            elif len(value) == 4:
                tr_list = random_obj.sample(value, 3)
                val_list = [val for val in value if val not in tr_list]

                train_images.extend([list_images[idx] for idx in tr_list])
                train_labels.extend([labels[idx] for idx in tr_list])

                validation_images.extend([list_images[idx] for idx in val_list])
                validation_labels.extend([labels[idx] for idx in val_list])


            elif len(value) > 4:
                tr_list = random_obj.sample(value, len(value)-2)
                val_list = [val for val in value if val not in tr_list]

                train_images.extend([list_images[idx] for idx in tr_list])
                train_labels.extend([labels[idx] for idx in tr_list])

                validation_images.extend([list_images[idx] for idx in val_list])
                validation_labels.extend([labels[idx] for idx in val_list])
        else:
            tr_list = random_obj.sample(value, int(len(value) * split_size))
            val_list = [val for val in value if val not in tr_list]

            train_images.extend([list_images[idx] for idx in tr_list])
            train_labels.extend([labels[idx] for idx in tr_list])

            validation_images.extend([list_images[idx] for idx in val_list])
            validation_labels.extend([labels[idx] for idx in val_list])


    return train_images, train_labels, validation_images, validation_labels



def parse_multi_label_train(filename, label):


    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    #image = tf.image.resize(image, constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE)

    image = tf.image.random_jpeg_quality(image, 20, 100)
    #image = tf.expand_dims(image, 0)
    #image = data_augmentation_weld(image)
    #image = tf.squeeze(image, axis=0)

    # Augmen
    #image = tf.image.random_hue(image, 0.15)
    #image = tf.image.random_saturation(image, 0.1, 1.2)
    #image = tf.image.random_contrast(image, 0.7, 0.9)
    noise = tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.05)

    image = tf.image.random_brightness(image, 0.1)
    image = image + noise
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def parse_multi_label_validation(filename, label):


    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    #image = tf.image.resize(image, constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE)

    return image, label


def get_marking_balanced_dataset(jsons):

    images = {}
    labels = {}
    counter = {}

    for defect in constants.CLASSIFIER_MULTI_LABEL_CLASSES:
        counter[defect] = 0
        images[defect] = []
        labels[defect] = []

    for file in jsons:
         with open(file, 'r') as j:
            json_data = json.load(j)

            path_set = walk_up_folder(os.path.split(file)[0], 2)
            for image in json_data['Images']:
                path_img = path_set + '/IMAGES/JPEG/' + image['ImageName']

                if os.path.isfile(path_img):
                    if len(image["Masks"]) > 0:
                        label, defects = make_label(image["Masks"])

                        for defect in defects:
                            counter[defect] +=1
                            images[defect].append(path_img)
                            labels[defect].append(label)

                    else:
                        print('Masks images({}) is empty'.format(path_img))

    return images, labels, counter

'''
load dataset for training classifier_defects
'''
def load_data_set_classifier_defects(split_size=0.9, seed = 1):

    jsons = statistics.get_jsons()
    images, labels, counter = get_marking(jsons)
    print(counter)
    train_images, train_labels, validation_images, validation_labels = split_strat_defects(images, labels, split_size, seed)

    ################# train ds #################
    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds_train = ds_train.shuffle(buffer_size=len(train_images))
    ds_train = ds_train.map(parse_multi_label_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.batch(constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()
    ############################################

    #####################validation ds###########
    ds_validation = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    ds_validation = ds_validation.map(parse_multi_label_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_validation = ds_validation.batch(constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)
    ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

    steps_per_epoch = np.ceil(len(train_images) / constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)

    '''cv.namedWindow('test', flags=cv.WINDOW_NORMAL)
    for element in ds_validation.as_numpy_iterator():
            img, label = element
            for i in range(len(img)):
                cv.imshow('test', cv.cvtColor(img[i], code=cv.COLOR_RGB2BGR))
                cv.waitKey()'''

    return ds_train, ds_validation, steps_per_epoch

'''
ЖЕСТКаЕ КАСТОМИЗАЗВЦИЯ ПО КЛАССАМ!!!
'''
def split_balanced_defects(dict_images, dict_labels, split_size, seed):
    random_obj = random.Random(seed)

    #-----------------------GLASS-------------------------------------------------
    list_idx = list(range(len(dict_images['glass'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    glass_images_train = [dict_images['glass'][i] for i in list_idx_train]
    glass_labels_train = [dict_labels['glass'][i] for i in list_idx_train]
    glass_images_validation = [dict_images['glass'][i] for i in list_idx_val]
    glass_labels_validation = [dict_labels['glass'][i] for i in list_idx_val]
    #-------------------------------------------------------------------------------

    # -----------------------burn_and_fistula---------------------------------------
    list_idx = list(range(len(dict_images['burn_and_fistula'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    burn_and_fistula_images_train = [dict_images['burn_and_fistula'][i] for i in list_idx_train]
    burn_and_fistula_labels_train = [dict_labels['burn_and_fistula'][i] for i in list_idx_train]
    burn_and_fistula_images_validation = [dict_images['burn_and_fistula'][i] for i in list_idx_val]
    burn_and_fistula_labels_validation = [dict_labels['burn_and_fistula'][i] for i in list_idx_val]
    # -------------------------------------------------------------------------------

    # -----------------------metal_spray---------------------------------------------
    list_idx = list(range(len(dict_images['metal_spray'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    metal_spray_images_train = [dict_images['metal_spray'][i] for i in list_idx_train]
    metal_spray_labels_train = [dict_labels['metal_spray'][i] for i in list_idx_train]
    metal_spray_images_validation = [dict_images['metal_spray'][i] for i in list_idx_val]
    metal_spray_labels_validation = [dict_labels['metal_spray'][i] for i in list_idx_val]
    # -------------------------------------------------------------------------------

    # -----------------------pores_and_inclusions------------------------------------
    list_idx = list(range(len(dict_images['pores_and_inclusions'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    pores_and_inclusions_images_train = [dict_images['pores_and_inclusions'][i] for i in list_idx_train]
    pores_and_inclusions_labels_train = [dict_labels['pores_and_inclusions'][i] for i in list_idx_train]
    pores_and_inclusions_images_validation = [dict_images['pores_and_inclusions'][i] for i in list_idx_val]
    pores_and_inclusions_labels_validation = [dict_labels['pores_and_inclusions'][i] for i in list_idx_val]
    # -------------------------------------------------------------------------------

    # -----------------------crater--------------------------------------------------
    list_idx = list(range(len(dict_images['crater'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    crater_images_train = [dict_images['crater'][i] for i in list_idx_train]
    crater_labels_train = [dict_labels['crater'][i] for i in list_idx_train]
    crater_images_validation = [dict_images['crater'][i] for i in list_idx_val]
    crater_labels_validation = [dict_labels['crater'][i] for i in list_idx_val]
    # -------------------------------------------------------------------------------

    # -----------------------shell--------------------------------------------------
    list_idx = list(range(len(dict_images['shell'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    shell_images_train = [dict_images['shell'][i] for i in list_idx_train]
    shell_labels_train = [dict_labels['shell'][i] for i in list_idx_train]
    shell_images_validation = [dict_images['shell'][i] for i in list_idx_val]
    shell_labels_validation = [dict_labels['shell'][i] for i in list_idx_val]
    # -------------------------------------------------------------------------------

    # -----------------------background----------------------------------------------
    list_idx = list(range(len(dict_images['background'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    background_images_train = [dict_images['background'][i] for i in list_idx_train]
    background_labels_train = [dict_labels['background'][i] for i in list_idx_train]
    background_images_validation = [dict_images['background'][i] for i in list_idx_val]
    background_labels_validation = [dict_labels['background'][i] for i in list_idx_val]
    # -------------------------------------------------------------------------------

    return glass_images_train, glass_labels_train, glass_images_validation, glass_labels_validation,\
           burn_and_fistula_images_train, burn_and_fistula_labels_train, burn_and_fistula_images_validation, burn_and_fistula_labels_validation, \
           metal_spray_images_train, metal_spray_labels_train, metal_spray_images_validation, metal_spray_labels_validation, \
           pores_and_inclusions_images_train, pores_and_inclusions_labels_train, pores_and_inclusions_images_validation, pores_and_inclusions_labels_validation, \
           crater_images_train, crater_labels_train, crater_images_validation, crater_labels_validation, \
           shell_images_train, shell_labels_train, shell_images_validation, shell_labels_validation, \
           background_images_train, background_labels_train, background_images_validation, background_labels_validation

def make_ds_val(images, labels):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    #ds = ds.repeat()
    ds = ds.map(parse_multi_label_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

def make_ds(images, labels):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))


    ds = ds.shuffle(buffer_size=constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)
    ds = ds.repeat()
    ds = ds.map(parse_multi_label_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #ds.cache()

    return ds

def load_data_set_balanced_classifier_defects(split_size=0.9, seed=1):

    jsons = statistics.get_jsons()
    images, labels, counter = get_marking_balanced_dataset(jsons)
    print(counter)
    glass_images_train, glass_labels_train, glass_images_validation, glass_labels_validation, \
    burn_and_fistula_images_train, burn_and_fistula_labels_train, burn_and_fistula_images_validation, burn_and_fistula_labels_validation, \
    metal_spray_images_train, metal_spray_labels_train, metal_spray_images_validation, metal_spray_labels_validation, \
    pores_and_inclusions_images_train, pores_and_inclusions_labels_train, pores_and_inclusions_images_validation, pores_and_inclusions_labels_validation, \
    crater_images_train, crater_labels_train, crater_images_validation, crater_labels_validation, \
    shell_images_train, shell_labels_train, shell_images_validation, shell_labels_validation, \
    background_images_train, background_labels_train, background_images_validation, background_labels_validation = split_balanced_defects(images, labels, split_size, seed)

    ds_train_glass = make_ds(glass_images_train, glass_labels_train)
    ds_val_glass = make_ds_val(glass_images_validation, glass_labels_validation)

    ds_train_burn_and_fistula = make_ds(burn_and_fistula_images_train, burn_and_fistula_labels_train)
    ds_val_burn_and_fistula = make_ds_val(burn_and_fistula_images_validation, burn_and_fistula_labels_validation)

    ds_train_metal_spray = make_ds(metal_spray_images_train, metal_spray_labels_train)
    ds_val_metal_spray = make_ds_val(metal_spray_images_validation, metal_spray_labels_validation)

    ds_train_pores_and_inclusions = make_ds(pores_and_inclusions_images_train, pores_and_inclusions_labels_train)
    ds_val_pores_and_inclusions = make_ds_val(pores_and_inclusions_images_validation, pores_and_inclusions_labels_validation)

    ds_train_crater = make_ds(crater_images_train, crater_labels_train)
    ds_val_crater = make_ds_val(crater_images_validation, crater_labels_validation)

    ds_train_shell = make_ds(shell_images_train, shell_labels_train)
    ds_val_shell = make_ds_val(shell_images_validation, shell_labels_validation)

    ds_train_background = make_ds(background_images_train, background_labels_train)
    ds_val_background = make_ds_val(background_images_validation, background_labels_validation)


    ################# train ds #################
    resampled_ds_train = tf.data.experimental.sample_from_datasets([ds_train_glass, ds_train_burn_and_fistula, ds_train_metal_spray,
                                                                    ds_train_pores_and_inclusions, ds_train_crater,
                                                                    ds_train_shell, ds_train_background], weights=[0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16])

    resampled_ds_train = resampled_ds_train.batch(constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)
    resampled_ds_train = resampled_ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    #resampled_ds_train = resampled_ds_train.repeat()
    ############################################

    #####################validation ds###########
    resampled_ds_val = tf.data.experimental.sample_from_datasets(
        [ds_val_glass, ds_val_burn_and_fistula, ds_val_metal_spray,
         ds_val_pores_and_inclusions, ds_val_crater,
         ds_val_shell, ds_val_background], weights=[0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16])

    resampled_ds_val = resampled_ds_val.batch(constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)
    resampled_ds_val = resampled_ds_val.prefetch(tf.data.experimental.AUTOTUNE)
    #resampled_ds_val = resampled_ds_val.repeat()

    min_quantity_samples = min(len(glass_images_train), len(burn_and_fistula_images_train), len(metal_spray_images_train), len(pores_and_inclusions_images_train),
                            len(crater_images_train), len(shell_images_train), len(background_images_train))

    resampled_steps_per_epoch = np.ceil(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES)*min_quantity_samples / constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)

    print(min_quantity_samples)
    '''cv.namedWindow('test', flags=cv.WINDOW_NORMAL)
    for element in resampled_ds_train.as_numpy_iterator():
            img, label = element
            for i in range(len(img)):
                print(label[i])
                cv.imshow('test', cv.cvtColor(img[i], code=cv.COLOR_RGB2BGR))
                cv.waitKey()'''

    return resampled_ds_train, resampled_ds_val, resampled_steps_per_epoch



################################################################################
def get_marking_balanced_dataset_cast(jsons):
    '''
    Формируем словари изображений и меток к ним
    :param jsons: список файлов аннотаций
    :return:
    '''
    images = {}
    labels = {}
    counter = {}

    for defect in constants.CLASSIFIER_MULTI_LABEL_CLASSES:
        counter[defect] = 0
        images[defect] = []
        labels[defect] = []

    for file in jsons:
         with open(file, 'r') as j:
            json_data = json.load(j)

            path_set = walk_up_folder(os.path.split(file)[0], 2)
            for image in json_data['Images']:
                path_img = path_set + '/IMAGES/JPEG/' + image['ImageName']

                if os.path.isfile(path_img):
                    if len(image["Masks"]) > 0:
                        #
                        label, defects = make_label_cast(image["Masks"])


                        for defect in defects:
                            counter[defect] += 1
                            images[defect].append(path_img)
                            labels[defect].append(label)


                    else:
                        print('Masks images({}) is empty'.format(path_img))

    return images, labels, counter

def split_balanced_defects_cast_old(dict_images, dict_labels, split_size, seed):
    random_obj = random.Random(seed)

    #-----------------------GLASS-------------------------------------------------
    list_idx = list(range(len(dict_images['glass'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]


    glass_images_train = [dict_images['glass'][i] for i in list_idx_train]
    glass_labels_train = [dict_labels['glass'][i] for i in list_idx_train]
    glass_images_validation = [dict_images['glass'][i] for i in list_idx_val]
    glass_labels_validation = [dict_labels['glass'][i] for i in list_idx_val]

    print("______________________")
    print("TRAIN GLASS: {}".format(len(glass_images_train)))
    print("VALIDATION GLASS: {}".format(len(glass_images_validation)))
    print("______________________")
    #-------------------------------------------------------------------------------

    # -----------------------burn_and_fistula---------------------------------------
    list_idx = list(range(len(dict_images['burn_and_fistula_pores_and_inclusions'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    burn_and_fistula_images_train = [dict_images['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_train]
    burn_and_fistula_labels_train = [dict_labels['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_train]
    burn_and_fistula_images_validation = [dict_images['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_val]
    burn_and_fistula_labels_validation = [dict_labels['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN PORES: {}".format(len(burn_and_fistula_images_train)))
    print("VALIDATION PORES: {}".format(len(burn_and_fistula_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------

    # -----------------------metal_spray---------------------------------------------
    list_idx = list(range(len(dict_images['metal_spray'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    metal_spray_images_train = [dict_images['metal_spray'][i] for i in list_idx_train]
    metal_spray_labels_train = [dict_labels['metal_spray'][i] for i in list_idx_train]
    metal_spray_images_validation = [dict_images['metal_spray'][i] for i in list_idx_val]
    metal_spray_labels_validation = [dict_labels['metal_spray'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN SPRAY: {}".format(len(metal_spray_images_train)))
    print("VALIDATION SPRAY: {}".format(len(metal_spray_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------

    # -----------------------crater--------------------------------------------------
    list_idx = list(range(len(dict_images['crater_shell'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    crater_shell_images_train = [dict_images['crater_shell'][i] for i in list_idx_train]
    crater_shell_labels_train = [dict_labels['crater_shell'][i] for i in list_idx_train]
    crater_shell_images_validation = [dict_images['crater_shell'][i] for i in list_idx_val]
    crater_shell_labels_validation = [dict_labels['crater_shell'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN CRATER: {}".format(len(crater_shell_images_train)))
    print("VALIDATION CRATER: {}".format(len(crater_shell_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------

    '''# -----------------------shell--------------------------------------------------
    list_idx = list(range(len(dict_images['shell'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    shell_images_train = [dict_images['shell'][i] for i in list_idx_train]
    shell_labels_train = [dict_labels['shell'][i] for i in list_idx_train]
    shell_images_validation = [dict_images['shell'][i] for i in list_idx_val]
    shell_labels_validation = [dict_labels['shell'][i] for i in list_idx_val]'''
    # -------------------------------------------------------------------------------

    # -----------------------background----------------------------------------------
    list_idx = list(range(len(dict_images['background'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * 0.95)) ###!!!!!!!!!!!!!!!!!!изменен порог!
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    background_images_train = [dict_images['background'][i] for i in list_idx_train]
    background_labels_train = [dict_labels['background'][i] for i in list_idx_train]
    background_images_validation = [dict_images['background'][i] for i in list_idx_val]
    background_labels_validation = [dict_labels['background'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN BACKGROUND: {}".format(len(background_images_train)))
    print("VALIDATION BACKGROUND: {}".format(len(background_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------

    return glass_images_train, glass_labels_train, glass_images_validation, glass_labels_validation,\
           burn_and_fistula_images_train, burn_and_fistula_labels_train, burn_and_fistula_images_validation, burn_and_fistula_labels_validation, \
           metal_spray_images_train, metal_spray_labels_train, metal_spray_images_validation, metal_spray_labels_validation, \
           crater_shell_images_train, crater_shell_labels_train, crater_shell_images_validation, crater_shell_labels_validation, \
           background_images_train, background_labels_train, background_images_validation, background_labels_validation


def split_balanced_defects_cast(dict_images, dict_labels, split_size, seed):
    '''
    Делим всю дату на тренировочный и валидационный набор. При расширении классов здесь тоже нужно
    изменить. Делаем непересикающиеся выборки
    :param dict_images: словарь изображений по типу дефектов
    :param dict_labels: соответствующие им метки
    :param split_size: размер среза
    :param seed: начальное значения рандома
    :return:
    '''
    random_obj = random.Random(seed)

    count_glass_val = len(dict_images['glass']) - int(len(dict_images['glass'])*split_size)
    count_burn_and_fistula_val = len(dict_images['burn_and_fistula_pores_and_inclusions']) - int(len(dict_images['burn_and_fistula_pores_and_inclusions'])*split_size)
    count_metal_spray_val = len(dict_images['metal_spray']) - int(len(dict_images['metal_spray'])*split_size)
    count_crater_shell_val = len(dict_images['crater_shell']) - int(len(dict_images['crater_shell'])*split_size)

    # -----------------------metal_spray---------------------------------------------
    list_idx = list(range(len(dict_images['metal_spray'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    metal_spray_images_train = [dict_images['metal_spray'][i] for i in list_idx_train]
    metal_spray_labels_train = [dict_labels['metal_spray'][i] for i in list_idx_train]
    metal_spray_images_validation = [dict_images['metal_spray'][i] for i in list_idx_val]
    metal_spray_labels_validation = [dict_labels['metal_spray'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN SPRAY: {}".format(len(metal_spray_images_train)))
    print("VALIDATION SPRAY: {}".format(len(metal_spray_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------

    #____________________________CLEAR_______________________________________________
    for sample in metal_spray_images_train:
        if sample in dict_images['crater_shell']:
            idx = dict_images['crater_shell'].index(sample)
            dict_images['crater_shell'].pop(idx)
            dict_labels['crater_shell'].pop(idx)
            print("delete crater_shell")

        if sample in dict_images['glass']:
            idx = dict_images['glass'].index(sample)
            dict_images['glass'].pop(idx)
            dict_labels['glass'].pop(idx)
            print("delete glass")

        if sample in dict_images['burn_and_fistula_pores_and_inclusions']:
            idx = dict_images['burn_and_fistula_pores_and_inclusions'].index(sample)
            dict_images['burn_and_fistula_pores_and_inclusions'].pop(idx)
            dict_labels['burn_and_fistula_pores_and_inclusions'].pop(idx)
            print("delete burn_and_fistula_pores_and_inclusions")

    for sample in metal_spray_images_validation:
        if sample in dict_images['crater_shell']:
            idx = dict_images['crater_shell'].index(sample)
            dict_images['crater_shell'].pop(idx)
            dict_labels['crater_shell'].pop(idx)
            print("delete crater_shell")

        if sample in dict_images['glass']:
            idx = dict_images['glass'].index(sample)
            dict_images['glass'].pop(idx)
            dict_labels['glass'].pop(idx)
            print("delete glass")

        if sample in dict_images['burn_and_fistula_pores_and_inclusions']:
            idx = dict_images['burn_and_fistula_pores_and_inclusions'].index(sample)
            dict_images['burn_and_fistula_pores_and_inclusions'].pop(idx)
            dict_labels['burn_and_fistula_pores_and_inclusions'].pop(idx)
            print("delete burn_and_fistula_pores_and_inclusions")
    #________________________________________________________________________________

    # -----------------------crater--------------------------------------------------
    list_idx = list(range(len(dict_images['crater_shell'])))
    list_idx_val = random_obj.sample(list_idx, count_crater_shell_val)
    list_idx_train = [val for val in list_idx if val not in list_idx_val]

    crater_shell_images_train = [dict_images['crater_shell'][i] for i in list_idx_train]
    crater_shell_labels_train = [dict_labels['crater_shell'][i] for i in list_idx_train]
    crater_shell_images_validation = [dict_images['crater_shell'][i] for i in list_idx_val]
    crater_shell_labels_validation = [dict_labels['crater_shell'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN CRATER: {}".format(len(crater_shell_images_train)))
    print("VALIDATION CRATER: {}".format(len(crater_shell_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------

    # ____________________________CLEAR_______________________________________________
    for sample in crater_shell_images_train:
        if sample in dict_images['glass']:
            idx = dict_images['glass'].index(sample)
            dict_images['glass'].pop(idx)
            dict_labels['glass'].pop(idx)
            print("delete glass")

        if sample in dict_images['burn_and_fistula_pores_and_inclusions']:
            idx = dict_images['burn_and_fistula_pores_and_inclusions'].index(sample)
            dict_images['burn_and_fistula_pores_and_inclusions'].pop(idx)
            dict_labels['burn_and_fistula_pores_and_inclusions'].pop(idx)
            print("delete burn_and_fistula_pores_and_inclusions")

    for sample in crater_shell_images_validation:
        if sample in dict_images['glass']:
            idx = dict_images['glass'].index(sample)
            dict_images['glass'].pop(idx)
            dict_labels['glass'].pop(idx)
            print("delete glass")

        if sample in dict_images['burn_and_fistula_pores_and_inclusions']:
            idx = dict_images['burn_and_fistula_pores_and_inclusions'].index(sample)
            dict_images['burn_and_fistula_pores_and_inclusions'].pop(idx)
            dict_labels['burn_and_fistula_pores_and_inclusions'].pop(idx)
            print("delete burn_and_fistula_pores_and_inclusions")
    # ________________________________________________________________________________

    #-----------------------GLASS-------------------------------------------------
    list_idx = list(range(len(dict_images['glass'])))
    list_idx_val = random_obj.sample(list_idx, count_glass_val)
    list_idx_train = [val for val in list_idx if val not in list_idx_val]


    glass_images_train = [dict_images['glass'][i] for i in list_idx_train]
    glass_labels_train = [dict_labels['glass'][i] for i in list_idx_train]
    glass_images_validation = [dict_images['glass'][i] for i in list_idx_val]
    glass_labels_validation = [dict_labels['glass'][i] for i in list_idx_val]

    print("______________________")
    print("TRAIN GLASS: {}".format(len(glass_images_train)))
    print("VALIDATION GLASS: {}".format(len(glass_images_validation)))
    print("______________________")
    #-------------------------------------------------------------------------------

    # ____________________________CLEAR_______________________________________________
    for sample in glass_images_train:

        if sample in dict_images['burn_and_fistula_pores_and_inclusions']:
            idx = dict_images['burn_and_fistula_pores_and_inclusions'].index(sample)
            dict_images['burn_and_fistula_pores_and_inclusions'].pop(idx)
            dict_labels['burn_and_fistula_pores_and_inclusions'].pop(idx)
            print("delete burn_and_fistula_pores_and_inclusions")

    for sample in glass_images_validation:
        if sample in dict_images['burn_and_fistula_pores_and_inclusions']:
            idx = dict_images['burn_and_fistula_pores_and_inclusions'].index(sample)
            dict_images['burn_and_fistula_pores_and_inclusions'].pop(idx)
            dict_labels['burn_and_fistula_pores_and_inclusions'].pop(idx)
            print("delete burn_and_fistula_pores_and_inclusions")
    # ________________________________________________________________________________

    # -----------------------burn_and_fistula---------------------------------------
    list_idx = list(range(len(dict_images['burn_and_fistula_pores_and_inclusions'])))
    list_idx_val = random_obj.sample(list_idx, count_burn_and_fistula_val)
    list_idx_train = [val for val in list_idx if val not in list_idx_val]

    burn_and_fistula_images_train = [dict_images['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_train]
    burn_and_fistula_labels_train = [dict_labels['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_train]
    burn_and_fistula_images_validation = [dict_images['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_val]
    burn_and_fistula_labels_validation = [dict_labels['burn_and_fistula_pores_and_inclusions'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN PORES: {}".format(len(burn_and_fistula_images_train)))
    print("VALIDATION PORES: {}".format(len(burn_and_fistula_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------





    '''# -----------------------shell--------------------------------------------------
    list_idx = list(range(len(dict_images['shell'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * split_size))
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    shell_images_train = [dict_images['shell'][i] for i in list_idx_train]
    shell_labels_train = [dict_labels['shell'][i] for i in list_idx_train]
    shell_images_validation = [dict_images['shell'][i] for i in list_idx_val]
    shell_labels_validation = [dict_labels['shell'][i] for i in list_idx_val]'''
    # -------------------------------------------------------------------------------

    # -----------------------background----------------------------------------------
    list_idx = list(range(len(dict_images['background'])))
    list_idx_train = random_obj.sample(list_idx, int(len(list_idx) * 0.97)) ###!!!!!!!!!!!!!!!!!!изменен порог!
    list_idx_val = [val for val in list_idx if val not in list_idx_train]

    background_images_train = [dict_images['background'][i] for i in list_idx_train]
    background_labels_train = [dict_labels['background'][i] for i in list_idx_train]
    background_images_validation = [dict_images['background'][i] for i in list_idx_val]
    background_labels_validation = [dict_labels['background'][i] for i in list_idx_val]
    print("______________________")
    print("TRAIN BACKGROUND: {}".format(len(background_images_train)))
    print("VALIDATION BACKGROUND: {}".format(len(background_images_validation)))
    print("______________________")
    # -------------------------------------------------------------------------------

    return glass_images_train, glass_labels_train, glass_images_validation, glass_labels_validation,\
           burn_and_fistula_images_train, burn_and_fistula_labels_train, burn_and_fistula_images_validation, burn_and_fistula_labels_validation, \
           metal_spray_images_train, metal_spray_labels_train, metal_spray_images_validation, metal_spray_labels_validation, \
           crater_shell_images_train, crater_shell_labels_train, crater_shell_images_validation, crater_shell_labels_validation, \
           background_images_train, background_labels_train, background_images_validation, background_labels_validation

def load_data_set_balanced_classifier_defects_cast(split_size=0.9, seed=1):
    '''
    Формируем сбалансированные тренировочный и валидационный наборы
    :param split_size:
    :param seed:
    :return:
    '''
    # Пробегаем по сетам и собираем jsons
    jsons = statistics.get_jsons()

    #формируем списки путей изображений и меток к ним
    images, labels, counter = get_marking_balanced_dataset_cast(jsons)
    print(counter)
    glass_images_train, glass_labels_train, glass_images_validation, glass_labels_validation, \
    burn_and_fistula_images_train, burn_and_fistula_labels_train, burn_and_fistula_images_validation, burn_and_fistula_labels_validation, \
    metal_spray_images_train, metal_spray_labels_train, metal_spray_images_validation, metal_spray_labels_validation, \
    crater_shell_images_train, crater_shell_labels_train, crater_shell_images_validation, crater_shell_labels_validation, \
    background_images_train, background_labels_train, background_images_validation, background_labels_validation = split_balanced_defects_cast(images, labels, split_size, seed)

    '''All_Valid_image = []
    All_Valid_label = []
'''
    ds_train_glass = make_ds(glass_images_train, glass_labels_train)


    ds_val_glass = make_ds_val(glass_images_validation, glass_labels_validation)
    '''All_Valid_image.extend(glass_images_validation)
    All_Valid_label.extend(glass_labels_validation)'''


    ds_train_burn_and_fistula = make_ds(burn_and_fistula_images_train, burn_and_fistula_labels_train)
    ds_val_burn_and_fistula = make_ds_val(burn_and_fistula_images_validation, burn_and_fistula_labels_validation)

    '''All_Valid_image.extend(burn_and_fistula_images_validation)
    All_Valid_label.extend(burn_and_fistula_labels_validation)'''

    ds_train_metal_spray = make_ds(metal_spray_images_train, metal_spray_labels_train)
    ds_val_metal_spray = make_ds_val(metal_spray_images_validation, metal_spray_labels_validation)

    '''All_Valid_image.extend(metal_spray_images_validation)
    All_Valid_label.extend(metal_spray_labels_validation)'''

    ds_train_crater_shell = make_ds(crater_shell_images_train, crater_shell_labels_train)
    ds_val_crater_shell = make_ds_val(crater_shell_images_validation, crater_shell_labels_validation)
    '''All_Valid_image.extend(crater_shell_images_validation)
    All_Valid_label.extend(crater_shell_labels_validation)'''

    #ds_train_shell = make_ds(shell_images_train, shell_labels_train)
    #ds_val_shell = make_ds_val(shell_images_validation, shell_labels_validation)

    ds_train_background = make_ds(background_images_train, background_labels_train)
    ds_val_background = make_ds_val(background_images_validation, background_labels_validation)
    '''All_Valid_image.extend(background_images_validation)
    All_Valid_label.extend(background_labels_validation)'''



    ################# train ds #################
    resampled_ds_train = tf.data.experimental.sample_from_datasets([ds_train_glass, ds_train_burn_and_fistula, ds_train_metal_spray,
                                                                     ds_train_crater_shell,
                                                                     ds_train_background], weights=[0.2, 0.2, 0.2, 0.2, 0.2 ]) #, ds_train_background


    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    resampled_ds_train = resampled_ds_train.with_options(options)

    resampled_ds_train = resampled_ds_train.batch(constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)
    #resampled_ds_train = resampled_ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    #resampled_ds_train = resampled_ds_train.repeat()
    ############################################

    #####################validation ds###########
    resampled_ds_val = tf.data.experimental.sample_from_datasets(
        [ds_val_glass, ds_val_burn_and_fistula, ds_val_metal_spray, ds_val_crater_shell,
          ds_val_background], weights=[0.2, 0.2, 0.2, 0.2, 0.2 ]) #, ds_val_background

    #resampled_ds_val = make_ds_val(All_Valid_image, All_Valid_label)
    resampled_ds_val = resampled_ds_val.with_options(options)
    resampled_ds_val = resampled_ds_val.batch(constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE) #constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE
    #resampled_ds_val = resampled_ds_val.prefetch(1)
    #resampled_ds_val.cache()
    #resampled_ds_val = resampled_ds_val.repeat()

    max_quantity_samples = min(len(glass_images_train),
                             len(background_images_train)) #, len(background_images_train)

    resampled_steps_per_epoch = np.ceil(max_quantity_samples/(constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE*0.25))
    val_steps = np.ceil(len(crater_shell_images_validation)/constants.CLASSIFIER_MULTI_LABEL_BATCH_SIZE)


    '''cv.namedWindow('test', flags=cv.WINDOW_NORMAL)
    for element in resampled_ds_train.as_numpy_iterator():
            img, label = element
            for i in range(len(img)):
                print(label[i])
                cv.imshow('test', cv.cvtColor(img[i], code=cv.COLOR_RGB2BGR))
                cv.waitKey()'''

    return resampled_ds_train, resampled_steps_per_epoch, resampled_ds_val, val_steps


def make_label_cast(masks):
    '''
    Формируем списки дефектов и метки к ним. Здесь нужно будет доработать при расширении
    на другие классы дефектов
    :param masks: маска, содержащая информацию о наличии дефектов
    :return:
    '''
    label = np.zeros(shape=(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES))) # -1

    defects = []
    for mask in masks:
        #Объеденяем прожог,свищ, поры и включения в один класс. Данных по каждому было мало, но в цлом они похожи визуально
        if mask['class_name'] == 'burn_and_fistula' or mask['class_name'] == 'pores_and_inclusions':
            defects.append('burn_and_fistula_pores_and_inclusions')
        if mask['class_name'] == 'crater' or mask['class_name'] == 'shell':
            defects.append('crater_shell')
        else:
            if mask['class_name'] in constants.CLASSIFIER_MULTI_LABEL_CLASSES:
                defects.append(mask['class_name'])

    defects = list(set(defects))
    if len(defects) == 0:
        if 'background' in constants.CLASSIFIER_MULTI_LABEL_CLASSES:
            defects.append('background')
        else:
            pass

    for defect in defects:
        #if defect != 'background':
        label[constants.CLASSIFIER_MULTI_LABEL_CLASSES.index(defect)] = 1.0


    return label, defects