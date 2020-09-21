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
                                list_YES_weld.append(os.path.splitext(dir + "/IMAGES/original/" + image['ImageName'])[0] + '.npy')
                            elif  mode == 'JPEG':
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
    print('train_pos: {}, train_neg: {}; all_train: {}'.format(len(train_pos), len(train_neg), len(train_neg) + len(train_neg)))

    validation_neg = [file for file in list_NO_weld if file not in train_neg]
    validation_pos = [file for file in list_YES_weld if file not in train_pos]
    validation = validation_neg + validation_pos
    print('validation_pos: {}, validation_neg: {}; all_validation: {}'.format(len(validation_pos), len(validation_neg), len(validation)))

    return train_pos, train_neg, validation_pos, validation_neg



'''
load dataset for training classifier "weld or no weld"
'''
def load_data_set_classifier_weld(split_size=0.8, seed = 1):

    print('Start load data info')
    list_NO_weld = [constants.PATH_CLASSIFIER_NO_WELD + file for file in os.listdir(constants.PATH_CLASSIFIER_NO_WELD)]
    list_YES_weld = make_list_yes_weld()
    train_pos, train_neg, validation_pos, validation_neg = split_dataset_classifier_weld(list_NO_weld, list_YES_weld, split_size, seed)

    #print(list_YES_weld)

