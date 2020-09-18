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
import time


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
load dataset for training classifier "weld or no weld"
'''
def load_data_set_classifier_weld():

    print('Start load data info')
    list_NO_weld = [constants.PATH_CLASSIFIER_NO_WELD + file for file in os.listdir(constants.PATH_CLASSIFIER_NO_WELD)]
    list_YES_weld = make_list_yes_weld()


    #print(list_YES_weld)
