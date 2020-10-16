'''
Модуль формирования статистики по датасету
Автор: Сергей Сисюкин
e-mail: sergei.sisyukin@gmail.com

Датасет представляет собой шлавную директорию "DataSet", в которой расположены
директории с множестом сетов Set1...Setn
'''
import os
from tools import constants
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_jsons():
    list_DIR_JSON= os.listdir(constants.PATH_DATASET)
    list_DIR_JSON.remove('Set_no_weld')
    list_DIR_JSON = [constants.PATH_DATASET + dir + '/JSON/' for dir in list_DIR_JSON]

    jsons = []
    for dir in list_DIR_JSON:
        files = os.listdir(dir)
        if len(files)>0:
            jsons.append(dir+files[0])

    return jsons

def parse_stat_json(jsons):

    dict_stat = {'weld': 0, 'glass':0, 'burn_and_fistula':0, 'metal_spray':0, 'pores_and_inclusions':0, 'cracks':0, 'crater':0, 'shell':0, 'undercut':0}

    for file in jsons:

            with open(file, 'r') as j:
                json_data = json.load(j)

                for image in json_data['Images']:
                    for mask in image["Masks"]:
                        dict_stat[mask['class_name']] += 1

    return dict_stat

def plot_stat(dict_stat):

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%'.format(p=pct, v=val)

        return my_autopct

    data_names = []
    data_values = []


    for key, velue in dict_stat.items():
        data_names.append(key)
        data_values.append(velue)

    data_values, data_names = zip(*sorted(zip(data_values, data_names)))
    data_names = [data_names[i] + ' : ' + str(data_values[i]) for i in range(len(data_names))]

    dpi = 80
    fig = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
    mpl.rcParams.update({'font.size': 9})

    explode = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01)
    plt.pie(data_values, labels=data_names, autopct=make_autopct(data_values), explode=explode)

    plt.legend(
        bbox_to_anchor=(-0.4, 0.8, 0.0, 0.0),
        loc='lower left', labels=data_names)

    plt.show()

