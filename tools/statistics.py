'''
Модуль формирования статистики по датасету
Автор: Сергей Сисюкин
e-mail: sergei.sisyukin@gmail.com

Датасет представляет собой шлавную директорию "DataSet", в которой расположены
директории с множестом сетов Set1...Setn
'''
import os
from tools import constants
from tools import load_data
import json
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np

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

    dict_stat.pop('weld')

    return dict_stat

'''def plot_DataSet():
    jsons = get_jsons()
    images, labels, counter = load_data.get_marking_balanced_dataset_cast(jsons)
    print(counter)
    #dict_now = {'glass': 3368, 'burn_and_fistula_pores_and_inclusions': 147, 'metal_spray': 1066, 'crater_shell': 747, 'background': 7282}
    #dict_old = {'glass': 1829, 'burn_and_fistula_pores_and_inclusions': 147, 'metal_spray': 810, 'crater_shell': 699, 'background': 2111}
    сurrent = {'glass': 885, 'burn_and_fistula_pores_and_inclusions': 146, 'metal_spray': 730, 'crater_shell': 638,
     'background': 11807}

    labels = ['Шлак', 'Поры\nСвищи\nПрожоги', 'Брызги', 'Кратеры\nРаковины', 'Бездефектный шов']
    now = [885, 146, 730, 638, 11807]
    old = [503, 146, 586, 552, 5430]
    sept = [352, 146, 380, 420, 2302]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects0 = ax.bar(x - 4*width / 3, sept, width, label='14.09.2020')
    rects1 = ax.bar(x - 2*width / 3, old, width, label='1.02.2021')
    rects2 = ax.bar(x + width / 3, now, width, label='9.04.2021')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Количество собранных примеров')
    ax.set_title('Количество собранных(и размеченных) примеров дефектов и бездефектного шва')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects0)
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

def plot_stat(dict_stat):

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%'.format(p=pct, v=val)

        return my_autopct

    data_names = []
    data_values = []

    dict_stat_revers = {'weld': 'Шов', 'glass': 'Шлак', 'burn_and_fistula': "Прожог и свищ",
                        'metal_spray': "Брызги металла", 'pores_and_inclusions': "Поры и включения",
                        'cracks': "Трещены",
                        'crater': "Кратеры", 'shell': "Раковины", 'undercut': "Подрезы"}

    for key, velue in dict_stat.items():
        data_names.append(dict_stat_revers[key])
        data_values.append(velue)



    data_values, data_names = zip(*sorted(zip(data_values, data_names)))

    data_names = list(data_names)
    data_names.reverse()
    data_values = list(data_values)
    data_values.reverse()

    data_names = [data_names[i] + ' : ' + str(data_values[i]) for i in range(len(data_names))]

    dpi = 80
    fig = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
    mpl.rcParams.update({'font.size': 9})

    explode = (0.006, 0.012, 0.025, 0.05, 0.1, 0.12, 0.15, 0.5) #(0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025)
    plt.pie(data_values, labels=data_names, autopct=make_autopct(data_values), explode=explode)

    plt.legend(
        bbox_to_anchor=(-0.4, 0.8, 0.0, 0.0),
        loc='lower left', labels=data_names)

    plt.show()

if __name__ == '__main__':
    dict_true = {'glass': 302, 'burn_and_fistula_pores_and_inclusions': 0, 'metal_spray': 194, 'crater': 59, 'shell': 143, 'no_weld': 98, 'yes_weld': 1182}
    dict_false = {'glass': 10, 'burn_and_fistula_pores_and_inclusions': 0, 'metal_spray': 4, 'crater': 0, 'shell': 2, 'no_weld': 0, 'yes_weld': 66}
    dict_pass = {'glass': 27, 'burn_and_fistula_pores_and_inclusions': 1, 'metal_spray': 32, 'crater': 4, 'shell': 18}

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%'.format(p=pct, v=val)

        return my_autopct

    data_names = []
    data_values = []

    dict_stat_revers = {'glass': 'Шлак', 'burn_and_fistula_pores_and_inclusions': "Прожог\n или\n свищ\n или\n поры",
                        'metal_spray': "Брызги металла",
                        'crater': "Кратеры", 'shell': "Раковины"}

    for key, velue in dict_pass.items():
        data_names.append(dict_stat_revers[key])
        data_values.append(velue)

    print(data_values)
    print(data_names)
    data_values, data_names = zip(*sorted(zip(data_values, data_names)))

    data_names = list(data_names)
    data_names.reverse()
    data_values = list(data_values)
    data_values.reverse()


    data_names = [data_names[i] + ' : ' + str(data_values[i]) for i in range(len(data_names))]

    dpi = 80
    fig = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
    plt.title("Дефекты обнаружены нейронной сетью, но отсутствуют на снимках")
    mpl.rcParams.update({'font.size': 9})

    explode = (0.006, 0.012, 0.025, 0.05, 0.1) #(0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025)
    plt.pie(data_values, labels=data_names, autopct=make_autopct(data_values), explode=explode, labeldistance=1.1, rotatelabels = False)

    plt.legend(
        bbox_to_anchor=(-0.4, 0.8, 0.0, 0.0),
        loc='lower left', labels=data_names)

    plt.show()'''