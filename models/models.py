'''
Модуль структуры моделей CNN
Автор: Сергей Сисюкин
e-mail: sergei.sisyukin@gmail.com

Датасет представляет собой шлавную директорию "DataSet", в которой расположены
директории с множестом сетов Set1...Setn
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_hub as hub
from tools import constants
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121



# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu', name='inception_module_conv1')(layer_in)
    # 3x3 conv
    conv3 = layers.Conv2D(f2_in, (1, 1), padding='same', activation='relu', name='inception_module_conv2')(layer_in)
    conv3 = layers.Conv2D(f2_out, (3, 3), padding='same', activation='relu', name='inception_module_conv3')(conv3)
    # 5x5 conv
    conv5 = layers.Conv2D(f3_in, (1,1), padding='same', activation='relu', name='inception_module_conv4')(layer_in)
    conv5 = layers.Conv2D(f3_out, (5, 5), padding='same', activation='relu', name='inception_module_conv5')(conv5)
    # 3x3 max pooling
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_module_max_pool')(layer_in)
    pool = layers.Conv2D(f4_out, (1, 1), padding='same', activation='relu', name='inception_module_conv6')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def get_pretrain_model_VGG16():

    base_model = VGG16(input_shape=(constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1], 3), include_top=False)
    for layer in base_model.layers:
        if(layer.name == 'block5_conv3' or layer.name == 'block5_conv2'):
            layer.trainable = True
        else:
            layer.trainable = False

    x = base_model.output
    x = layers.Flatten()(x)
    predictions = layers.Dense(7, activation=tf.nn.sigmoid)(x)
    model_final = Model(base_model.input, predictions, name='classifier_weld_model')
    return model_final



def get_model_classifier(shape = None):

    Input = layers.Input(shape=shape)

    layer = layers.Conv2D(64, [7, 7], padding='same', activation=tf.nn.relu)(Input)
    layer = layers.Dropout(0.1)(layer)
    pool_start = layers.MaxPooling2D(pool_size=(3, 3))(layer)

    layer = layers.Conv2D(32, [3, 3], padding='same', activation=tf.nn.relu)(pool_start)
    layer = layers.Conv2D(192, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(3, 3))(layer)

    incep_module = inception_module(layer, 64, 96, 128, 16, 32, 32)
    incep_module = layers.Dropout(0.1)(incep_module)

    layer_concat = layers.Conv2D(128, [3, 3], padding='same', activation=tf.nn.relu)(pool_start)
    layer_concat = layers.MaxPooling2D(pool_size=(3, 3))(layer_concat)
    layer_concat = layers.Conv2D(256, [3, 3], padding='same', activation=tf.nn.relu)(layer_concat)

    avg = layers.Average()([incep_module, layer_concat])

    layer_fin = layers.Dropout(0.15)(avg)
    layer_fin = layers.MaxPooling2D(pool_size=(2, 2))(layer_fin)
    layer_fin = layers.Conv2D(256, [1, 1], padding='valid', activation=tf.nn.relu)(layer_fin)
    layer_fin = layers.Conv2D(512, [3, 3], padding='valid', activation=tf.nn.relu)(layer_fin)


    flatten = layers.Flatten()(layer_fin)
    layer = layers.Dropout(0.3)(flatten)
    layer = layers.Dense(1, activation=tf.nn.sigmoid)(layer)

    

    return Model(Input, layer)


def get_model_multi_label_classifier(shape=None):
    Input = layers.Input(shape=shape)

    layer = layers.Conv2D(16, [5, 5], padding='same', activation=tf.nn.relu, name='C1')(Input)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool1')(layer)
    layer = layers.Dropout(0.1, name='drop1')(layer)


    pool_start = layers.MaxPooling2D(pool_size=(3, 3), name='branch_pool')(layer)

    layer = layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu, name='inception_branch_conv1')(pool_start)
    layer = layers.Dropout(0.3, name='drop3')(layer)

    incep_module = inception_module(layer, 32, 32, 32, 16, 32, 32)
    incep_module = layers.Dropout(0.1, name='inception_branch_drop1')(incep_module)

    layer_concat = layers.Conv2D(128, [3, 3], padding='same', activation=tf.nn.relu, name='resnet_branch_conv1')(pool_start)
    avg = layers.Average()([incep_module, layer_concat])


    layer = layers.Dropout(0.25)(avg)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = layers.Conv2D(156, [3, 3], padding='same', activation=tf.nn.relu)(layer)

    layer = layers.Dropout(0.25)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = layers.Conv2D(256, [3, 3], padding='same', activation=tf.nn.relu)(layer)

    layer = layers.Dropout(0.25)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = layers.Conv2D(356, [3, 3], padding='same', activation=tf.nn.relu)(layer)

    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.3)(flatten)
    #layer = layers.Dense(14, activation=tf.nn.relu)(layer)
    layer = layers.Dense(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES), activation=tf.nn.sigmoid)(layer)

    return Model(Input, layer)