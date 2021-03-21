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
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out, num_mod):
    # 1x1 conv
    conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu', name='inception_module_conv1'+'_'+num_mod)(layer_in)
    # 3x3 conv
    conv3 = layers.Conv2D(f2_in, (1, 1), padding='same', activation='relu', name='inception_module_conv2'+'_'+num_mod)(layer_in)
    conv3 = layers.Conv2D(f2_out, (3, 3), padding='same', activation='relu', name='inception_module_conv3'+'_'+num_mod)(conv3)
    # 5x5 conv
    conv5 = layers.Conv2D(f3_in, (1,1), padding='same', activation='relu', name='inception_module_conv4'+'_'+num_mod)(layer_in)
    conv5 = layers.Conv2D(f3_out, (5, 5), padding='same', activation='relu', name='inception_module_conv5'+'_'+num_mod)(conv5)
    # 3x3 max pooling
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_module_max_pool'+'_'+num_mod)(layer_in)
    pool = layers.Conv2D(f4_out, (1, 1), padding='same', activation='relu', name='inception_module_conv6'+'_'+num_mod)(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def get_model_classifier_XXX(shape = None):

    Input = layers.Input(shape=shape)

    layer = layers.Conv2D(12, [5, 5], padding='same', activation=tf.nn.relu)(Input)
    pool_start = layers.MaxPooling2D(pool_size=(3, 3))(layer)

    layer = layers.Conv2D(18, [3, 3], padding='same', activation=tf.nn.relu)(pool_start)
    layer = layers.Conv2D(24, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)

    incep_module = inception_module(layer, 8, 8, 16, 8, 16, 16, 'mod1')
    incep_module = layers.Dropout(0.1)(incep_module)

    layer_concat = layers.Conv2D(32, [3, 3], padding='same', activation=tf.nn.relu)(pool_start)
    layer_concat = layers.MaxPooling2D(pool_size=(2, 2))(layer_concat)
    layer_concat = layers.Conv2D(56, [3, 3], padding='same', activation=tf.nn.relu)(layer_concat)

    avg = layers.Average()([incep_module, layer_concat])

    layer_fin = layers.MaxPooling2D(pool_size=(2, 2))(avg)
    layer_fin = layers.Conv2D(20, [1, 1], padding='valid', activation=tf.nn.relu)(layer_fin)
    layer_fin = layers.Conv2D(30, [3, 3], padding='valid', activation=tf.nn.relu)(layer_fin)

    layer_fin = layers.MaxPooling2D(pool_size=(2, 2))(layer_fin)
    layer_fin = layers.Conv2D(40, [3, 3], padding='valid', activation=tf.nn.relu)(layer_fin)

    flatten = layers.Flatten()(layer_fin)
    layer = layers.Dense(1, activation=tf.nn.sigmoid)(flatten)

    

    return Model(Input, layer)


def get_model_classifier(shape=None):
    Input = layers.Input(shape=shape)

    layer = layers.Conv2D(6, [5, 5], padding='valid', activation=tf.nn.relu, name='C1')(Input)

    pool_start = layers.MaxPooling2D(pool_size=(2, 2), name='branch_pool')(layer)

    layer = layers.Conv2D(12, [3, 3], padding='valid', activation=tf.nn.relu, name='inception_branch_conv1')(
        pool_start)  #

    incep_module = inception_module(layer, 3, 3, 6, 3, 6, 6, 'mod1')

    layer_concat = layers.Conv2D(14, [3, 3], padding='same', activation=tf.nn.relu, name='resnet_branch_conv1')(pool_start)
    layer_concat = layers.Conv2D(incep_module.shape[3], [3, 3], padding='valid', activation=tf.nn.relu,
                                 name='resnet_branch_conv2')(layer_concat)

    resize = layers.experimental.preprocessing.Resizing(height=layer_concat.shape[1], width=layer_concat.shape[2])(
        Input)

    incep_module2 = inception_module(resize, 3, 1, 6, 1, 6, 6, 'mod2')
    avg = layers.Average()([incep_module, layer_concat, incep_module2])
    concat = layers.concatenate([avg, resize])

    ################################# пробрасываем ресайз ####################################################
    res_layer = layers.Conv2D(30, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv1',
                              kernel_initializer=tf.keras.initializers.HeNormal(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat)

    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool1')(res_layer)
    res_layer = layers.Conv2D(36, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv2',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    layer = layers.Dropout(0.25)(res_layer)
    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool2')(layer)
    res_layer = layers.Conv2D(40, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv3',
                              kernel_initializer=tf.keras.initializers.lecun_uniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    ########################################################################################################################################################

    layer = layers.Conv2D(45, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv4',
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)



    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.3)(flatten)
    layer = layers.Dense(1, activation=tf.nn.sigmoid, name='output',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(layer)

    return Model(Input, layer)



def get_model_multi_label_classifier_XXX_best(shape=None):
    Input = layers.Input(shape=shape)


    layer = layers.Conv2D(6, [5, 5], padding='valid', activation=tf.nn.relu, name='C1')(Input)

    pool_start = layers.MaxPooling2D(pool_size=(2, 2), name='branch_pool')(layer)

    layer = layers.Conv2D(12, [3, 3], padding='valid', activation=tf.nn.relu, name='inception_branch_conv1')(pool_start) #

    incep_module = inception_module(layer, 3, 3, 10, 3, 10, 10, 'mod1')

    layer_concat = layers.Conv2D(16, [3, 3], padding='same', activation=tf.nn.relu, name='resnet_branch_conv1')(pool_start)
    layer_concat = layers.Conv2D(incep_module.shape[3], [3, 3], padding='valid', activation=tf.nn.relu, name='resnet_branch_conv2')(layer_concat)

    resize = layers.experimental.preprocessing.Resizing(height=layer_concat.shape[1], width=layer_concat.shape[2])(Input)




    incep_module2 = inception_module(resize, 3, 1, 10, 1, 10, 10, 'mod2')
    avg = layers.Average()([incep_module, layer_concat, incep_module2])
    concat = layers.concatenate([avg, resize])


    ################################# пробрасываем ресайз ####################################################
    res_layer = layers.Conv2D(30, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv1', kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat)

    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool1')(res_layer)
    res_layer = layers.Conv2D(36, [3, 3], padding='valid', activation=tf.nn.relu,  name='resize_concat_conv2', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    res_layer = layers.Dropout(0.15)(res_layer)
    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool2')(res_layer)
    res_layer = layers.Conv2D(40, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv3', kernel_initializer=tf.keras.initializers.lecun_uniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    ########################################################################################################################################################


    layer = layers.Conv2D(45, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv4',
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    layer = layers.MaxPooling2D(pool_size=(3, 3), name='post_concat_pool4')(layer)
    layer = layers.Conv2D(50, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv5', kernel_initializer=tf.keras.initializers.LecunNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.MaxPooling2D(pool_size=(3, 3), name='post_concat_pool3')(layer)

    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.3)(flatten)
    layer = layers.Dense(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES), activation=tf.nn.sigmoid, name='output',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(layer)

    return Model(Input, layer)

def branch_pyramid_1(input_layers):
    layer = layers.Conv2D(6, [3, 3], padding='same', activation=tf.nn.relu, name='C1_branch_pyramid_1')(input_layers)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool1_branch_pyramid_1')(layer)
    layer = layers.Conv2D(12, [3, 3], padding='same', activation=tf.nn.relu, name='C2_branch_pyramid_1')(layer)
    layer = layers.Conv2D(6, [1, 1], padding='same', activation=tf.nn.relu, name='C3_branch_pyramid_1')(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool2_branch_pyramid_1')(layer)
    layer = layers.Conv2D(12, [3, 3], padding='same', activation=tf.nn.relu, name='C4_branch_pyramid_1')(layer)
    layer = layers.Dropout(0.2)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool3_branch_pyramid_1')(layer)
    layer = layers.Conv2D(24, [3, 3], padding='same', activation=tf.nn.relu, name='C6_branch_pyramid_1')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool4_branch_pyramid_1')(layer)
    layer = layers.Conv2D(48, [3, 3], padding='same', activation=tf.nn.relu, name='C7_branch_pyramid_1')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool5_branch_pyramid_1')(layer)
    layer = layers.Conv2D(96, [3, 3], padding='same', activation=tf.nn.relu, name='C8_branch_pyramid_1')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool6_branch_pyramid_1')(layer)
    layer = layers.Conv2D(192, [3, 3], padding='same', activation=tf.nn.relu, name='C9_branch_pyramid_1')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool7_branch_pyramid_1')(layer)
    layer = layers.Conv2D(384, [3, 3], padding='same', activation=tf.nn.relu, name='C10_branch_pyramid_1')(layer)

    '''flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.3)(flatten)
    layer = layers.Dense(5, activation=tf.nn.sigmoid, name='output_branch_pyramid_1',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(layer)'''
    return layer

def branch_pyramid_2(input_layers):
    resize = layers.experimental.preprocessing.Resizing(height=int(input_layers.shape[1]/2), width=int(input_layers.shape[2]/2))(input_layers)
    layer = layers.Conv2D(6, [3, 3], padding='same', activation=tf.nn.relu, name='C1_branch_pyramid_2')(resize)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool1_branch_pyramid_2')(layer)
    layer = layers.Conv2D(12, [3, 3], padding='same', activation=tf.nn.relu, name='C2_branch_pyramid_2')(layer)
    layer = layers.Dropout(0.2)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool2_branch_pyramid_2')(layer)
    layer = layers.Conv2D(24, [3, 3], padding='same', activation=tf.nn.relu, name='C4_branch_pyramid_2')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool3_branch_pyramid_2')(layer)
    layer = layers.Conv2D(48, [3, 3], padding='same', activation=tf.nn.relu, name='C6_branch_pyramid_2')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool4_branch_pyramid_2')(layer)
    layer = layers.Conv2D(96, [3, 3], padding='same', activation=tf.nn.relu, name='C7_branch_pyramid_2')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool5_branch_pyramid_2')(layer)
    layer = layers.Conv2D(192, [3, 3], padding='same', activation=tf.nn.relu, name='C8_branch_pyramid_2')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool6_branch_pyramid_2')(layer)
    layer = layers.Conv2D(384, [3, 3], padding='same', activation=tf.nn.relu, name='C9_branch_pyramid_2')(layer)

    '''layer = layers.MaxPooling2D(pool_size=(2, 2), name='pool6_branch_pyramid_2')(layer)
    layer = layers.Conv2D(192, [3, 3], padding='same', activation=tf.nn.relu, name='C9_branch_pyramid_2')(layer)
    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.2)(flatten)
    layer = layers.Dense(5, activation=tf.nn.sigmoid, name='output_branch_pyramid_2',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(layer)'''
    return layer



def get_model_multi_label_classifier_XXX(shape=None):
    Input = layers.Input(shape=shape)



    layer_branch_pyramid_1 = branch_pyramid_1(Input)
    layer_branch_pyramid_2 = branch_pyramid_2(Input)
    #layer_branch_pyramid_3 = branch_pyramid_3(Input)

    final_layer = layers.Add()([layer_branch_pyramid_1, layer_branch_pyramid_2])
    final_layer = layers.Dropout(0.5)(final_layer)
    final_layer = layers.Flatten()(final_layer)
    final_layer = layers.Dense(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES), activation=tf.nn.sigmoid, name='output',
                         kernel_initializer=tf.keras.initializers.LecunNormal(1),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(final_layer)

    return Model(Input, final_layer)


def get_model_multi_label_classifier_XXX_882(shape=None):
    Input = layers.Input(shape=shape)


    layer = layers.Conv2D(6, [5, 5], padding='valid', activation=tf.nn.relu, name='C1')(Input)

    pool_start = layers.MaxPooling2D(pool_size=(2, 2), name='branch_pool')(layer)

    layer = layers.Conv2D(12, [3, 3], padding='valid', activation=tf.nn.relu, name='inception_branch_conv1')(pool_start) #

    incep_module = inception_module(layer, 3, 3, 6, 3, 6, 6, 'mod1')

    layer_concat = layers.Conv2D(14, [3, 3], padding='same', activation=tf.nn.relu, name='resnet_branch_conv1')(pool_start)
    layer_concat = layers.Conv2D(incep_module.shape[3], [3, 3], padding='valid', activation=tf.nn.relu, name='resnet_branch_conv2')(layer_concat)

    resize = layers.experimental.preprocessing.Resizing(height=layer_concat.shape[1], width=layer_concat.shape[2])(Input)




    incep_module2 = inception_module(resize, 1, 1, 3, 1, 3, 3, 'mod2')
    avg = layers.Average()([incep_module, layer_concat])
    #concat = layers.concatenate([avg, resize])

    #########################################################################################################
    concat_layer = layers.Conv2D(25, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv1',
                              kernel_initializer=tf.keras.initializers.HeNormal(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(avg)

    concat_layer = layers.Conv2D(10, [1, 1], padding='valid', activation=tf.nn.relu, name='post_concat_conv2_down',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat_layer)

    concat_layer = layers.MaxPooling2D(pool_size=(2, 2), name='post_concat_pool1')(concat_layer)
    concat_layer = layers.Conv2D(15, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv2',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat_layer)

    concat_layer = layers.MaxPooling2D(pool_size=(2, 2), name='post_concat_pool2')(concat_layer)
    concat_layer = layers.Conv2D(20, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv3',
                              kernel_initializer=tf.keras.initializers.lecun_uniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat_layer)
    #########################################################################################################

    ################################# пробрасываем ресайз ####################################################
    res_layer = layers.Conv2D(25, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv1', kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(incep_module2)

    res_layer = layers.Conv2D(10, [1, 1], padding='valid', activation=tf.nn.relu,  name='resize_concat_conv2_down', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool1')(res_layer)
    res_layer = layers.Conv2D(15, [3, 3], padding='valid', activation=tf.nn.relu,  name='resize_concat_conv2', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)


    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool2')(res_layer)
    res_layer = layers.Conv2D(20, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv3', kernel_initializer=tf.keras.initializers.lecun_uniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    ########################################################################################################################################################

    layer = layers.Average()([res_layer, concat_layer])

    layer = layers.Conv2D(25, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv4',
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.MaxPooling2D(pool_size=(3, 3), name='post_concat_pool4')(layer)
    layer = layers.Conv2D(30, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv5', kernel_initializer=tf.keras.initializers.LecunNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.MaxPooling2D(pool_size=(2, 2), name='post_concat_pool5')(layer)

    layer = layers.Conv2D(35, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv6',
                          kernel_initializer=tf.keras.initializers.LecunNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.MaxPooling2D(pool_size=(2, 2), name='post_concat_pool6')(layer)

    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.3)(flatten)
    layer = layers.Dense(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES)-1, activation=tf.nn.sigmoid, name='output',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(layer)

    return Model(Input, layer)


def get_model_multi_label_classifier_XXX_corr(shape=None):
    Input = layers.Input(shape=shape)


    layer = layers.Conv2D(6, [5, 5], padding='valid', activation=tf.nn.relu, name='C1')(Input)

    pool_start = layers.MaxPooling2D(pool_size=(2, 2), name='branch_pool')(layer)

    layer = layers.Conv2D(12, [3, 3], padding='valid', activation=tf.nn.relu, name='inception_branch_conv1')(pool_start) #

    incep_module = inception_module(layer, 3, 3, 6, 3, 6, 6, 'mod1')

    layer_concat = layers.Conv2D(14, [3, 3], padding='same', activation=tf.nn.relu, name='resnet_branch_conv1')(pool_start)
    layer_concat = layers.Conv2D(incep_module.shape[3], [3, 3], padding='valid', activation=tf.nn.relu, name='resnet_branch_conv2')(layer_concat)

    resize = layers.experimental.preprocessing.Resizing(height=layer_concat.shape[1], width=layer_concat.shape[2])(Input)




    incep_module2 = inception_module(resize, 1, 1, 3, 1, 3, 3, 'mod2')
    avg = layers.Average()([incep_module, layer_concat])
    #concat = layers.concatenate([avg, resize])

    #########################################################################################################
    concat_layer = layers.Conv2D(25, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv1',
                              kernel_initializer=tf.keras.initializers.HeNormal(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(avg)

    concat_layer = layers.Conv2D(10, [1, 1], padding='valid', activation=tf.nn.relu, name='post_concat_conv2_down',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat_layer)

    concat_layer = layers.MaxPooling2D(pool_size=(2, 2), name='post_concat_pool1')(concat_layer)
    concat_layer = layers.Conv2D(15, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv2',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat_layer)

    concat_layer = layers.MaxPooling2D(pool_size=(2, 2), name='post_concat_pool2')(concat_layer)
    concat_layer = layers.Conv2D(20, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv3',
                              kernel_initializer=tf.keras.initializers.lecun_uniform(),
                              bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat_layer)
    #########################################################################################################

    ################################# пробрасываем ресайз ####################################################
    res_layer = layers.Conv2D(25, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv1', kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(incep_module2)

    res_layer = layers.Conv2D(10, [1, 1], padding='valid', activation=tf.nn.relu,  name='resize_concat_conv2_down', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool1')(res_layer)
    res_layer = layers.Conv2D(15, [3, 3], padding='valid', activation=tf.nn.relu,  name='resize_concat_conv2', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)


    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool2')(res_layer)
    res_layer = layers.Conv2D(20, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv3', kernel_initializer=tf.keras.initializers.lecun_uniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    ########################################################################################################################################################

    layer = layers.Concatenate()([res_layer, concat_layer])

    layer = layers.Conv2D(50, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv4',
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.Conv2D(20, [1, 1], padding='valid', activation=tf.nn.relu, name='post_concat_conv4_1',
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.MaxPooling2D(pool_size=(3, 3), name='post_concat_pool4')(layer)
    layer = layers.Conv2D(40, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv5', kernel_initializer=tf.keras.initializers.LecunNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.MaxPooling2D(pool_size=(3, 3), name='post_concat_pool5')(layer)

    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.3)(flatten)
    layer = layers.Dense(len(constants.CLASSIFIER_MULTI_LABEL_CLASSES), activation=tf.nn.sigmoid, name='output',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(layer)

    return Model(Input, layer)


def get_model_binary_classifier_XXX(shape=None):
    Input = layers.Input(shape=shape)


    layer = layers.Conv2D(6, [5, 5], padding='valid', activation=tf.nn.relu, name='C1')(Input)

    pool_start = layers.MaxPooling2D(pool_size=(2, 2), name='branch_pool')(layer)

    layer = layers.Conv2D(12, [3, 3], padding='valid', activation=tf.nn.relu, name='inception_branch_conv1')(pool_start) #

    incep_module = inception_module(layer, 3, 3, 6, 3, 6, 6, 'mod1')

    layer_concat = layers.Conv2D(16, [3, 3], padding='same', activation=tf.nn.relu, name='resnet_branch_conv1')(pool_start)
    layer_concat = layers.Conv2D(incep_module.shape[3], [3, 3], padding='valid', activation=tf.nn.relu, name='resnet_branch_conv2')(layer_concat)

    resize = layers.experimental.preprocessing.Resizing(height=layer_concat.shape[1], width=layer_concat.shape[2])(Input)




    incep_module2 = inception_module(resize, 3, 1, 6, 1, 6, 6, 'mod2')
    avg = layers.Average()([incep_module, layer_concat, incep_module2])
    concat = layers.concatenate([avg, resize])


    ################################# пробрасываем ресайз ####################################################
    res_layer = layers.Conv2D(25, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv1', kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(concat)

    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool1')(res_layer)
    res_layer = layers.Conv2D(30, [3, 3], padding='valid', activation=tf.nn.relu,  name='resize_concat_conv2', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    res_layer = layers.Dropout(0.15)(res_layer)
    res_layer = layers.MaxPooling2D(pool_size=(2, 2), name='resize_concat_pool2')(res_layer)
    res_layer = layers.Conv2D(40, [3, 3], padding='valid', activation=tf.nn.relu, name='resize_concat_conv3', kernel_initializer=tf.keras.initializers.lecun_uniform(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    ########################################################################################################################################################

    res_layer = layers.Conv2D(15, [1, 1], padding='valid', activation=tf.nn.relu, name='post_concat_conv4_1_1',
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    layer = layers.Conv2D(25, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv4',
                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(res_layer)

    layer = layers.MaxPooling2D(pool_size=(3, 3), name='post_concat_pool4')(layer)
    layer = layers.Conv2D(30, [3, 3], padding='valid', activation=tf.nn.relu, name='post_concat_conv5', kernel_initializer=tf.keras.initializers.LecunNormal(),
                          bias_initializer=tf.keras.initializers.GlorotUniform(), use_bias=True)(layer)

    layer = layers.MaxPooling2D(pool_size=(3, 3), name='post_concat_pool3')(layer)

    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.3)(flatten)
    layer = layers.Dense(1, activation=tf.nn.sigmoid, name='output',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         bias_initializer=tf.keras.initializers.LecunNormal(),
                         use_bias=True)(layer)

    return Model(Input, layer)