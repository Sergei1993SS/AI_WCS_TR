import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_hub as hub
from tools import constants
from tensorflow.keras.applications import VGG16


# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = layers.Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = layers.Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
    # 5x5 conv
    conv5 = layers.Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
    conv5 = layers.Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = layers.Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

def get_pretrain_model_inceptionV3():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1], 3)),
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/classification/4", trainable=True),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation=tf.nn.sigmoid)
    ])

    model.build([None, constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1], 3])  # Batch input shape.
    return model

def get_pretrain_model_VGG16():

    base_model = VGG16(input_shape=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1], 3), include_top=False)
    for layer in base_model.layers:
        if(layer.name != 'block5_conv3' and layer.name != 'block5_pool' and layer.name != 'block5_conv2' ):
            layer.trainable = True
        else:
            layer.trainable = True
    #input = layers.InputLayer(input_shape=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1], 3))
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dropout(0.35)(x)
    predictions = layers.Dense(1, activation="sigmoid")(x)
    model_final = Model(base_model.input, predictions, name='classifier_weld_model')
    return model_final

def get_model_classifier(shape = None):

    Input = layers.Input(shape=shape)

    layer = layers.Conv2D(16, [7, 7], padding='same', activation=tf.nn.relu)(Input)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = layers.Conv2D(64, [1, 1], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = inception_module(layer, 64, 96, 128, 16, 32, 32)
    layer = layers.BatchNormalization()(layer)
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
    layer = layers.BatchNormalization()(layer)

    layer = layers.Conv2D(32, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)


    layer = layers.Conv2D(32, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)


    layer = layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)

    flatten = layers.Flatten()(layer)
    layer = layers.Dropout(0.2)(flatten)
    layer = layers.Dense(1, activation=tf.nn.sigmoid)(layer)

    

    return Model(Input, layer)

'''def get_model_classifier(shape = None):

    Input = layers.Input(shape=shape)

    layer = inception_module(Input, 64, 96, 128, 16, 32, 32)
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)

    layer = layers.Conv2D(32, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)


    layer = layers.Conv2D(32, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)


    layer = layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = layers.Conv2D(128, [3, 3], padding='same', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = layers.Dropout(rate=0.2)(layer)


    flatten = layers.Flatten()(layer)
    layer = layers.Dense(50, activation=tf.nn.relu)(flatten)
    layer = layers.Dense(1, activation=tf.nn.sigmoid)(layer)

    return Model(Input, layer)'''