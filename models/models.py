import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

def get_model_classifier(shape = None):

    Input = layers.Input(shape=shape)

    layer = layers.Conv2D(32, [3, 3], padding='valid', activation=tf.nn.relu)(Input)
    layer = layers.MaxPooling2D(pool_size=[2, 2], padding='valid')(layer)

    layer = layers.Conv2D(64, [3, 3], padding='valid', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=[2, 2], padding='valid')(layer)

    layer = layers.Conv2D(128, [3, 3], padding='valid', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=[2, 2], padding='valid')(layer)

    layer = layers.Conv2D(128, [3, 3], padding='valid', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=[2, 2], padding='valid')(layer)

    layer = layers.Conv2D(128, [3, 3], padding='valid', activation=tf.nn.relu)(layer)
    layer = layers.MaxPooling2D(pool_size=[2, 2], padding='valid')(layer)

    flatten = layers.Flatten()(layer)
    layer = layers.Dense(50, activation=tf.nn.relu)(flatten)
    layer = layers.Dense(10, activation=tf.nn.relu)(layer)
    layer = layers.Dense(1, activation=tf.nn.sigmoid)(layer)

    return Model(Input, layer)