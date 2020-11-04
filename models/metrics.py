import tensorflow as tf
import tensorflow_addons as tfa
from tools import constants
from tensorflow.keras import backend as K


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def acc_glass(y_true, y_pred, treshhold = 0.5):

    b_y = tf.slice(y_true, [0, 0], [-1, 1])
    b_y_hat = tf.slice(y_pred, [0, 0], [-1, 1])

    tr_y_pred_bool = tf.greater(b_y_hat, treshhold)
    tr_y_pred_float = tf.cast(tr_y_pred_bool, dtype=tf.float32)

    acc = tf.reduce_mean(b_y*tr_y_pred_float)
    return acc

