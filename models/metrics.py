from tensorflow.keras import backend as K
import tensorflow as tf
from tools import constants



def precision(y_true, y_pred):
    r_y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), constants.DEFECTS_THRESHOLD), K.floatx())
    TP = tf.cast(tf.math.count_nonzero(y_true * r_y_pred), dtype=y_true.dtype)

    FP = tf.cast(tf.math.count_nonzero(r_y_pred * (y_true - 1.0)), dtype=y_true.dtype)

    precision_keras = TP / (TP + FP + K.epsilon())
    TP = None
    FP = None
    return precision_keras


def recall(y_true, y_pred):
    r_y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), constants.DEFECTS_THRESHOLD), K.floatx())
    TP = tf.cast(tf.math.count_nonzero(y_true * r_y_pred), dtype=y_true.dtype)
    FN = tf.cast(tf.math.count_nonzero((r_y_pred - 1.0) * y_true), dtype=y_true.dtype)

    recall_keras = TP / (TP + FN + K.epsilon())
    TP = None
    FN = None
    return recall_keras


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 5 * ((p * r) / (4*p + r + K.epsilon()))




