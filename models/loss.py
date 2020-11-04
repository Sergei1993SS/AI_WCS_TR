import tensorflow as tf
import tensorflow_addons as tfa
from tools import constants

def loss_weld_defect(y, y_hat):


    '''background_y = tf.slice(y, [0, len(constants.CLASSIFIER_MULTI_LABEL_CLASSES)-1], [-1, 1])

    inv_y = tf.equal(background_y, 0)
    non_background = tf.cast(inv_y, dtype=tf.float32)

    #defects_y_hat = tf.slice(y_hat, [0, 0], [-1, len(constants.CLASSIFIER_MULTI_LABEL_CLASSES) - 1])
    #defects_y = tf.slice(y, [0, 0], [-1, len(constants.CLASSIFIER_MULTI_LABEL_CLASSES) - 1])

    shape = tf.shape(non_background)
    loss_defects_binary_crossentripy = tf.multiply(tf.reshape(non_background, [shape[0]]), tf.losses.binary_crossentropy(y, y_hat, from_logits=False))

    loss_defect_catecorical_crossentropy = tf.multiply(tf.reshape(background_y, [shape[0]]), tf.losses.categorical_crossentropy(y, tf.nn.softmax(y_hat)))

    loss = loss_defect_catecorical_crossentropy + loss_defects_binary_crossentripy

    inv_y = None
    background_y = None
    non_background = None
    shape = None
    loss_defects_binary_crossentripy = None
    loss_defect_catecorical_crossentropy = None'''


    return tf.losses.binary_crossentropy(y, y_hat, from_logits=False)


def hamming_loss(y_true, y_pred):

    loss = tfa.metrics.hamming.hamming_loss_fn(y_true, y_pred, threshold=0.8, mode='multilabel')
    return loss