from tensorflow.keras import backend as K


def recall(y_true, y_pred):
    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    possible_positives = K.sum(y_true)
    recall_keras = true_positives / (possible_positives + K.epsilon())
    possible_positives = None
    true_positives = None
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    predicted_positives = K.sum(K.clip(y_pred, 0, 1))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    true_positives = None
    predicted_positives = None
    return precision_keras


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 1 - (2 * ((p * r) / (p + r + K.epsilon())))