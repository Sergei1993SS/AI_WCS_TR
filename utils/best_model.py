import tensorflow as tf
from tools import constants
from tools import load_data
import numpy as np
import os
from models import models, metrics


def get_models():
    models_path = os.listdir(constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH)
    models_path.sort()
    models_path = models_path[::-1]
    models_path = [constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/' + model for model in models_path]

    return models_path


# create a model from the weights of multiple models
def model_weight_ensemble(members, weights):
    # determine how many layers need to be averaged
    n_layers = len(members[0].get_weights())
    # create an set of average model weights
    avg_model_weights = list()
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = np.array([model.get_weights()[layer] for model in members])
        # weighted average of weights for this layer
        avg_layer_weights = np.average(layer_weights, axis=0, weights=weights)
        # store average layer weights
        avg_model_weights.append(avg_layer_weights)
    # create a new model with the same structure
    model = tf.keras.models.clone_model(members[0])
    # set the weights in the new
    model.set_weights(avg_model_weights)

    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = tf.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.f1], run_eagerly=False)
    return model


def evaluate_f1(model, ds):
    history = model.evaluate(ds)
    return history[1]


def choose_the_best(list_models, ds):
    best_model = tf.keras.models.load_model(list_models[0], compile=False)

    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = tf.losses.BinaryCrossentropy()
    best_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.f1], run_eagerly=False)
    best_acc = evaluate_f1(best_model, ds)

    print('START BEST ACC: {}'.format(best_acc))
    corrent_best_model = tf.keras.models.clone_model(best_model)

    while len(list_models) != 1:
        idx_remove = 1
        for i in range(1, len(list_models), 1):
            model = tf.keras.models.load_model(list_models[i], compile=False)

            for coef in range(1, 10, 1):

                weights = [1.0, 1.0 / coef]
                avg_model = model_weight_ensemble([best_model, model], weights)
                avg_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.f1], run_eagerly=False)
                acc = evaluate_f1(avg_model, ds)
                if acc > best_acc:
                    print()
                    print('ЛУЧШЕ: {} + {} : current_f1: {}, best_f1: {}'.format(os.path.split(list_models[0])[1],
                                                                                os.path.split(list_models[i])[1],
                                                                                acc, best_acc))
                    idx_remove = i
                    corrent_best_model = tf.keras.models.clone_model(avg_model)
                    best_acc = acc

                else:
                    print()
                    print('ХУЖЕ: {} + {} : current_f1: {}, best_f1: {}'.format(os.path.split(list_models[0])[1],
                                                                               os.path.split(list_models[i])[1],
                                                                               acc, best_acc))
        list_models.pop(idx_remove)
        print()
        print("BEST ACC - {}".format(best_acc))
        print()
        best_model = tf.keras.models.clone_model(corrent_best_model)

    best_model.save(
        constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/best_model_' + str(np.round(best_acc, decimals=3)) + '.h5')


def run():
    tf.keras.backend.clear_session()
    models_path = get_models()

    ds_train, ds_validation, steps_per_epoch = load_data.load_data_set_balanced_classifier_defects(
        split_size=constants.CLASSIFIER_MULTI_LABEL_SPLIT,
        seed=constants.CLASSIFIER_MULTI_LABEL_RANDOM_SEED)

    choose_the_best(models_path, ds_validation)


if __name__ == '__main__':
    run()