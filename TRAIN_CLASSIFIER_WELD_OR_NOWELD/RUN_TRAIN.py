from tools import load_data
from tools import constants
from models import models
import tensorflow as tf
import numpy as np

def run():
    resampled_ds_train, resampled_ds_validation, resampled_steps_per_epoch = load_data.load_data_set_classifier_weld(split_size=constants.CLASSIFIER_BINARY_SPLIT_SIZE,
                                                                                                                     seed=constants.CLASSIFIER_BINARY_NP_SEED)

    classifier_model = models.get_model_classifier(shape=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1],  3))
    classifier_model.summary()


    '''dataset = resampled_ds_validation.take(1)

    images, labels = list(dataset.as_numpy_iterator())[0]


    print(images.shape)
    print(labels.shape)
    print(np.mean(labels))'''


    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2= 0.999)
    classifier_model.compile(optimizer=optimizer, loss=tf.losses.binary_crossentropy, metrics=['acc'])

    history = classifier_model.fit(
        resampled_ds_train,
        epochs=constants.CLASIIFIER_EPOCHS,
        steps_per_epoch=resampled_steps_per_epoch,
        validation_data=resampled_ds_validation,
        verbose=1
    )

if __name__ == '__main__':
    run()