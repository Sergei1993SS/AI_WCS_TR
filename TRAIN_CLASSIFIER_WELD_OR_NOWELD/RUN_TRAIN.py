from tools import load_data
from tools import constants
from models import models
import tensorflow as tf
from callbacks import callback
import shutil
import numpy as np
import cv2 as cv



def run():
    shutil.rmtree(constants.CLASIIFIER_BINARY_LOG_DIR, ignore_errors=True)

    resampled_ds_train, resampled_ds_validation, resampled_steps_per_epoch = load_data.load_data_set_classifier_weld(split_size=constants.CLASSIFIER_BINARY_SPLIT_SIZE,
                                                                                                                     seed=constants.CLASSIFIER_BINARY_NP_SEED)

    # = models.get_model_classifier(shape=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1],  3))
    #classifier_model = tf.keras.models.load_model(constants.CLASSIFIER_BINARY_SAVE_PATH + '/classifier_weld_97_15.h5')
    classifier_model = models.get_pretrain_model_VGG16()
    classifier_model.summary()

    CallBack_SaveModel = callback.Classifier_Weld_CallBack()
    CallBack_TensorDoard = tf.keras.callbacks.TensorBoard(log_dir=constants.CLASIIFIER_BINARY_LOG_DIR, histogram_freq=1, write_images=True, profile_batch=0)

    '''cv.namedWindow("img", cv.WINDOW_NORMAL)
    for element in resampled_ds_train.as_numpy_iterator():
        print(element[0][0].max())
        print(element[0][0].min())
        cv.imshow('img', cv.cvtColor(element[0][0], cv.COLOR_RGB2BGR))
        cv.waitKey()'''




    '''print(images.min())
    print(images.max())
    print(images)
    print(labels.shape)
    print(np.mean(labels))'''


    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2= 0.999)
    classifier_model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001, decay=1e-6), loss=tf.losses.binary_crossentropy, metrics=['acc'])

    history = classifier_model.fit(
        resampled_ds_train,
        epochs=constants.CLASIIFIER_EPOCHS,
        steps_per_epoch=resampled_steps_per_epoch,
        validation_data=resampled_ds_validation,
        verbose=1,
        use_multiprocessing=True,
        callbacks=[CallBack_SaveModel, CallBack_TensorDoard]
    )

if __name__ == '__main__':
    run()