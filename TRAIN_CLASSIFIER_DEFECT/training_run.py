from tools import statistics
from tools import load_data
from tools import constants
from models import models
from callbacks import callback
import tensorflow as tf


def run():
    jsons = statistics.get_jsons()
    '''dict_stat = statistics.parse_stat_json(jsons)
    statistics.plot_stat(dict_stat)'''

    ds_train, ds_validation, steps_per_epoch = load_data.load_data_set_classifier_defects(split_size=constants.CLASSIFIER_MULTI_LABEL_SPLIT,
                                                                                          seed=constants.CLASSIFIER_MULTI_LABEL_RANDOM_SEED)

    classifier_model = models.get_model_multi_label_classifier(
        shape=(constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1], 3))

    classifier_model.summary()

    CallBack_SaveModel = callback.Classifier_Defect_CallBack()
    CallBack_TensorDoard = tf.keras.callbacks.TensorBoard(log_dir=constants.CLASIIFIER_BINARY_LOG_DIR, histogram_freq=1,
                                                          write_images=True, profile_batch=0)

    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    optimizerRMS = tf.optimizers.RMSprop()
    optimizerNAdam = tf.optimizers.Nadam()

    classifier_model.compile(optimizer=optimizerNAdam, loss=tf.losses.binary_crossentropy, metrics=['acc'])

    history = classifier_model.fit(
        ds_train,
        epochs=constants.CLASSIFIER_MULTI_LABEL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_validation,
        verbose=1,
        use_multiprocessing=True,
        callbacks=[CallBack_SaveModel, CallBack_TensorDoard]
    )

if __name__ == '__main__':
    run()