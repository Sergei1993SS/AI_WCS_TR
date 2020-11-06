from tools import statistics
from tools import load_data
from tools import constants
from models import models
from callbacks import callback
import tensorflow as tf
import shutil
from  models import loss
from models import metrics


def run():
    #jsons = statistics.get_jsons()
    '''dict_stat = statistics.parse_stat_json(jsons)
    statistics.plot_stat(dict_stat)'''

    shutil.rmtree(constants.CLASSIFIER_MULTI_LABEL_LOG_DIR, ignore_errors=True)

    ds_train, ds_validation, steps_per_epoch = load_data.load_data_set_classifier_defects(split_size=constants.CLASSIFIER_MULTI_LABEL_SPLIT,
                                                                                          seed=constants.CLASSIFIER_MULTI_LABEL_RANDOM_SEED)
    tf.keras.backend.clear_session()

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    with strategy.scope():
        classifier_model = models.get_model_multi_label_classifier(
                        shape=(constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1], 3))

        #classifier_model = tf.keras.models.load_model(constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/classifier_defects_r67_p73_f69.h5', compile=False)
        classifier_model.summary()





        CallBack_SaveModel = callback.Classifier_Defect_CallBack()
        CallBack_TensorDoard = tf.keras.callbacks.TensorBoard(log_dir=constants.CLASSIFIER_MULTI_LABEL_LOG_DIR,
                                                              histogram_freq=1,
                                                              write_images=False, profile_batch=0)

        optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        #optimizerRMS = tf.optimizers.RMSprop()
        optimizerNAdam = tf.optimizers.Nadam()
        #accuracy = tf.metrics.BinaryAccuracy(threshold=0.7)
        loss = tf.losses.BinaryCrossentropy()

        classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.f1,
                                                                          tf.metrics.Recall(thresholds=0.5, class_id=0),
                                                                          tf.metrics.Precision(thresholds=0.5, class_id=0),
                                                                          tf.metrics.Recall(thresholds=0.5, class_id=1),
                                                                          tf.metrics.Precision(thresholds=0.5, class_id=1),
                                                                          tf.metrics.Recall(thresholds=0.5, class_id=2),
                                                                          tf.metrics.Precision(thresholds=0.5, class_id=2),
                                                                          tf.metrics.Recall(thresholds=0.5, class_id=3),
                                                                          tf.metrics.Precision(thresholds=0.5, class_id=3),
                                                                          tf.metrics.Recall(thresholds=0.5, class_id=4),
                                                                          tf.metrics.Precision(thresholds=0.5, class_id=4),
                                                                          tf.metrics.Recall(thresholds=0.5, class_id=5),
                                                                          tf.metrics.Precision(thresholds=0.5, class_id=5),
                                                                          tf.metrics.Recall(thresholds=0.5, class_id=6),
                                                                          tf.metrics.Precision(thresholds=0.5, class_id=6),
                                                                            ],
                                                                          run_eagerly=False)

        history = classifier_model.fit(
            ds_train,
            epochs=constants.CLASSIFIER_MULTI_LABEL_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=ds_validation,
            verbose=1,
            use_multiprocessing=True,
            callbacks=[CallBack_SaveModel, CallBack_TensorDoard]
        )

        print(history.history)




if __name__ == '__main__':
    run()