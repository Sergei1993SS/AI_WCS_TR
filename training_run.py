from tools import load_data
from tools import constants
from models import models
from callbacks import callback
import tensorflow as tf
import shutil
from models import metrics

def run():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__)
    #jsons = statistics.get_jsons()
    '''dict_stat = statistics.parse_stat_json(jsons)
    statistics.plot_stat(dict_stat)'''

    shutil.rmtree(constants.CLASSIFIER_MULTI_LABEL_LOG_DIR, ignore_errors=True)

    ds_train,  steps_per_epoch, ds_validation,val_steps = load_data.load_data_set_balanced_classifier_defects_cast(split_size=constants.CLASSIFIER_MULTI_LABEL_SPLIT,
                                                                                          seed=constants.CLASSIFIER_MULTI_LABEL_RANDOM_SEED) #ds_validation,
    tf.keras.backend.clear_session()

    #strategy = tf.distribute.MultiWorkerMirroredStrategy()

    '''#with strategy.scope():'''
    classifier_model = models.get_model_multi_label_classifier_XXX(
        shape=(constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[0], constants.CLASSIFIER_MULTI_LABEL_IMG_SIZE[1], 3))

    #classifier_model = tf.keras.models.load_model(constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/classifier_defects0.629.h5', compile=False)
    classifier_model.summary()
    CallBack_SaveModel = callback.Classifier_Defect_CallBack()
    CallBack_TensorDoard = tf.keras.callbacks.TensorBoard(log_dir=constants.CLASSIFIER_MULTI_LABEL_LOG_DIR,
                                                          histogram_freq=1,
                                                          write_images=False, profile_batch=0)

    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    optimizerNAdam = tf.optimizers.Nadam(learning_rate=0.001)
    # accuracy = tf.metrics.BinaryAccuracy(threshold=0.7)
    loss = tf.losses.BinaryCrossentropy()
    # ls = loss.f1

    classifier_model.compile(optimizer=optimizerNAdam, loss=loss, metrics=[metrics.f1,

                                                                      tf.metrics.Recall(thresholds=0.5,
                                                                                        class_id=0,
                                                                                        name='Recall_GLASS'),
                                                                      tf.metrics.Precision(thresholds=0.5,
                                                                                           class_id=0,
                                                                                           name='Precision_GLASS'),
                                                                      tf.metrics.Recall(thresholds=0.5,
                                                                                        class_id=1,
                                                                                        name='Recall_burn_and_fistula'),
                                                                      tf.metrics.Precision(thresholds=0.5,
                                                                                           class_id=1,
                                                                                           name='Precision_burn_and_fistula'),
                                                                      tf.metrics.Recall(thresholds=0.5,
                                                                                        class_id=2,
                                                                                        name='Recall_metal_spray'),
                                                                      tf.metrics.Precision(thresholds=0.5,
                                                                                           class_id=2,
                                                                                           name='Precision_metal_spray'),
                                                                      tf.metrics.Recall(thresholds=0.5,
                                                                                        class_id=3,
                                                                                        name='Recall_crater_shell'),
                                                                      tf.metrics.Precision(thresholds=0.5,
                                                                                           class_id=3,
                                                                                           name='Precision_crater_shell'),
                                                                      tf.metrics.Recall(thresholds=0.5,
                                                                                        class_id=4,
                                                                                        name='Recall_background'),
                                                                      tf.metrics.Precision(thresholds=0.5,
                                                                                           class_id=4,
                                                                                           name='Precision_background'),
                                                                      # tf.metrics.Recall(thresholds=0.5, class_id=5),
                                                                      # tf.metrics.Precision(thresholds=0.5, class_id=5),
                                                                      # tf.metrics.Recall(thresholds=0.5, class_id=6),
                                                                      # tf.metrics.Precision(thresholds=0.5, class_id=6),
                                                                      ],
                             run_eagerly=False)

    class_weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.,
                    # Установим вес "2" для класса "5",
                    # сделав этот класс в 2x раз важнее
                    # 5: 1.,
                    }

    history = classifier_model.fit(
        ds_train,
        epochs=constants.CLASSIFIER_MULTI_LABEL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_validation,
        #validation_steps=val_steps,
        validation_freq=1,
        class_weight=class_weight,
        verbose=1,
        use_multiprocessing=False,
        workers=1,
        callbacks=[CallBack_SaveModel, CallBack_TensorDoard]
    )

    print(history.history)






if __name__ == '__main__':
    run()