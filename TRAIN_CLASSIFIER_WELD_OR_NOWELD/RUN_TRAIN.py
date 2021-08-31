from tools import load_data
from tools import constants
from models import models
import tensorflow as tf
from callbacks import callback
import shutil
from models import metrics




def run():
    shutil.rmtree(constants.CLASIIFIER_BINARY_LOG_DIR, ignore_errors=True)

    resampled_ds_train, resampled_ds_validation,  resampled_steps_per_epoch = load_data.load_data_set_classifier_weld(split_size=constants.CLASSIFIER_BINARY_SPLIT_SIZE,
                                                                                                                     seed=constants.CLASSIFIER_BINARY_NP_SEED) #resampled_ds_validation,


    classifier_model = models.get_model_binary_classifier_XXX(shape=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1],  3))
    classifier_model.summary()



    CallBack_SaveModel = callback.Classifier_Weld_CallBack()
    CallBack_TensorDoard = tf.keras.callbacks.TensorBoard(log_dir=constants.CLASIIFIER_BINARY_LOG_DIR, histogram_freq=1,)

    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2= 0.999)
    #optimizerRMS = tf.optimizers.RMSprop()
    optimizerNAdam = tf.optimizers.Nadam()



    classifier_model.compile(optimizer=optimizerNAdam, loss=tf.losses.binary_crossentropy, metrics=[metrics.f1, tf.metrics.Recall(thresholds=0.5),
                                                                                               tf.metrics.Precision(thresholds=0.5)],
                                                                                                )



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