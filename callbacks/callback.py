'''
Модуль анализа хода обучения и логирования
Автор: Сергей Сисюкин
e-mail: sergei.sisyukin@gmail.com
'''
import  tensorflow as tf
from tools import constants
import numpy as np

'''
Функция обработки хода обучения классификатора шов/не шов
'''
class Classifier_Weld_CallBack(tf.keras.callbacks.Callback):
    current_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('val_f1')>self.current_val_acc and logs.get('val_f1')>0.9 ):
            self.model.save(constants.CLASSIFIER_BINARY_SAVE_PATH + '/classifier_weld'+str(np.round(logs.get('val_f1'), decimals=3))+'.h5')
            self.current_val_acc = logs.get('val_f1')
            print()
            print("Model saving with val_acc -  {} %".format(self.current_val_acc*100.0))
            print()
            if(self.current_val_acc >0.9999):
                self.model.stop_training = True


class Classifier_Defect_CallBack(tf.keras.callbacks.Callback):
    current_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):

        if(logs.get('val_f1')>self.current_val_acc and logs.get('val_f1')>0.6):

            self.model.save(constants.CLASSIFIER_MULTI_LABEL_SAVE_PATH + '/classifier_defects'+str(np.round(logs.get('val_f1'), decimals=3))+'.h5')
            self.current_val_acc = logs.get('val_f1')
            print()
            print("Model saving with val_f1 -  {} %".format(self.current_val_acc*100.0))
            print()
            if(self.current_val_acc >0.9999):
                self.model.stop_training = True
