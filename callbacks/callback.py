'''
Модуль анализа хода обучения и логирования
Автор: Сергей Сисюкин
e-mail: sergei.sisyukin@gmail.com
'''
import  tensorflow as tf
from tools import constants

'''
Функция обработки хода обучения классификатора шов/не шов
'''
class Classifier_Weld_CallBack(tf.keras.callbacks.Callback):
    current_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('val_acc')>self.current_val_acc):
            self.model.save(constants.CLASSIFIER_BINARY_SAVE_PATH + '/classifier_weld.h5')
            self.current_val_acc = logs.get('val_acc')
            print()
            print("Model saving with val_acc -  {} %".format(self.current_val_acc*100.0))
            print()
            if(self.current_val_acc >0.9999):
                self.model.stop_training = True



