from tools import load_data
from tools import constants
from models import models
import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QFileDialog, QApplication,QLabel,QPushButton, QVBoxLayout,QWidget
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QIcon, QPixmap, QImage
import matplotlib.pyplot as plt
from PIL import Image, ImageQt

import sys
path = 'home\sergei\DataSet'

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 60, 900, 800)

        self.button_open = QPushButton('Выбрать картинку')
        self.button_open.clicked.connect(self._on_open_image)

        self.button_res = QPushButton('Результат анализа')
        self.model = tf.keras.models.load_model(constants.CLASSIFIER_BINARY_SAVE_PATH + '/classifier_weld_cast_99_03.h5')


        self.label_image = QLabel()


        main_layout = QVBoxLayout()
        main_layout.addWidget(self.button_open)
        main_layout.addWidget(self.button_res)
        main_layout.addWidget(self.label_image)
        print(os.environ['XDG_RUNTIME_DIR'])

        self.setLayout(main_layout)

    def _on_open_image(self):

        file_name = QFileDialog.getOpenFileName(self, 'Open File', path, 'Images (*.png *.xpm *.jpg)')[0]
        if not file_name:
            return
        print(file_name)
        pixmap = QPixmap(file_name)
        pixmap = pixmap.scaled(self.height(), self.width(), QtCore.Qt.KeepAspectRatio)
        self.label_image.setPixmap(pixmap)

        pre_pixmap = pixmap.copy()
        pre_pixmap = pre_pixmap.scaled(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1])

        arr = Image.fromqpixmap(pre_pixmap)
        arr = np.asarray(arr)
        arr = arr/255.0




        pred = self.model.predict(np.expand_dims(arr, axis=0))
        if pred[0][0]<0.4:
            self.button_res.setText('Вероятность наличия шва: {}%.   ШВА НЕТ!'.format(round(pred[0][0]*100, 3)))
        else:
            self.button_res.setText('Вероятность наличия шва: {}%.   ШОВ ЕСТЬ!'.format(round(pred[0][0]*100, 3)))







def run():
    print('Start load data info')
    list_NO_weld = [constants.CLASSIFIER_BINARY_PATH_NO_WELD + file for file in os.listdir(constants.CLASSIFIER_BINARY_PATH_NO_WELD)]
    list_YES_weld = load_data.make_list_yes_weld(constants.CLASIIFIER_MODE_LOAD)
    train_pos, train_neg, validation_pos, validation_neg = load_data.split_dataset_classifier_weld(list_NO_weld, list_YES_weld, constants.CLASSIFIER_BINARY_SPLIT_SIZE, constants.CLASSIFIER_BINARY_NP_SEED)

    model = tf.keras.models.load_model(constants.CLASSIFIER_BINARY_SAVE_PATH + '/classifier_weld_cast_99_03.h5')
    model.summary()
    counter = 0


    cv.namedWindow('valid_pos', cv.WINDOW_NORMAL)
    root = tk.Tk()
    root.withdraw()

    while True:
        file_path = filedialog.askopenfilename(initialdir="/home/sergei/DataSet", title="Select file", filetypes=(("jpeg files","*.jpg"),("all files","*.*")))
        img = cv.imread(file_path)
        img = cv.resize(img, dsize=(constants.CLASSIFIER_BINARY_IMG_SIZE[0], constants.CLASSIFIER_BINARY_IMG_SIZE[1]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img / 255.0
        pred = model.predict(np.expand_dims(img, axis=0))
        print(pred[0][0])
        cv.imshow('valid_pos', img)
        cv.waitKey()



if __name__ == '__main__':
    #run()
    #subprocess.Popen(["xdg-open /select", "/home/sergei/DataSet"])

    app = QApplication([])

    mw = MainWindow()
    mw.show()

    app.exec()
