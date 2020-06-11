import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split,StratifiedKFold
import os
import keras
from keras import backend as K
from keras.utils import Sequence
from PIL import Image
import cv2
import albumentations
from classification_models.keras import Classifiers
import segmentation_models as sm

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D , MaxPool2D, UpSampling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization, Multiply
from keras.layers import LeakyReLU, Dense, GlobalAveragePooling2D, Lambda, GlobalMaxPooling2D
from keras.layers import ZeroPadding2D, Flatten
from keras.losses import binary_crossentropy
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, add, multiply
from keras.layers.merge import concatenate, add
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from tqdm import tqdm_notebook
import keras.callbacks as callbacks
import tensorflow as tf
import seaborn as sns
import shutil

from collections import defaultdict


train_dir = 'severstal-steel-defect-detection/train_images'
mask_dir = 'severstal-steel-defect-detection/train.csv'
test_dir = 'severstal-steel-defect-detection/test_images'

epochs = 30
batch_size = 12
swa_nb = epochs-5

lr = 0.001
image_size = (256, 1600)
shrink = 2
channels = 3


model_name = 'Uefficientnet_bce'


def run():
    train_dir_image_check = glob.glob(train_dir + '/*')[0]
    # train_dir_image_check = './severstal-steel-defect-detection/train_images/005d86c25.jpg'
    img = Image.open(train_dir_image_check)
    plt.figure(figsize=(200, 40))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


    print(np.array(img).shape)





if __name__ == '__main__':
    run()