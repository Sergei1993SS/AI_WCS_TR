import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import color, data, restoration
from scipy.signal import convolve2d

plt_R = None
plt_G = None
plt_B = None


def make_histogram(img_channel):
    """
    Расчет гистограммы
    :param img_channel:
    :return:
    """
    histogram = cv.calcHist(img_channel, [0], None, [256], (0, 256), accumulate=False)

    return histogram

def plot_histogram(hist):

    x = range(0, 256)

    plt.ion()

    plt.subplot(2, 2, 1)
    plt.plot(x, hist[0], 'r-', label="Red channel")
    plt.plot(x, hist[1], 'g-', label="Green channel")
    plt.plot(x, hist[2], 'b-', label="Blue channel")
    plt.title('Histograms origin image')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x, hist[3], 'r-', label="Red channel")
    plt.plot(x, hist[4], 'g-', label="Green channel")
    plt.plot(x, hist[5], 'b-', label="Blue channel")
    plt.title('Each channel')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x, hist[6], 'r-', label="Red channel")
    plt.plot(x, hist[7], 'g-', label="Green channel")
    plt.plot(x, hist[8], 'b-', label="Blue channel")
    plt.title('ycrcb')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x, hist[9], 'r-', label="Red channel")
    plt.plot(x, hist[10], 'g-', label="Green channel")
    plt.plot(x, hist[11], 'b-', label="Blue channel")
    plt.title('clahe')
    plt.legend()

def equalization_hist(image_channel):
    return cv.equalizeHist(image_channel)

def deblur_Wiener_filter(img):

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('test', img)
    psf = np.ones((5, 5))/25
    img = convolve2d(img, psf, 'same')
    deconvolved_img = restoration.wiener(img, psf, 11000)
    return deconvolved_img
