import os
import cv2 as cv
from PreProcessing import preproc
import numpy as np
from PIL import ImageEnhance, Image

path_image = r'C:\Users\Sergei\Desktop\08.07\JPEG'

def load_path(path):
    list_file = os.listdir(path_image)
    return list_file


files = load_path(path_image)

cv.namedWindow('Origin Image', cv.WINDOW_NORMAL)
#cv.namedWindow('Equalize each channel', cv.WINDOW_NORMAL)
#cv.namedWindow('ycrcb', cv.WINDOW_NORMAL)
#cv.namedWindow('clahe', cv.WINDOW_NORMAL)
#cv.namedWindow('Wiener', cv.WINDOW_NORMAL)
#cv.namedWindow('gray', cv.WINDOW_NORMAL)


for file in files:

    image = cv.imread(path_image + '/' + file)
    '''image_ch = np.copy(image)

    B = preproc.make_histogram(image[:, :, 0])
    G = preproc.make_histogram(image[:, :, 1])
    R = preproc.make_histogram(image[:, :, 2])



    B_ch_equaliz, G_ch_equaliz, R_ch_equaliz = cv.split(image_ch)

    B_ch_equaliz = preproc.equalization_hist(B_ch_equaliz)
    G_ch_equaliz = preproc.equalization_hist(G_ch_equaliz)
    R_ch_equaliz = preproc.equalization_hist(R_ch_equaliz)

    image_ch = cv.merge((B_ch_equaliz, G_ch_equaliz, R_ch_equaliz))


    B_ch = preproc.make_histogram(image_ch[:, :, 0])
    G_ch = preproc.make_histogram(image_ch[:, :, 1])
    R_ch = preproc.make_histogram(image_ch[:, :, 2])

    ################################
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    channels = cv.split(ycrcb)
    channels[0] = cv.equalizeHist(channels[0])
    ycrcb = cv.merge(channels)
    ycrcb = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

    ################################

    B_ch_y = preproc.make_histogram(ycrcb[:, :, 0])
    G_ch_y = preproc.make_histogram(ycrcb[:, :, 1])
    R_ch_y = preproc.make_histogram(ycrcb[:, :, 2])

    ################################
    bgr = np.copy(image)
    gridsize = 50
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv.merge(lab_planes)
    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    #################################

    B_ch_clahe = preproc.make_histogram(bgr[:, :, 0])
    G_ch_clahe = preproc.make_histogram(bgr[:, :, 1])
    R_ch_clahe = preproc.make_histogram(bgr[:, :, 2])


    #################################
'''
    #Wiener = preproc.deblur_Wiener_filter(image)
    im = Image.open( path_image + '/' + file )
    enh = ImageEnhance.Contrast(im)
    image_enh = enh.enhance(1.4).copy()
    enh_sh = ImageEnhance.Sharpness(image_enh)
    enh_sh.enhance(2.5).show('res')

    #image_enh = enh.enhance(1.4).copy()

    cv.imshow("Origin Image", image)
    #cv.imshow('Wiener', )
    #cv.imshow("Equalize each channel", image_ch)
    #cv.imshow('ycrcb', ycrcb)
    #cv.imshow('clahe', bgr)
    #cv.imshow('gray', cv.cvtColor(image, cv.COLOR_BGR2GRAY))

    #################################

    #preproc.plot_histogram([R, G, B, R_ch, G_ch, B_ch, R_ch_y, G_ch_y, B_ch_y, R_ch_clahe, G_ch_clahe, B_ch_clahe])
    #preproc.plt.draw()
    cv.waitKey()

    #preproc.plt.clf()
