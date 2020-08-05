import numpy as np
import pandas as pd


def rle2mask(rle, imgshape ,shrink=1):
    if (pd.isnull(rle)) | (rle == ''):
        return np.zeros((imgshape[0] // shrink, imgshape[1] // shrink), dtype=np.uint8)
    height = imgshape[0]
    width = imgshape[1]
    mask = np.zeros(width * height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1

    return mask.reshape((height, width), order='F')[::shrink, ::shrink]