PATH_DATASET = "/home/sergei/DataSet/"
SHAPE_ORIGIN_IMAGE = [2048, 2448, 3]
'''
<BINARY CLASSIFIER>
'''
CLASSIFIER_BINARY_PATH_NO_WELD = PATH_DATASET + "Set_no_weld/IMAGES/original/"

CLASSIFIER_BINARY_NP_SEED = 1
CLASSIFIER_BINARY_SPLIT_SIZE = 0.8
CLASSIFIER_BINARY_TRAIN_MODE_NPY = True


CLASSIFIER_BINARY_IMG_SIZE = [256, 306]  # origin shape=(2048, 2448, 3)
CLASSIFIER_BINARY_AUGMENTATION_FLIP_LEFT_RIGT = False
CLASSIFIER_BINARY_AUGMENTATION_FLIP_UP_DOWN = False
CLASSIFIER_BINARY_AUGMENTATION_NOISE = False
CLASSIFIER_BINARY_AUGMENTATION_NOISE_MEAN = 0
CLASSIFIER_BINARY_AUGMENTATION_NOISE_STDEV = 100
CLASSIFIER_BINARY_NORNALIZE = 4096.0 #camera 12 bit
CLASSIFIER_BINARY_CORE_CPU = 4
CLASSIFIER_BATCH_SIZE = 64
CLASIIFIER_EPOCHS = 10




'''
</BINARY CLASSIFIER>
'''