PATH_DATASET = "/home/sergei/DataSet/"
SHAPE_ORIGIN_IMAGE = [2048, 2448, 3]
'''
<BINARY CLASSIFIER>
'''
CLASIIFIER_MODE_LOAD = 'JPEG'
CLASSIFIER_BINARY_PATH_NO_WELD = PATH_DATASET + "Set_no_weld/IMAGES/" + CLASIIFIER_MODE_LOAD + "/"

CLASSIFIER_BINARY_NP_SEED = 21115
CLASSIFIER_BINARY_SPLIT_SIZE = 0.9
CLASSIFIER_BINARY_TRAIN_MODE_NPY = True


CLASSIFIER_BINARY_IMG_SIZE = [548, 948]  # origin shape=(2048, 2448, 3)
CLASSIFIER_BINARY_AUGMENTATION_NOISE_MEAN = 0
CLASSIFIER_BINARY_AUGMENTATION_NOISE_STDEV = 100
CLASSIFIER_BINARY_NORNALIZE = 1.0 #camera 12 bit
CLASSIFIER_BINARY_CORE_CPU = 4
CLASSIFIER_BATCH_SIZE = 40

CLASIIFIER_EPOCHS = 3000

CLASSIFIER_BINARY_SAVE_PATH = '/home/sergei/PycharmProjects/AI_WCS_TRAIN/fit_models/classifier_weld'
CLASIIFIER_BINARY_LOG_DIR = '/home/sergei/PycharmProjects/AI_WCS_TRAIN/logs/fit/classifier_weld/'




'''
</BINARY CLASSIFIER>
'''

'''
<MULTI-LABEL CLASSIFIER
'''
CLASSIFIER_MULTI_LABEL_CLASSES = ['glass', 'burn_and_fistula_pores_and_inclusions', 'metal_spray', 'crater_shell', 'background'] #'cracks', 'undercut',
                                                                                                # background always in the end
CLASSIFIER_MULTI_LABEL_RANDOM_SEED = 5
CLASSIFIER_MULTI_LABEL_SPLIT = 0.90
CLASSIFIER_MULTI_LABEL_IMG_SIZE = [2048, 2448]
CLASSIFIER_MULTI_LABEL_BATCH_SIZE = 12
CLASSIFIER_MULTI_LABEL_SAVE_PATH = '/home/sergei/PycharmProjects/AI_WCS_TRAIN/fit_models/classifier_defects'
CLASSIFIER_MULTI_LABEL_EPOCHS = 2000

CLASSIFIER_MULTI_LABEL_LOG_DIR = '/home/sergei/PycharmProjects/AI_WCS_TRAIN/logs/fit/classifier_defects/'

'''
</MULTI-LABEL CLASSIFIER
'''
