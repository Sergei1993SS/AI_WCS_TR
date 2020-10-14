import os
from tools import constants

def get_jsons():
    list_DIR_JSON= os.listdir(constants.PATH_DATASET)
    list_DIR_JSON.remove('Set_no_weld')
    list_DIR_JSON = [constants.PATH_DATASET + dir + '/JSON/' for dir in list_DIR_JSON]

    for dir in list_DIR_JSON:
