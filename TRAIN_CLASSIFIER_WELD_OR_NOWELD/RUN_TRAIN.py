from tools import load_data
from tools import constants

def run():
    load_data.load_data_set_classifier_weld(split_size=constants.SPLIT_SIZE_CLASSIFIER, seed=constants.NP_SEED_CLASSIFIER)


if __name__ == '__main__':
    run()