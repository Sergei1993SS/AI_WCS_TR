from tools import statistics
from tools import load_data


def run():
    jsons = statistics.get_jsons()

    load_data.load_data_set_classifier_defects()


if __name__ == '__main__':
    run()