from tools import statistics


def run():
    jsons = statistics.get_jsons()
    dict_stat = statistics.parse_stat_json(jsons)
    statistics.plot_stat(dict_stat)


if __name__ == '__main__':
    run()