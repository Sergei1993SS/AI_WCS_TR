from tools import statistics

def run():
    json = statistics.get_jsons()
    dict_stat = statistics.parse_stat_json(json)
    statistics.plot_stat(dict_stat)


if __name__ == '__main__':
    statistics.plot_DataSet()