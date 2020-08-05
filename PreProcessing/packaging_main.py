import os

date = '3.07.20'
set = 'Set1'
path_origin_image = 'E:\\DataSet\\' + date + '\\original'
path_correct_set = 'C:\\Users\\Sergei\\OneDrive - 2050-integrator.com\\Marking DataSet Weld\\DataSet\\'+ set +'\\IMAGES\\JPEG'

def run_packaging():

    list_path_correct_set = os.listdir(path_correct_set)
    list_path_correct_set = [os.path.splitext(file)[0] for file in list_path_correct_set]

    #for file in list_path_correct_set:


    print(list_path_correct_set)



if __name__ == '__main__':
    run_packaging()