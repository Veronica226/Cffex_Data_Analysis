from settings import *
import pandas as pd
import os
import datetime
import codecs

# 获取时间间隔小时数
def __get_hour_interval(time_begin_str, time_end_str):
    time_begin = datetime.datetime.strptime(time_begin_str, "%Y-%m-%d %H:%M:%S")
    time_end = datetime.datetime.strptime(time_end_str, "%Y-%m-%d %H:%M:%S")
    return (time_end - time_begin).total_seconds() / 3600

# 计算文件应有数据条数
def __predict_data_length(time_begin_str, time_end_str):
    return __get_hour_interval(time_begin_str, time_end_str) + 1


# 主函数
if __name__ == '__main__':
    # 获取文件列表
    f_list = os.listdir(plot_data_dir)
    print(f_list)
    print("文件个数:" + str(len(f_list)) + "\n\n")

    # 逐个检验文件
    time_incorrect = []
    maxmin_incorrect = []
    for file_name in f_list:
        print(str(file_name)+"处理中...")

        # 获取文件数据
        file_path = os.path.join(plot_data_dir, file_name)
        data = pd.read_csv(file_path, sep=',', header=None, names=["time", "max", "min"], dtype=str)

        # 检验数据条数是否正常
        time_begin = data["time"][0]
        time_end = data["time"][len(data) - 1]

        if len(data) != __predict_data_length(time_begin, time_end):
            for i in range(0, len(data) - 2):
                if __get_hour_interval(data["time"][i], data["time"][i + 1]) != 1:
                    time_incorrect.append(str(file_name) + ":" + data["time"][i] + "与下一条间隔")

        # 检验数据最大最小值是否正常
        for i in range(0, len(data) - 1):
            if float(data["max"][i]) <= 0:
                maxmin_incorrect.append(str(file_name) + ":" + data["time"][i] + "对应最大值为" + data["max"][i])
            if float(data["min"][i]) <= 0:
                maxmin_incorrect.append(str(file_name) + ":" + data["time"][i] + "对应最小值为" + data["min"][i])

    # 写文件
    fp = codecs.open("incorrect.csv", 'w+', "utf-8")

    print("数据间隔不正常位置:")
    fp.write("数据间隔不正常位置:")
    fp.write("\n")
    for info in time_incorrect:
        print(info)
        fp.write(info)
        fp.write("\n")

    fp.write("\n最大最小值小于零位置::")
    fp.write("\n")
    print("\n最大最小值小于零位置:")
    for info in maxmin_incorrect:
        print(info)
        fp.write(info)
        fp.write("\n")

    fp.close()