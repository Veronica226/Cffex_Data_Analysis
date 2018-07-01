import sys

from code.preprocessing import data_preprocessing
import os

sys.path.append("..")

BASE_DIR = os.getcwd()

ORIGIN_DATA = os.path.join(BASE_DIR, "raw_data", "cffex-host-info")
PRE_DATA = os.path.join(BASE_DIR, "raw_data", "pre_data")

output_dir= os.path.join(BASE_DIR,"output_data", "cffex-host-info")
plot_data_dir = os.path.join(BASE_DIR,"output_data","plot-data")
plot_dir = os.path.join(BASE_DIR,"output_data","kpi-plot")
origin_dir = ORIGIN_DATA
pre_dir = PRE_DATA

raw_data_dir = os.path.join(BASE_DIR, 'raw_data').replace('\\', '/')

#告警事件原始数据路径
host_alarm_dir = os.path.join(BASE_DIR,"raw_data","cffex-host-alarm","cffex-host-alarm.csv")

#数据预处理
data_preprocessing.data_preprocessing_process(origin_dir,output_dir,host_alarm_dir)

data_preprocessing.insert_missing_data(plot_data_dir)

# f_list = os.listdir(plot_data_dir)
# for file_name in f_list:
#     print (file_name)

