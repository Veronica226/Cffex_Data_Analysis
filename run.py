import sys

from code.preprocessing import data_preprocessing
import os

sys.path.append("..")

BASE_DIR = os.getcwd()

ORIGIN_DATA = os.path.join(BASE_DIR, "raw_data", "cffex-host-info")
PRE_DATA = os.path.join(BASE_DIR, "raw_data", "pre_data")

output_dir= os.path.join(BASE_DIR,"output_data")
plot_data_dir = os.path.join(BASE_DIR,"output_data","plot-data")
plot_dir = os.path.join(BASE_DIR,"output_data","kpi-plot")
predict_data_dir =  os.path.join(BASE_DIR,"output_data","predicting_data")
origin_dir = ORIGIN_DATA
pre_dir = PRE_DATA


raw_data_dir = os.path.join(BASE_DIR, 'raw_data').replace('\\', '/')

#告警事件原始数据路径
host_alarm_dir = os.path.join(BASE_DIR,"raw_data","cffex-host-alarm")

#数据预处理
data_preprocessing.data_preprocessing_process(origin_dir,output_dir,host_alarm_dir)

predict_data = os.path.join(predict_data_dir,"predict_data.csv")

# data_preprocessing.data_preprocessing_process(origin_dir,output_dir)
# data_preprocessing.generate_plot_data(output_dir,plot_data_dir)
#data_preprocessing.generate_kpi_plot(origin_dir, plot_dir)
# data_preprocessing.insert_missing_data(plot_data_dir)
#data_preprocessing.generate_feature_by_hostname(plot_data_dir, predict_data)
# data_preprocessing.get_host_name(os.path.join(plot_data_dir,'/alarmsvr1_cpu.csv'))



f_list = os.listdir(plot_data_dir)
for file_name in f_list:
    loc_list = data_preprocessing.find_missing_loc(plot_data_dir,file_name)
    if loc_list != []:
        print (file_name)

