import sys

from code.preprocessing import data_preprocessing
import os

sys.path.append("..")
BASE_DIR = os.path.abspath('.')
root_dir = os.getcwd()

ORIGIN_DATA = os.path.join(BASE_DIR, "raw_data", "cffex-host-info")
PRE_DATA = os.path.join(BASE_DIR, "raw_data", "pre_data")

output_dir= os.path.join(BASE_DIR,"output_data", "cffex-host-info")
plot_data_dir = os.path.join(root_dir,"output_data","plot-data")
plot_dir = os.path.join(BASE_DIR,"output_data","kpi-plot")
predict_data_dir =  os.path.join(BASE_DIR,"output_data","predicting_data")
origin_dir = ORIGIN_DATA
pre_dir = PRE_DATA
root_dir = os.getcwd()
raw_data_dir = os.path.join(root_dir, 'raw_data').replace('\\', '/')
predict_data = os.path.join(predict_data_dir,"predict_data.csv")

#data_preprocessing.data_preprocessing_process(origin_dir,output_dir)
data_preprocessing.generate_plot_data(output_dir,plot_data_dir)
#data_preprocessing.generate_kpi_plot(origin_dir, plot_dir)
#data_preprocessing.insert_missing_data(plot_data_dir)
#data_preprocessing.generate_feature_by_hostname(plot_data_dir, predict_data)
#data_preprocessing.get_host_name(os.path.join(plot_data_dir,'/alarmsvr1_cpu.csv'))


# f_list = os.listdir(plot_data_dir)
# for file_name in f_list:
#     loc_list = data_preprocessing.find_missing_loc(plot_data_dir,file_name)
#     if loc_list != []:
#         print (file_name)
#
