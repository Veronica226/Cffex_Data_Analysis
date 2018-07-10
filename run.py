import datetime
import sys
import pandas as pd

from code.preprocessing import data_preprocessing
from code.feature_engineering import feature_extraction
from code.model import predict_model

import os

sys.path.append("..")

BASE_DIR = os.getcwd()

ORIGIN_DATA = os.path.join(BASE_DIR, "raw_data", "cffex-host-info")
PRE_DATA = os.path.join(BASE_DIR, "raw_data", "pre_data")

output_dir= os.path.join(BASE_DIR,"output_data", "cffex-host-info")
plot_data_dir = os.path.join(BASE_DIR,"output_data","plot-data")
alarm_data_dir = os.path.join(BASE_DIR,"output_data","cffex-host-alarm")
# output_dir= os.path.join(BASE_DIR,"output_data")
plot_dir = os.path.join(BASE_DIR,"output_data","kpi-plot")
predict_data_dir =  os.path.join(BASE_DIR,"output_data","predicting_data")
origin_dir = ORIGIN_DATA
pre_dir = PRE_DATA
raw_data_dir = os.path.join(BASE_DIR, 'raw_data').replace('\\', '/')

#告警事件原始数据路径
host_alarm_dir = os.path.join(BASE_DIR,"raw_data","cffex-host-alarm")

#调用数据预处理的函数
def call_data_preprocessing_func(flag=False):
    if(flag):
        data_preprocessing.data_preprocessing_process(origin_dir,output_dir)
        data_preprocessing.generate_plot_data(output_dir,plot_data_dir)
        data_preprocessing.insert_missing_data(plot_data_dir)
        alarm_processed_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-processed.csv')
        node_alias_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-node-alias.csv')
        alarm_out_file = os.path.join(predict_data_dir, "alarm_data.csv")
        data_preprocessing.generate_alarm_data(alarm_processed_file, node_alias_file, alarm_out_file)

#调用特征提取的函数
def call_feature_extraction_func(flag=False):
    if(flag):
        predict_data = os.path.join(predict_data_dir, 'predict_data.csv')
        feature_extraction.generate_feature_by_hostname(plot_data_dir, predict_data)
        feature_extraction.get_host_name(os.path.join(plot_data_dir,'/alarmsvr1_cpu.csv'))
        data_preprocessing.check_completeness(plot_data_dir)

#调用预测模型的函数
def call_predict_model_func(flag=False):
    if(flag):
        merged_data_file = os.path.join(predict_data_dir,"merged_data.csv")
        model_save_file = os.path.join(predict_data_dir,"model_save.csv")
        predict_model.classifiers_for_prediction(merged_data_file, model_save_file)


if __name__ == '__main__':
    call_data_preprocessing_func()
    call_data_preprocessing_func()
    call_predict_model_func(flag=True)

