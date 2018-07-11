import datetime
import sys
import pandas as pd

from codes.preprocessing import data_preprocessing
from codes.feature_engineering import feature_extraction
from codes.model import predict_model
from settings import *
import os





pre_dir = pre_data_dir
raw_data_dir = os.path.join(base_dir, 'raw_data').replace('\\', '/')

#告警事件原始数据路径
host_alarm_dir = os.path.join(base_dir,"raw_data","cffex-host-alarm")

#调用数据预处理的函数
def call_data_preprocessing_func(flag=False):
    if(flag):
        # data_preprocessing.data_preprocessing_process(origin_data_dir,output_cffex_info_dir)
        # data_preprocessing.generate_plot_data(output_cffex_info_dir,plot_data_dir)
         data_preprocessing.insert_missing_data(plot_data_dir)
        # alarm_processed_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-processed.csv')
        # node_alias_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-node-alias.csv')
        # alarm_out_file = os.path.join(predict_data_dir, "alarm_data.csv")
        # data_preprocessing.generate_alarm_data(alarm_processed_file, node_alias_file, alarm_out_file)

#调用特征提取的函数
def call_feature_extraction_func(flag=False):
    if(flag):
        predict_data = os.path.join(predict_data_dir, 'predict_data.csv')
        alarm_file = os.path.join(predict_data_dir, 'alarm_data.csv')
        merged_data_file = os.path.join(predict_data_dir, "merged_data.csv")
        feature_extraction.generate_data_matrix_and_vector(predict_data,alarm_file,merged_data_file)
        # feature_extraction.generate_feature_by_hostname(plot_data_dir, predict_data)
        # feature_extraction.get_host_name(os.path.join(plot_data_dir,'/alarmsvr1_cpu.csv'))
        # data_preprocessing.check_completeness(plot_data_dir)

#调用预测模型的函数
def call_predict_model_func(flag=False):
    if(flag):
        merged_data_file = os.path.join(predict_data_dir,"merged_data.csv")
        model_save_file = os.path.join(predict_data_dir,"model_save.csv")
        predict_model.classifiers_for_prediction(merged_data_file, model_save_file)


if __name__ == '__main__':
    call_data_preprocessing_func()
    call_feature_extraction_func()
    call_predict_model_func(flag=True)

