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
        alarm_processed_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-processed.csv')
        node_alias_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-node-alias.csv')
        alarm_out_file = os.path.join(predict_data_dir, "alarm_data.csv")

        # 处理原始数据，将json格式的原始log文件数据解析为dataframe格式的csv文件数据
        #data_preprocessing.process_raw_data(origin_data_dir, output_cffex_info_dir)
        # 提取output_data中的cffex-host-info数据的时间、最大值、最小值数据，将每个主机按cpu、磁盘、mem等部件存入plot_data中
        #data_preprocessing.generate_plot_data(output_cffex_info_dir, plot_data_dir)

        # plot_data中部分数据存在23点数据缺失问题，对数据进行线性插值处理
        data_preprocessing.insert_missing_data(plot_data_dir, plot_data_dir) #测试一下
        # 将特征数据与告警数据match到一起，按照主机名和时间 左连接将告警事件match到对应的特征数据中
        #data_preprocessing.generate_alarm_data(alarm_processed_file, node_alias_file, alarm_out_file)
        # 将predict_data中各主机的特征数据独立存储成csv文件，供matlab画图使用
        #data_preprocessing.generate_subplot_data(predict_data, subplot_data_dir)
        # 检查plot_data数据完整性
        #data_preprocessing.check_completeness(plot_data_dir)

#调用特征提取的函数
def call_feature_extraction_func(flag=False):
    if(flag):
        predict_data = os.path.join(predict_data_dir, 'predict_data.csv')
        alarm_file = os.path.join(predict_data_dir, 'alarm_data.csv')
        merged_data_file = os.path.join(predict_data_dir, "merged_data.csv")

        #将每个主机的cpu、六个公共磁盘、内存的最大值、最小值作为特征，整合到同一个dataframe中，并将所有主机的dataframe拼接在一起，形成一个特征矩阵
        feature_extraction.generate_feature_by_hostname(plot_data_dir, predict_data)
        # 将特征数据与告警数据match到一起，按照主机名和时间 左连接将告警事件match到对应的特征数据中
        feature_extraction.generate_data_matrix_and_vector(predict_data,alarm_file,merged_data_file)

#调用预测模型的函数
def call_predict_model_func(flag=False):
    if(flag):
        predict_proba_file = os.path.join(predict_data_dir,"predict_proba.csv")
        merged_data_file = os.path.join(predict_data_dir,"merged_data.csv")
        model_save_file = os.path.join(predict_data_dir,"model_save.csv")

        #包含若干分类器的预测模型
        predict_model.classifiers_for_prediction(merged_data_file, model_save_file,predict_proba_file)


if __name__ == '__main__':
    call_data_preprocessing_func(flag=True)
    call_feature_extraction_func()
    call_predict_model_func()

