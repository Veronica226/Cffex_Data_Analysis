import datetime
import sys
import pandas as pd

from codes.preprocessing import data_preprocessing
from codes.feature_engineering import feature_extraction
from codes.model import predict_model, anomaly_detection
from codes.clustering import level_division
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

        deleted_alarm_processed_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-processed_deleted.csv')
        fixed_alarm_data_file =  os.path.join(alarm_data_dir, 'cffex-host-alarm-processed_fixed.csv')
        final_alarm_data_file =  os.path.join(alarm_data_dir, 'cffex-host-alarm-processed_final.csv')
        raw_alarm_processed_file =  os.path.join(origin_alarm_data_dir, 'cffex-host-alarm-processed.csv')

        alarm_origin_file = os.path.join(raw_data_dir, 'cffex-host-alarm', 'cffex-host-alarm-processed.csv')

        node_alias_file = os.path.join(alarm_data_dir, 'cffex-host-alarm-node-alias.csv')

        alarm_out_file = os.path.join(predict_data_dir, "alarm_data.csv")
        predict_data = os.path.join(predict_data_dir, 'predict_data.csv')
        alarm_out_file_fixed = os.path.join(predict_data_dir, "alarm_fixed_data.csv")
        alarm_out_file_final = os.path.join(predict_data_dir, "alarm_final_data.csv")
        alarm_out_file_cluster = os.path.join(cluster_data_dir, "cluster_alarm_data.csv")
        multicalss_alarm_out_file = os.path.join(multiclass_data_dir, "multiclass_alarm_data.csv")

        # #处理原始告警数据
        # data_preprocessing.process_alarm_data(os.path.join(raw_data_dir, 'cffex-host-alarm'), alarm_data_dir)
        # # 处理原始数据，将json格式的原始log文件数据解析为dataframe格式的csv文件数据
        # data_preprocessing.process_raw_data(origin_data_dir, output_cffex_info_dir)
        # # 提取output_data中的cffex-host-info数据的时间、最大值、最小值数据，将每个主机按cpu、磁盘、mem等部件存入plot_data中
        # #plot data实际上是用来生成分类器所需要的特征
        # data_preprocessing.generate_plot_data(output_cffex_info_dir, plot_data_dir)
        #
        # # plot_data中部分数据存在23点数据缺失问题，对数据进行线性插值处理
        # data_preprocessing.insert_missing_data(plot_data_dir, plot_data_dir) #测试一下

        #删除告警数据中的ping数据
        # data_preprocessing.delete_ping_data(alarm_processed_file,deleted_alarm_processed_file)

        #修改ping告警事件对应的主机名
        # data_preprocessing.fix_ping_data(alarm_processed_file,raw_alarm_processed_file,node_alias_file,fixed_alarm_data_file)
        #data_preprocessing.check_ping_alarm_data(fixed_alarm_data_file,final_alarm_data_file)

        # 将特征数据与告警数据match到一起，按照主机名和时间 左连接将告警事件match到对应的特征数据中
        data_preprocessing.generate_alarm_data(final_alarm_data_file, node_alias_file, multicalss_alarm_out_file)

        #处理原始告警数据
        #data_preprocessing.process_alarm_data(os.path.join(raw_data_dir, 'cffex-host-alarm'), alarm_data_dir)
        # 处理原始数据，将json格式的原始log文件数据解析为dataframe格式的csv文件数据
        #data_preprocessing.process_raw_data(origin_data_dir, output_cffex_info_dir)
        # 提取output_data中的cffex-host-info数据的时间、最大值、最小值数据，将每个主机按cpu、磁盘、mem等部件存入plot_data中
        #plot data实际上是用来生成分类器所需要的特征
        #data_preprocessing.generate_plot_data(output_cffex_info_dir, plot_data_dir)

        # plot_data中部分数据存在23点数据缺失问题，对数据进行线性插值处理
        #data_preprocessing.insert_missing_data(plot_data_dir, plot_data_dir) #测试一下
        # 将告警数据只存储主机名、时间和事件（bool标记）
        #data_preprocessing.generate_alarm_data(alarm_processed_file, node_alias_file, alarm_out_file)
        #
        # data_preprocessing.genereate_host_event_sets(alarm_origin_file, plot_dir)
        # data_preprocessing.generate_alarm_level_content(alarm_origin_file, os.path.join(raw_data_dir, 'cffex-host-alarm'))

#调用特征提取的函数
def call_feature_extraction_func(flag=False):
    if(flag):
        predict_data = os.path.join(predict_data_dir, 'predict_data.csv')
        alarm_file = os.path.join(predict_data_dir, 'alarm_data.csv')
        alarm_file_fixed = os.path.join(predict_data_dir, "alarm_fixed_data.csv")
        alarm_file_final = os.path.join(predict_data_dir, "alarm_final_data.csv")
        merged_data_file = os.path.join(predict_data_dir, "merged_data.csv")
        history_data_file = os.path.join(predict_data_dir, "history_data.csv")
        merged_history_file =os.path.join(predict_data_dir, "merged_history_data.csv")    #删掉ping数据
        merged_fixed_file = os.path.join(predict_data_dir, "merged_fixed_data.csv")      #修改ping数据对应的正确主机
        merged_final_file = os.path.join(predict_data_dir, "merged_final_data.csv")
        no_cpu_file = os.path.join(predict_data_dir, "no_cpu_data.csv")
        no_disk_file = os.path.join(predict_data_dir, "no_disk_data.csv")
        no_mem_file = os.path.join(predict_data_dir, "no_mem_data.csv")
        cpu_only_file = os.path.join(predict_data_dir, "cpu_only_data.csv")
        disk_only_file = os.path.join(predict_data_dir, "disk_only_data.csv")
        mem_only_file = os.path.join(predict_data_dir, "mem_only_data.csv")

        cluster_history_data_file = os.path.join(cluster_data_dir, "cluster_history_data.csv")
        alarm_file_cluster = os.path.join(cluster_data_dir, "cluster_alarm_data.csv")
        cluster_series_data_file= os.path.join(cluster_data_dir, "cluster_series_data.csv")

        multicalss_alarm_out_file = os.path.join(multiclass_data_dir, "multiclass_alarm_data.csv")
        multiclass_data_file = os.path.join(multiclass_data_dir, "multiclass_data.csv")



        #将每个主机的cpu、六个公共磁盘、内存的最大值、最小值作为特征，整合到同一个dataframe中，并将所有主机的dataframe拼接在一起，形成一个特征矩阵
        # feature_extraction.generate_feature_by_hostname(plot_data_dir, predict_data)

        # 将predict_data中各主机的特征数据独立存储成csv文件，供matlab画图使用。在feature_extraction之后才会生成predict_data
        # generatesubplot实际上是用来获得Matlab画图的数据，获得了predict_data之后才可以执行这条语句
        # data_preprocessing.generate_subplot_data(predict_data, subplot_data_dir)

        #获取时间序列的历史特征
        #feature_extraction.generate_history_feature(plot_data_dir,history_data_file)

        # 将特征数据与告警数据match到一起，按照主机名和时间 左连接将告警事件match到对应的特征数据中
        feature_extraction.generate_data_matrix_and_vector(history_data_file,multicalss_alarm_out_file,multiclass_data_file)

        #保留部分特征
        # feature_extraction.delete_feature(merged_final_file,no_disk_file)

        #生成聚类所用的特征历史数据
        # feature_extraction.generate_cluster_history_data(plot_data_dir,cluster_history_data_file)

        # feature_extraction.generate_cluster_data(history_data_file,multicalss_alarm_out_file,multiclass_data_file)

        #获取按主机的时间序列分解的数据
        #data_preprocessing.generate_kpi_data_decomposition(merged_data_file, host_data_dir)

#调用分类器预测模型的函数
def call_predict_model_func(flag=False):
    if(flag):
        predict_proba_file = os.path.join(predict_data_dir,"predict_proba.csv")
        history_predict_proba_file = os.path.join(predict_data_dir,"history_predict_proba.csv")
        merged_data_file = os.path.join(predict_data_dir,"merged_data.csv")
        model_save_file = os.path.join(predict_data_dir,"model_save.csv")
        merged_history_file = os.path.join(predict_data_dir, "merged_history_data.csv")
        merged_fixed_file = os.path.join(predict_data_dir, "merged_fixed_data.csv")
        no_cpu_file = os.path.join(predict_data_dir, "no_cpu_data.csv")
        no_disk_file = os.path.join(predict_data_dir, "no_disk_data.csv")
        no_mem_file = os.path.join(predict_data_dir, "no_mem_data.csv")
        cpu_only_file = os.path.join(predict_data_dir, "cpu_only_data.csv")
        disk_only_file = os.path.join(predict_data_dir, "disk_only_data.csv")
        mem_only_file = os.path.join(predict_data_dir, "mem_only_data.csv")
        multiclass_data_file = os.path.join(multiclass_data_dir, "multiclass_data.csv")


        # #包含若干分类器的预测模型
        # print('no cpu')
        # predict_model.classifiers_for_prediction(no_cpu_file, model_save_file,history_predict_proba_file)
        # print('no disk')
        # predict_model.classifiers_for_prediction(no_disk_file, model_save_file,history_predict_proba_file)
        # print('no mem')
        # predict_model.classifiers_for_prediction(no_mem_file, model_save_file,history_predict_proba_file)
        # print('only cpu')
        # predict_model.classifiers_for_prediction(cpu_only_file, model_save_file,history_predict_proba_file)
        # print('only disk')
        # predict_model.classifiers_for_prediction(disk_only_file, model_save_file,history_predict_proba_file)
        # print('only mem')
        # predict_model.classifiers_for_prediction(mem_only_file, model_save_file,history_predict_proba_file)

        predict_model.classifiers_for_prediction(multiclass_data_file, model_save_file, history_predict_proba_file)


def call_level_division_func(flag=False):
    if(flag):
        cluster_series_data_file =os.path.join(cluster_data_dir, "cluster_series_data.csv")
        # level_division.hierarchical_clusterting()
        # level_division.get_cluster_data(cluster_series_data_file)
        level_division.hierarchical_clusterting(cluster_series_data_file,4)




def call_anomaly_detection_func(flag=False):
    if(flag):
        #对每个主机训练异常检测模型
        #anomaly_detection.generate_time_series_decomposition_model(host_data_dir)
        #训练完模型之后会输出测试集的结果文件，下面需要针对结果计算相关指标
        #anomaly_detection.calc_model_evaluation_score(host_data_dir)
        #计算所有主机那些评价指标的平均值
        anomaly_detection.calc_host_ave_model_evaluation_score(host_data_dir)
        return

if __name__ == '__main__':
    call_data_preprocessing_func()
    call_feature_extraction_func()

    call_predict_model_func()
    call_anomaly_detection_func()
    call_level_division_func(flag=True)

