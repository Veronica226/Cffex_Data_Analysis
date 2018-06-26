#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import DataFrame
import pandas as pd
import numpy as np
import os, sys, json, csv, re

sys.path.append("..")
BASE_DIR = os.path.abspath('..')

ORIGIN_DATA = os.path.join(BASE_DIR, "raw_data", "cffex-host-info")
PRE_DATA = os.path.join(BASE_DIR, "raw_data", "pre_data")

output_dir= os.path.join(BASE_DIR,"output_data")
origin_dir = ORIGIN_DATA
pre_dir = PRE_DATA

#告警事件相关参数
#普俊韬发现但未能解决问题：
#在我的电脑上需要在BASE_DIR后加上"Cffex_Data_Analysis"才能访问到正确的数据
# host_alarm_dir = os.path.join(BASE_DIR,"Cffex_Data_Analysis","raw_data","cffex-host-alarm.csv")
# host_alarm_processed_dir = os.path.join(BASE_DIR,"Cffex_Data_Analysis","output_data","cffex-host-alarm-processed.csv")
# host_alarm_component_dir = os.path.join(BASE_DIR,"Cffex_Data_Analysis","output_data","cffex-host-alarm-component.csv")
# host_alarm_category_dir = os.path.join(BASE_DIR,"Cffex_Data_Analysis","output_data","cffex-host-alarm-category.csv")

def data_preprocessing_process(origin_dir, output_dir):
    f_list = os.listdir(origin_dir)
    for i in f_list:  ##每个log文件
        if os.path.splitext(i)[1] == '.log':
            file_name = os.path.join(output_dir, os.path.splitext(i)[0] + '.csv')
            # if not os.path.exists(file_name):
            #     os.makedirs(file_name)
            data = {}
            keys = []
            with open(origin_dir + "/" + i, "r") as fp1:
                origin_data = fp1.read()
                origin_data_list = origin_data.split(']')[:-1]  # 每一天的数据组成的list
                data_list = []  # json dict list
                for item in origin_data_list:
                    if (len(item) > 0 and item[-1] != ']'):
                        item = item + ']'
                    hour_data_list = item[1:-1].split('}, ')[:-1]
                    # print(hour_data_list)
                    for hour_data in hour_data_list:
                        if (len(hour_data) > 0 and item[-1] != '}'):
                            hour_data = hour_data + '}'
                        data_list.append(json.loads(hour_data))  # json list
                        # print(hour_data)
                print(data_list)
                print(list(data_list[0].keys()))
                data_dict = dict(zip(list(data_list[0].keys()), [[] for i in range(len(data_list[0].keys()))]))

                for data_item in data_list:
                    for key, value in data_item.items():
                        data_dict[key].append(value)

                df = pd.DataFrame(data_dict)
                df.to_csv(file_name,sep=',',index=False)

                # keys = data_list[0].keys()
                # data = data.fromkeys(keys, [])
                #
                # if not os.path.exists(file_name):
                #     with open(file_name,"w") as fp2:
                #         w = csv.DictWriter(fp2,keys)
                #         w.writeheader()
                #         for data_dict in data_list:
                #             w.writerow(data_dict)
                #
                #
                #             for key in keys:
                #                 data[key].append(data_dict[key])


# #START 'cffex-host-alarm.csv' process code
#     #Author: 普俊韬
#     #Last update: 20180626

#     #导入pandas、numpy库
#     # import pandas as pd
#     # import numpy as np


#     #TODO：进行原始数据的分割处理

#     #读取原始数据（GBK编码，数据不带列标题）
#     data = pd.read_csv(host_alarm_dir, encoding = 'GBK', header = None)

#     #分割数据字段，分隔符'|||'（'[|]+'),分割后扩展列
#     data_processed = data[0].str.split('[|]+',expand = True)

#     #插入列标题
#     data_processed.columns = ['node_name', 'node_alias', 'component', 'category', 'alarm_count', 'first_time', 'last_time', 'alarm_level', 'alarm_content']


#     #TODO：进行'component'字段的处理

#     #将'component'字段提取出来作为一个DataFrame
#     data_component = data_processed.loc[:,['component']]

#     #去掉重复数据
#     data_component_processed = data_component.drop_duplicates()

#     #插入id列，编号从1开始
#     data_component_processed['id'] = range(1,len(data_component_processed) + 1)

#     #将列顺序调整为['id', 'component']
#     data_component_processed = data_component_processed[['id','component']]

#     #将处理后结果写入'cffex-host-alarm-component.csv'（不带行标签，GBK编码）
#     data_component_processed.to_csv(host_alarm_component_dir, index = 0, encoding = 'GBK')


#     #TODO：进行'category'字段的处理

#     #将'category'字段提取出来作为一个DataFrame
#     data_category = data_processed.loc[:,['category']]

#     #去掉重复数据
#     data_category_processed = data_category.drop_duplicates()

#     #插入id列，编号从1开始
#     data_category_processed['id'] = range(1,len(data_category_processed) + 1)

#     #将列顺序调整为['id', 'category']
#     data_category_processed = data_category_processed[['id','category']]

#     #将处理后结果写入'cffex-host-alarm-category.csv'（不带行标签，GBK编码）
#     data_category_processed.to_csv(host_alarm_category_dir, index = 0, encoding = 'GBK')


#     #TODO：将'component'和'category'字段替换为对应的'id'值，方便后续的数据处理

#     #对'component'字段进行查找和替换
#     data_processed['component'] = data_processed['component'].replace(data_component_processed['component'].tolist(),data_component_processed['id'].tolist())

#     #对'category'字段进行查找和替换
#     data_processed['category'] = data_processed['category'].replace(data_category_processed['category'].tolist(),data_category_processed['id'].tolist())

#     #将处理后结果写入'cffex-host-alarm-processed.csv'（不带行标签，GBK编码）
#     data_processed.to_csv(host_alarm_processed_dir, index = 0, encoding = 'GBK')

# #END 'cffex-host-alarm.csv' process code


