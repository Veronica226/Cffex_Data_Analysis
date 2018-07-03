#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import DataFrame
import pandas as pd
import numpy as np
import os, sys, json, csv, re

#日期转换
def trans_date(date_str):
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8] + ' ' + date_str[8:] + ':00:00'

#数据预处理
def data_preprocessing_process(origin_dir, output_dir, host_alarm_dir):
    f_list = os.listdir(origin_dir)
    for i in f_list:  ##每个log文件
        if os.path.splitext(i)[1] == '.log':
            host_info_dir = os.path.join(output_dir, "cffex-host-info")
            file_name = os.path.join(host_info_dir, os.path.splitext(i)[0] + '.csv')
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


#START 'cffex-host-alarm.csv' process code
    #Author: 普俊韬
    #Last update: 20180627

    #定义原始数据路径
    host_alarm_rawdata_dir = os.path.join(host_alarm_dir,"cffex-host-alarm.csv")

    #定义告警事件数据路径
    host_alarm_content_dir = os.path.join(host_alarm_dir,"cffex-host-alarm-content.csv")

    #定义告警数据处理后输出路径
    #处理后的告警数据总表
    host_alarm_processed_dir = os.path.join(output_dir,"cffex-host-alarm","cffex-host-alarm-processed.csv")
    
    #处理后的告警组件数据
    host_alarm_component_dir = os.path.join(output_dir,"cffex-host-alarm","cffex-host-alarm-component.csv")
    
    #处理后的告警类别数据
    host_alarm_category_dir = os.path.join(output_dir,"cffex-host-alarm","cffex-host-alarm-category.csv")

    #处理后的组件别名数据
    host_alarm_node_alias_dir = os.path.join(output_dir,"cffex-host-alarm","cffex-host-alarm-node-alias.csv")


    #TODO：进行原始数据的分割处理

    #读取原始数据（GBK编码，数据不带列标题）
    data = pd.read_csv(host_alarm_rawdata_dir, encoding = 'GBK', header = None)

    #分割数据字段，分隔符'|||'（'[|]+'),分割后扩展列
    data_processed = data[0].str.split('[|]+',expand = True)

    #插入列标题
    data_processed.columns = ['node_name', 'node_alias', 'component', 'category', 'alarm_count', 'first_time', 'last_time', 'alarm_level', 'alarm_content']


    #TODO：进行'component'字段的处理

    #将'component'字段提取出来作为一个DataFrame
    data_component = data_processed.loc[:,['component']]

    #去掉重复数据
    data_component_processed = data_component.drop_duplicates()

    #插入id列，编号从1开始
    data_component_processed['id'] = range(1,len(data_component_processed) + 1)

    #将列顺序调整为['id', 'component']
    data_component_processed = data_component_processed[['id','component']]

    #将处理后结果写入'cffex-host-alarm-component.csv'（不带行标签，GBK编码）
    data_component_processed.to_csv(host_alarm_component_dir, index = 0, encoding = 'GBK')


    #TODO：进行'category'字段的处理

    #将'category'字段提取出来作为一个DataFrame
    data_category = data_processed.loc[:,['category']]

    #去掉重复数据
    data_category_processed = data_category.drop_duplicates()

    #插入id列，编号从1开始
    data_category_processed['id'] = range(1,len(data_category_processed) + 1)

    #将列顺序调整为['id', 'category']
    data_category_processed = data_category_processed[['id','category']]

    #将处理后结果写入'cffex-host-alarm-category.csv'（不带行标签，GBK编码）
    data_category_processed.to_csv(host_alarm_category_dir, index = 0, encoding = 'GBK')


    #TODO：进行'node_alias'字段的处理

    #将'node_alias'字段提取出来作为一个DataFrame
    data_node_alias = data_processed.loc[:,['node_alias']]

    #去掉重复数据
    data_node_alias_processed = data_node_alias.drop_duplicates()

    #插入id列，编号从1开始
    data_node_alias_processed['id'] = range(1,len(data_node_alias_processed) + 1)

    #将列顺序调整为['id', 'node_alias']
    data_node_alias_processed = data_node_alias_processed[['id','node_alias']]

    #将处理后结果写入'cffex-host-alarm-node-alias.csv'（不带行标签，GBK编码）
    data_node_alias_processed.to_csv(host_alarm_node_alias_dir, index = 0, encoding = 'GBK')



    #TODO：将'component','category'和'node_alias'字段替换为对应的'id'值，方便后续的数据处理

    #对'component'字段进行查找和替换
    data_processed['component'] = data_processed['component'].replace(data_component_processed['component'].tolist(),data_component_processed['id'].tolist())

    #对'category'字段进行查找和替换
    data_processed['category'] = data_processed['category'].replace(data_category_processed['category'].tolist(),data_category_processed['id'].tolist())

    #对'node_alias'字段进行查找和替换
    data_processed['node_alias'] = data_processed['node_alias'].replace(data_node_alias_processed['node_alias'].tolist(),data_node_alias_processed['id'].tolist())

    #TODO: 将'alarm_content'字段替换为相应的'id'值，方便后续的数据处理

    #读入告警事件表'cffex-host-alarm-content.csv'
    data_processed_content = pd.read_csv(host_alarm_content_dir,encoding = 'GBK')

    #替换函数定义
    def re_replace(data):
        for i in range(len(data_processed_content['id'])):
            #正则表达式和字符串不存在匹配串：继续遍历
            if re.match(data_processed_content['regular_expression'][i],data) == None:
                continue
            #正则表达式和字符串存在匹配串：替换id值并返回保存
            else:
                data = str(data_processed_content['id'][i])
                return data

    #调用替换函数
    data_processed['alarm_content'] = data_processed['alarm_content'].apply(re_replace)

    #将处理后结果写入'cffex-host-alarm-processed.csv'（不带行标签，GBK编码）
    data_processed.to_csv(host_alarm_processed_dir, index = 0, encoding = 'GBK')

#END 'cffex-host-alarm.csv' process code


def generate_plot_data(origin_dir, output_dir):
    f_list = os.listdir(origin_dir)
    for i in f_list:
        file_name = os.path.splitext(i)[0]
        file_name_list = file_name.split('_')

        with open(origin_dir + "/" + i, "r") as fp1:
            if file_name.endswith("disk"):           #磁盘文件中diskname字段有不同的磁盘名
                data = pd.read_csv(fp1, usecols=[0, 3, 6, 8], dtype=str)
                for diskname, group in data.groupby('diskname'):   #对diskname分组存储到不同文件中
                    # print (diskname)
                    # print (group)
                    output_file = os.path.join(output_dir, file_name_list[1] + '_' + file_name_list[3] + '_' + diskname[1:].replace('/', '-') + '.csv')
                    #del group['diskname']
                    group.drop(['diskname'],axis=1, inplace=True)
                    group['archour'] = group['archour'].apply(trans_date)
                    group.to_csv(output_file, sep=',', index=False, header=False)

            elif file_name.endswith("disk")==False:
                output_file = os.path.join(output_dir,file_name_list[1]+'_'+file_name_list[3]+'.csv') #主机名 部件名
                data = pd.read_csv(fp1,usecols=[0,5,7], dtype=str)  #时间 最大值 最小值
                data['archour'] = data['archour'].apply(trans_date)
                data.to_csv(output_file, sep=',', index=False, header=False)

def linear_insert(origin_dir, output_dir):
    with open("","r") as fp:
        data = fp.read()
        lines = data.split('\n')
        lines.insert(5,"new line")
        data = '\n'.join(lines)
        with open("","w") as fp:
            fp.write(data)

def insert_missing_data(origin_dir):
     f_list = os.listdir(origin_dir)
     for file_name in f_list:
        f_name = os.path.splitext(file_name)[0]
        if f_name.endswith("cpu") or f_name.endswith("mem"):
            print(file_name)
            loc_list = find_missing_loc(origin_dir,file_name)
            # print (loc_list)
            insert_multirows(origin_dir,file_name,loc_list)
            print(file_name+"success")

def insert_multirows(origin_dir,file_name,loc_list):
    with open(os.path.join(origin_dir,file_name), "r") as fp:
        data = fp.read()
        lines = data.split('\n')
        # loc_list = [22,45]#
        len = loc_list.__len__()
        cnt = 0
        for i in loc_list:
            i = i+cnt
            info1 = lines[i].split(',')
            loc = 12
            date = info1[0][:12]+'3'+info1[0][loc+1:]   #change date

            if(i<len):
                info2 = lines[i+1].split(',')
                max =str( (float(info1[1])+float(info2[1]))/2 )
                min =str( (float(info1[2])+float(info2[2]))/2 )
            else:
                info2 = lines[i-1].split(',')
                max = str( (float(info1[1])*2 - float(info2[1])))
                min = str((float(info1[2])*2 - float(info2[2])))

            lines.insert(i+1, date + ',' + max + ',' + min)
            # print(lines)
            cnt = cnt+1

        data = '\n'.join(lines)
        with open(os.path.join(origin_dir,file_name), "w") as fp:
            fp.write(data)
            # print("asuc")

def find_missing_loc(origin_dir,file_name):
    loc_list = []
    with open(os.path.join(origin_dir,file_name), "r") as fp:
        data = fp.read()
        lines = data.split('\n')  #line list []
        len = lines.__len__()
        cnt = 0
        for i in range(0,len):
            cnt = cnt+1
            info = lines[i].split(',')   #row info list
            date = info[0]  #date string
            if date != '' and date[11]is'2' and date[12]is'2':
                loc_list.append(cnt-1)

        # print(loc_list)
    return loc_list



