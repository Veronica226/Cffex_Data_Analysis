#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import os, sys, json, csv, re

common_disk_list = ['boot', 'rt', 'home', 'monitor', 'tmp']  #通过generate_plot_data得到所有主机公共的磁盘目录
######################################################################################
#Author: 王靖文

#日期转换
def trans_date(date_str):
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8] + ' ' + date_str[8:] + ':00:00'

def trans_alarm_date(date_str):
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:11] + ':00:00'

#最开始要处理的程序，把raw_data里指标数据的log文件转成csv文件
def process_raw_data(origin_dir, output_dir):

    f_list = os.listdir(origin_dir)
    num = 0
    for i in f_list:  ##每个log文件
        if os.path.splitext(i)[1] == '.log':
            file_name = os.path.join(output_dir, os.path.splitext(i)[0] + '.csv')
            # if not os.path.exists(file_name):
            #     os.makedirs(file_name)
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
                #print(data_list)
                print(list(data_list[0].keys()))

                df = pd.DataFrame(data_list)
                print(df.shape)
                df.to_csv(file_name,sep=',',index=False)
            num += 1
            if(num > 1):
                break
    print('process raw data finished!')

#只获取时间、最大值、最小值特征，一方面为了画图使用，另一方面为了后续合成特征
def generate_plot_data(origin_dir, output_dir):
    f_list = os.listdir(origin_dir)
    for file in f_list:
        file_path = os.path.join(origin_dir, file)
        file_name = os.path.splitext(file)[0]
        file_name_list = file_name.split('_')
        h = file_name_list.index("hourly")
        host_name_list = []
        for a in range(1, h):   #有的主机名用下划线连接
            host_name_list.append(file_name_list[a])  #主机名可能带有下划线，名字很长
        host_name = '_'.join(host_name_list)
        device_name = file_name_list[-1]  #设备名称是第-1个元素（list从末尾往前数）
        print('host_name = {0}, device_name = {1}'.format(host_name, device_name))
        if file_name.endswith("disk"):  # 磁盘文件中diskname字段有不同的磁盘名
            data = pd.read_csv(file_path, usecols=['archour','diskname', 'maxvalue','minvalue'], dtype=str)
            for diskname, group in data.groupby('diskname'):   #对diskname分组存储到不同文件中
                disk_name = 'rt' if len(diskname) == 1 and diskname[0] == '/' else diskname[1:]
                disk_name = disk_name.replace('/', '_')
                output_file_name = '_'.join([host_name, device_name, disk_name]) + '.csv'
                output_file = os.path.join(output_dir, output_file_name)
                group.drop(['diskname'],axis=1, inplace=True)
                group['archour'] = group['archour'].apply(trans_date)
                group.to_csv(output_file, sep=',', index=False, header=False)
        else:
            output_file_name = host_name+ '_' + device_name + '.csv'
            output_file = os.path.join(output_dir, output_file_name) #主机名 部件名
            data = pd.read_csv(file_path,usecols=['archour','maxvalue','minvalue'], dtype=str)  #时间 最大值 最小值
            data['archour'] = data['archour'].apply(trans_date)
            data.to_csv(output_file, sep=',', index=False, header=False)
    print('generate plot data finished!')

#对数据缺失的文件进行插值处理，取平均
def insert_missing_data(origin_dir, output_dir):
     f_list = os.listdir(origin_dir)
     if(not os.path.exists(output_dir)):
         os.makedirs(output_dir)
     for file in f_list:
        file_path = os.path.join(origin_dir, file)
        file_name = os.path.splitext(file)[0]
        output_file_path = os.path.join(output_dir, file_name + '.csv')
        df = pd.read_csv(file_path, sep=',', header=None, names=['archour', 'maxvalue', 'minvalue'], dtype=str)
        df['archour'] = df['archour'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df[['maxvalue', 'minvalue']] = df[['maxvalue', 'minvalue']].apply(np.float64)
        df_out = pd.DataFrame(columns=df.columns)
        row_num = 0
        for idx in range(df.shape[0]):
            hour_data = df.loc[idx]
            df_out.loc[row_num] = df.loc[idx]
            row_num += 1
            if(hour_data['archour'].hour == 22):
                now_time = hour_data['archour'] + timedelta(hours=1)
                if(idx < df.shape[0] - 1):
                    nxt_hour_data = df.loc[idx + 1]
                    now_max_value = (hour_data['maxvalue'] + nxt_hour_data['maxvalue']) / 2
                    now_min_value = (hour_data['minvalue'] + nxt_hour_data['minvalue']) / 2
                else:
                    pre_hour_data = df.loc[idx - 1]
                    now_max_value = hour_data['maxvalue'] * 2 - pre_hour_data['maxvalue']
                    now_min_value = hour_data['minvalue'] * 2 - pre_hour_data['minvalue']
                now_data_dict = {'archour': now_time, 'maxvalue': now_max_value, 'minvalue': now_min_value}
                df_out.loc[row_num] = pd.Series(now_data_dict)
                row_num += 1
        df_out.to_csv(output_file_path, sep=',', index=False, header=False, float_format='%.1f')

'''
#获取缺失index的list 采用线性插值的方法
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
                max =str(round((float(info1[1])+float(info2[1]))/2,1) )
                min =str(round((float(info1[2])+float(info2[2]))/2,1) )
            else:
                info2 = lines[i-1].split(',')
                max = str(round((float(info1[1])*2 - float(info2[1])),1))
                min = str(round((float(info1[2])*2 - float(info2[2])),1))

            lines.insert(i+1, date + ',' + max + ',' + min)
            # print(lines)
            cnt = cnt+1

        data = '\n'.join(lines)
        with open(os.path.join(origin_dir,file_name), "w") as fp:
            fp.write(data)
            # print("asuc")

#找到文件缺失的行索引
def find_missing_loc(origin_dir,file_name):
    loc_list = []
    with open(os.path.join(origin_dir,file_name), "r") as fp:
        data = fp.read()
        lines = data.split('\n')  #line list []
        len = lines.__len__()
        cnt = 0
        # print (lines[len-2])
        # if lines[len-2]!= ''and lines[len-2][0] != ''and lines[len-2][0][11] is '2' and lines[len-2][0][12] is '2':      #判断是否为数据缺失文件

        for i in range(0,len):
                cnt = cnt+1
                info = lines[i].split(',')   #row info list
                date = info[0]  #date string
                if date != '' and date[11]is'2' and date[12]is'2':  #如果用read().split('\n')划分，则list最后一个元素是''
                    loc_list.append(cnt-1)
    print (loc_list)
    return loc_list

def find_missing_files(origin_dir,file_name):
    with open(os.path.join(origin_dir, file_name), "r") as fp:
        data = fp.read()
        lines = data.split('\n')  # line list []
        i=0
        while(lines[i]!=''):
            i=i+1
        info = lines[i-1].split(',')
        date = info[0]
        if date != ''and date != ''and date[11] is '2' and date[12] is '2':
            return 1#判断是否为数据缺失文件
        else:
            return 0
'''

#检查所有文件是否数据完整  使用shape[0]是否能对24整除判断
def check_completeness(origin_dir):
    f_list = os.listdir(origin_dir)
    for file_name in f_list:
        with open(origin_dir + "/" + file_name, "r") as fp1:  # 通过时间字段 对hostname的不同部件的max min值merge到同一个dataframe中
            data = pd.read_csv(fp1, sep=',', dtype=str, header=None, index_col=None)  # header=None设置列名为空，自动用0开头的数字替代
            row_num = data.shape[0]
            if row_num % 24 != 0:
                print (file_name)
                print(row_num)

def generate_alarm_data(alarm_processed_file,node_alias_file,alarm_out_file):
    node_dict = csv_to_dict(node_alias_file)
    data = pd.read_csv(alarm_processed_file, sep=',', dtype=str, usecols=['node_alias','last_time','alarm_level'])  #提取告警事件文件内的主机、时间、事件内容
    data['node_alias'] = data['node_alias'].apply(find_node_alias_value,node_dict = node_dict)  #node数字转成对应主机名称
    data['last_time'] = data['last_time'].apply(trans_alarm_date)   #修改日期格式
    data['alarm_level'] = '1'    #将事件全部赋值为1
    data.columns = ['hostname', 'archour','event']
    print (data)
    data.to_csv(alarm_out_file, sep=',', index=False)

def csv_to_dict(file_name):
    node_dict={}     #创建node-alias文件对应的字典
    with open(file_name)as f:
        data = csv.reader(f,delimiter=',')
        for row in data:
            node_dict[row[0]]=row[1]  #每行第一个元素为key  第二个元素为value
    print (node_dict)
    return node_dict

def find_node_alias_value(node_key,node_dict): #在node_dict中 找到id对应node_alias 也就是主机名
    node_value = node_dict[node_key]
    return node_value.lower()        #全部转换为小写

def generate_subplot_data(predict_data, subplot_data_dir):
    data = pd.read_csv(predict_data, sep=',',dtype=str)
    for hostname,group in data.groupby('hostname'):
        subplot_data_file = os.path.join(subplot_data_dir,hostname+'.csv')
        group.drop(['hostname'], axis=1, inplace=True)
        group.to_csv(subplot_data_file, sep=',', index=False, header=False)


######################################################################################
#Author: 普俊韬

#START 'cffex-host-alarm.csv' process code
def process_alarm_data(host_alarm_dir, output_dir):

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




