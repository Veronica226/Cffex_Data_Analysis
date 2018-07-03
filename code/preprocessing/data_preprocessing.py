#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import DataFrame
import pandas as pd
import os, sys, json, csv, re

sys.path.append("..")
BASE_DIR = os.path.abspath('..')

ORIGIN_DATA = os.path.join(BASE_DIR, "raw_data", "cffex-host-info")
PRE_DATA = os.path.join(BASE_DIR, "raw_data", "pre_data")

output_dir= os.path.join(BASE_DIR,"output_data")
origin_dir = ORIGIN_DATA
pre_dir = PRE_DATA

def trans_date(date_str):
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8] + ' ' + date_str[8:] + ':00:00'

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

def generate_plot_data(origin_dir, output_dir):
    f_list = os.listdir(origin_dir)
    for i in f_list:
        file_name = os.path.splitext(i)[0]
        file_name_list = file_name.split('_')
        h = file_name_list.index("hourly")
        host_name_list = []
        for a in range(1,h):
            host_name_list.append(file_name_list[a])
        host_name = '_'.join(host_name_list)

        with open(origin_dir + "/" + i, "r") as fp1:
            if file_name.endswith("disk"):           #磁盘文件中diskname字段有不同的磁盘名
                data = pd.read_csv(fp1, usecols=[0, 3, 6, 8], dtype=str)
                for diskname, group in data.groupby('diskname'):   #对diskname分组存储到不同文件中
                    # print (diskname)
                    # print (group)
                    output_file = os.path.join(output_dir, host_name + '_' + file_name_list[h+1] + '_' + diskname[1:].replace('/', '_') + '.csv')
                    # output_file = os.path.join(output_dir, file_name_list[1] + '_' + file_name_list[3] + '_' + diskname[1:].replace('/', '-') + '.csv')
                    #del group['diskname']
                    group.drop(['diskname'],axis=1, inplace=True)
                    group['archour'] = group['archour'].apply(trans_date)
                    group.to_csv(output_file, sep=',', index=False, header=False)

            elif file_name.endswith("disk")==False:
                output_file = os.path.join(output_dir, host_name+ '_'+file_name_list[h+1]+'.csv') #主机名 部件名
                data = pd.read_csv(fp1,usecols=[0,5,7], dtype=str)  #时间 最大值 最小值
                data['archour'] = data['archour'].apply(trans_date)
                data.to_csv(output_file, sep=',', index=False, header=False)

# def linear_insert(origin_dir, output_dir):
#     with open("","r") as fp:
#         data = fp.read()
#         lines = data.split('\n')
#         lines.insert(5,"new line")
#         data = '\n'.join(lines)
#         with open("","w") as fp:
#             fp.write(data)

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

    return loc_list

def generate_feature_by_hostname(origin_dir, out_file):
    # feature_matrix = []     #样本矩阵，每行为主机对应特征
    # label = []   #标记某一时间的样本对应是否有事件发生
    f_list = os.listdir(origin_dir)    #csv list
    host_name_file = {}
    ignored_disk_name = ['oracle','']

    for file_name in f_list:
        host_name = get_host_name(file_name)        #创建host_name 对应的dict 每个host包含多个文件
        host_name_file[host_name]=[]
    for file_name in f_list:
        host_name = get_host_name(file_name)
        host_name_file[host_name].append(file_name)

    host_name_list = host_name_file.keys()
    for h_name in host_name_list:            #遍历每个主机对应的文件list
        f_list = host_name_file[h_name]
        # out_file = os.path.join(output_dir,h_name+'.csv')
        df = pd.DataFrame(columns = ['hostname','archour'])
        #创建dataframe
        cnt = 1
        for f in f_list:
            with open(origin_dir + "/" + f, "r") as fp1:            #通过时间字段 对hostname的不同部件的max min值merge到同一个dataframe中
                data = pd.read_csv(fp1, dtype=str,index_col=False)
                data.columns = ['archour', 'maxvalue-'+str(cnt), 'minvalue-'+str(cnt)]
                df = pd.merge(left=df, right=data, on=['archour'],how="outer",left_index= False,right_index= False)
                cnt = cnt + 1
        df['hostname'] = h_name
        df.to_csv(out_file, sep=',', index=False,mode='a+')
        # df.to_csv(out_file, sep=',', index=False)#将dataframe追加写入到同一个文件中
        print(df)


def get_host_name(file_name):
    f_name = os.path.splitext(file_name)[0].split('_')
    ele = ["cpu", "disk", "mem"]
    host_name_list = []
    for e in ele:  # 判断是cpu、disk 还是mem文件  根据索引获取主机名
        if e in f_name:
            h = f_name.index(e)
    for a in range(0, h):
        host_name_list.append(f_name[a])
    host_name = '_'.join(host_name_list)
    return host_name

















