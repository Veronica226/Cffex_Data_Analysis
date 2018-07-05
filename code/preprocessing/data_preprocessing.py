#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import DataFrame
import pandas as pd
import os, sys, json, csv, re

common_disk_list = ['boot', 'rt', 'home', 'monitor', 'tmp']  #通过generate_plot_data得到所有主机公共的磁盘目录

def trans_date(date_str):
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8] + ' ' + date_str[8:] + ':00:00'

def data_preprocessing_process(origin_dir, output_dir):
    f_list = os.listdir(origin_dir)
    for i in f_list:  ##每个log文件
        if os.path.splitext(i)[1] == '.log':
            file_name = os.path.join(output_dir, os.path.splitext(i)[0] + '.csv')
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


def generate_plot_data(origin_dir, output_dir):
    f_list = os.listdir(origin_dir)
    for file in f_list:
        file_path = os.path.join(origin_dir, file)
        file_name = os.path.splitext(file)[0]
        file_name_list = file_name.split('_')
        host_name = file_name_list[1] #主机名是第1个元素（从0开始）
        device_name = file_name_list[-1]  #设备名称是第-1个元素（list从末尾往前数）

        if file_name.endswith("disk"):  # 磁盘文件中diskname字段有不同的磁盘名
            data = pd.read_csv(file_path, usecols=[0, 3, 6, 8], dtype=str)
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
            data = pd.read_csv(file_path,usecols=[0,5,7], dtype=str)  #时间 最大值 最小值
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
        if find_missing_files(origin_dir, file_name) == 1:
        # if f_name.endswith("cpu") or f_name.endswith("mem"):
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
                if date != '' and date[11]is'2' and date[12]is'2':
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


def generate_feature_by_hostname(origin_dir, out_file):
    # feature_matrix = []     #样本矩阵，每行为主机对应特征
    # label = []   #标记某一时间的样本对应是否有事件发生
    f_list = os.listdir(origin_dir)    #csv list
    host_name_file_dict = {}
    ignored_disk_name = ['oracle','']

    for file_name in f_list:
        host_name = get_host_name(file_name)        #创建host_name 对应的dict 每个host包含多个文件
        host_name_file_dict[host_name] = host_name_file_dict.get(host_name, [])
        host_name_file_dict[host_name].append(file_name)

    host_name_list = host_name_file_dict.keys()
    # df_all = pd.DataFrame(columns = ['hostname','archour', 'cpu_max','cpu_min',
    #                                  'disk1_max','disk1_min' ,'disk2_max','disk2_min','disk3_max','disk3_min',
    #                                  'disk4_max','disk4_min','disk5_max','disk5_min','disk6_max','disk6_min','mem_max','mem_min'])
    df_all = pd.DataFrame(columns=['hostname', 'archour', 'cpu_max', 'cpu_min',
                                   'n_max', 'n_min', 'boot_max', 'boot_min',
                                   'home_max', 'home_min', 'monitor_max', 'monitor_min', 'tmp_max', 'tmp_min',
                                   'mem_max', 'mem_min'])
    #磁盘名字对应部件  disk1——.    disk2——boot  disk3——cffex  disk4——home  disk5——monitor   disk6——tmp

    for h_name in host_name_list:            #遍历每个主机对应的文件list
        file_list = host_name_file_dict[h_name]
        # out_file = os.path.join(output_dir,h_name+'.csv')
        f_list = file_filter(file_list)   #筛选需要的文件
        print (f_list)
        df = pd.DataFrame(columns = ['hostname','archour'])
        for f in f_list:
            prefix = str(get_prefix(f))
            with open(origin_dir + "/" + f, "r") as fp1:            #通过时间字段 对hostname的不同部件的max min值merge到同一个dataframe中
                data = pd.read_csv(fp1, sep=',', dtype=str, header=None,index_col=None)  #header=None设置列名为空，自动用0开头的数字替代
                data.columns = ['archour',prefix+ '_max', prefix+'_min']
                df = pd.merge(df, data, on=['archour'], how="outer",left_index= False,right_index= False)
                print (df)


        df['hostname'] = h_name
        df_all = pd.concat([df_all, df])
        print('yes')

    df_all.to_csv(out_file, sep=',', index=False)
    # df.to_csv(out_file, sep=',', index=False)#将dataframe追加写入到同一个文件中
    print(df)
    print('done')


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

def get_prefix(file):
    f_name = os.path.splitext(file)[0]         #根据后缀筛选需要的文件名
    contain_list = ['_cpu','_.csv','boot','_home','_monitor','_tmp','_mem']
    prefix_list = ['cpu', 'n', 'boot', 'home', 'monitor', 'tmp', 'mem']
    # postfix_list = ['cpu', '_', '_boot', '_cffex', '_home', '_monitor', '_tmp', '_mem']
    for item in contain_list:
        if item in f_name:
            index = contain_list.index(item)
            return prefix_list[index]            #返回前缀


def file_filter(f_list):
    save_str = ['cpu','_.csv','home','monitor.csv','tmp','mem']
    for f in f_list:
        cffex_list=[]
        flag = 0
        if 'boot' in f and f.endswith('boot.csv') == False:
            f_list.remove(f)

    #     elif 'cffex' in f:
    #           cffex_list.append(f)
    # # f_list = f_list - cffex_list[1:]
    # f_list = [i for i in f_list if i not in cffex_list[1:]]
    # for f in f_list:
        for item in save_str:
            if item in f:
                flag = 1
        if 'boot' in f==False  and flag == 0:
            f_list.remove(f)
    return f_list