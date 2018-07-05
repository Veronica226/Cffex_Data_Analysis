import os
import pandas as pd
from code.preprocessing.data_preprocessing import common_disk_list

######################################################################################
#Author: 王靖文

def generate_feature_by_hostname(origin_dir, out_file):
    f_list = os.listdir(origin_dir)    #csv list
    host_name_file_dict = {}
    for file_name in f_list:
        host_name = get_host_name(file_name)        #创建host_name 对应的dict 每个host包含多个文件
        host_name_file_dict[host_name] = host_name_file_dict.get(host_name, [])
        host_name_file_dict[host_name].append(file_name)

    host_name_list = host_name_file_dict.keys()
    df_all = pd.DataFrame(columns=['hostname', 'archour', 'cpu_max', 'cpu_min',       #创建空dataframe 存放merge之后的数据
                                    'boot_max', 'boot_min','home_max', 'home_min',
                                   'monitor_max', 'monitor_min','rt_max', 'rt_min',
                                    'tmp_max', 'tmp_min','mem_max', 'mem_min'])

    for h_name in host_name_list:            #遍历每个主机对应的文件list
        file_list = host_name_file_dict[h_name]
        f_list = file_filter(file_list)   #筛选需要的文件
        print (f_list)
        df = pd.DataFrame(columns = ['hostname','archour'])
        for f in f_list:    #遍历筛选之后的文件
            prefix = str(get_prefix(f))    #获取对应的部件作为前缀
            with open(origin_dir + "/" + f, "r") as fp1:
                data = pd.read_csv(fp1, sep=',', dtype=str, header=None,index_col=None)  #header=None设置列名为空，自动用0开头的数字替代
                data.columns = ['archour',prefix+ '_max', prefix+'_min']  #列名
                df = pd.merge(df, data, on=['archour'], how="outer",left_index= False,right_index= False)  #通过时间字段 对hostname的不同部件的max min值merge到同一个dataframe中
                # print (df)

        df['hostname'] = h_name
        df_all = pd.concat([df_all, df])   #连接当前主机的df数据到df_all中
        print('yes')

    df_all.to_csv(out_file, sep=',', index=False)
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
    f_name_list = os.path.splitext(file)[0].split('_')   #根据后缀筛选需要的文件名
    return f_name_list[-1]            #返回前缀

def file_filter(f_list):
    common_disk_list.extend(['cpu','mem'])
    saved_file_list = []
    for f in f_list:
        flag = 0
        file_name = os.path.splitext(f)[0]
        for item in common_disk_list:
            if file_name.endswith(item):
                flag = 1
        if flag == 1 and f.find('cffex')==-1 :        #文件名符合筛选条件
            saved_file_list.append(f)
    return saved_file_list

def generate_data_matrix_and_vector(feature_file,alarm_file,merged_data_file):
    feature_data_df = pd.read_csv(feature_file, sep=',',dtype=str)
    alarm_data_df = pd.read_csv(alarm_file, sep=',', dtype=str)
    #通过主机名和时间 左连接将告警事件match到对应的特征数据中
    merged_df = pd.merge(feature_data_df, alarm_data_df, on=['hostname','archour'], how="left",left_index= False,right_index= False ).fillna(0)
    merged_df.to_csv(merged_data_file, sep=',', index=False)
    # print (merged_df['event'].value_counts())
    print(merged_df)