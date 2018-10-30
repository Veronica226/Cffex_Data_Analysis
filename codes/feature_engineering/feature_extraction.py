import datetime
import os
import pandas as pd
from codes.preprocessing.data_preprocessing import common_disk_list

######################################################################################
#Author: 王靖文

def generate_feature_by_hostname(origin_dir, out_file):
    f_list = os.listdir(origin_dir)    #csv list
    host_name_file_dict = {}
    for file_name in f_list:
        host_name = get_host_name(file_name)
        #print (file_name)#创建host_name 对应的dict 每个host包含多个文件
        host_name_file_dict[host_name] = host_name_file_dict.get(host_name, [])
        host_name_file_dict[host_name].append(file_name)
    #这里得到的主机数量是201，与原有告警文件中主机总数261差了60台
    host_name_list = host_name_file_dict.keys()
    df_all = pd.DataFrame(columns=['hostname', 'archour',
                                   'cpu_avg','cpu_maxt','cpu_max','cpu_mint', 'cpu_min',     #创建空dataframe 存放merge之后的数据
                                   'boot_avg','boot_maxt','boot_max', 'boot_mint','boot_min',
                                  'home_avg','home_maxt', 'home_max','home_mint', 'home_min',
                                   'monitor_avg','monitor_maxt', 'monitor_max','monitor_mint', 'monitor_min',
                                   'rt_avg','rt_maxt','rt_max','rt_mint', 'rt_min',
                                   'tmp_avg','tmp_maxt', 'tmp_max', 'tmp_mint','tmp_min',
                                    'mem_avg','mem_maxt', 'mem_max','mem_mint', 'mem_min'])
    print('host number = ', len(host_name_list))
    for h_name in host_name_list:            #遍历每个主机对应的文件list
        file_list = host_name_file_dict[h_name]
        f_list = file_filter(file_list)   #筛选需要的文件
        print (h_name)
        df = pd.DataFrame(columns = ['hostname','archour'])
        idx = 0
        for f_name in f_list:    #遍历筛选之后的文件
            prefix = str(get_prefix(f_name))    #获取对应的部件作为前缀
            file_path = os.path.join(origin_dir, f_name)
            data = pd.read_csv(file_path, sep=',', dtype=str, header=None,index_col=None)  #header=None设置列名为空，自动用0开头的数字替代
            data.columns = ['archour',prefix+'_avg',prefix+ '_maxt',prefix+ '_max', prefix+'_mint', prefix+'_min' ]  #列名
            if(idx == 0):
                df = pd.merge(df, data, how='outer', on=['archour'])
                idx = 1
            else:
                df = pd.merge(df, data, on=['archour'])  #通过时间字段 对hostname的不同部件的max min值merge到同一个dataframe中
            # print (df)

        df['hostname'] = h_name
        df_all = pd.concat([df_all, df])   #连接当前主机的df数据到df_all中
    print('yes')

    df_all.to_csv(out_file, sep=',', index=False)
    print(df_all.shape)  #643560*16
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
    disk_file_list = common_disk_list.copy()
    disk_file_list.extend(['cpu','mem'])
    saved_file_list = []
    for f in f_list:
        flag = 0
        file_name = os.path.splitext(f)[0]
        for item in disk_file_list:
            if file_name.endswith(item):
                flag = 1
        if flag == 1 and f.find('cffex')==-1 :        #文件名符合筛选条件，防止找到cffex_home这样的文件
            saved_file_list.append(f)
    assert len(saved_file_list) == 7
    return saved_file_list

def generate_data_matrix_and_vector(feature_file,alarm_file,merged_data_file):
    feature_data_df = pd.read_csv(feature_file, sep=',',dtype=str) #643560*16
    # feature_data_df = pd.read_csv(feature_file, sep=',', engine='python', iterator=True)
    # loop = True
    # chunkSize = 1000
    # chunks = []
    # index = 0
    # while loop:
    #     try:
    #         print(index)
    #         chunk = feature_data_df.get_chunk(chunkSize)
    #         chunks.append(chunk)
    #         index += 1
    #
    #     except StopIteration:
    #         loop = False
    #         print("Iteration is stopped.")
    # print('开始合并')
    # feature_data_df = pd.concat(chunks, ignore_index=True)

    print(feature_data_df.shape)
    alarm_data_df = pd.read_csv(alarm_file, sep=',', dtype=str)  #13614*3
    print(alarm_data_df.shape)
    #通过主机名和时间 左连接将告警事件match到对应的特征数据中
    merged_df = pd.merge(feature_data_df, alarm_data_df, on=['hostname','archour'], how="left",left_index= False,right_index= False).fillna(0)
    #merge之后，有7997条告警数据，639199条非告警数据
    merged_df.to_csv(merged_data_file, sep=',', index=False)
    print(merged_df[merged_df['event']==0].shape)
    print(merged_df[merged_df['event'] == '1'].shape)

def generate_history_feature(origin_dir, history_data_file):
    f_list = os.listdir(origin_dir)  # csv list
    host_name_file_dict = {}
    for file_name in f_list:
        host_name = get_host_name(file_name)
        # print (file_name)#创建host_name 对应的dict 每个host包含多个文件
        host_name_file_dict[host_name] = host_name_file_dict.get(host_name, [])
        host_name_file_dict[host_name].append(file_name)
    # 这里得到的主机数量是201，与原有告警文件中主机总数261差了60台
    host_name_list = host_name_file_dict.keys()
    df_all = pd.DataFrame(columns=['hostname', 'archour',
                                   'cpu_avg', 'cpu_maxt', 'cpu_max', 'cpu_mint', 'cpu_min',  #创建空dataframe 存放merge之后的数据
                                   'boot_avg', 'boot_maxt', 'boot_max', 'boot_mint', 'boot_min',
                                   'home_avg', 'home_maxt', 'home_max', 'home_mint', 'home_min',
                                   'monitor_avg', 'monitor_maxt', 'monitor_max', 'monitor_mint', 'monitor_min',
                                   'rt_avg', 'rt_maxt', 'rt_max', 'rt_mint', 'rt_min',
                                   'tmp_avg', 'tmp_maxt', 'tmp_max', 'tmp_mint', 'tmp_min',
                                   'mem_avg', 'mem_maxt', 'mem_max', 'mem_mint', 'mem_min',
                                   'cpu_avg_1', 'cpu_maxt_1', 'cpu_max_1', 'cpu_mint_1', 'cpu_min_1',
                                   'boot_avg_1', 'boot_maxt_1', 'boot_max_1', 'boot_mint_1', 'boot_min_1',
                                   'home_avg_1', 'home_maxt_1', 'home_max_1', 'home_mint_1', 'home_min_1',
                                   'monitor_avg_1', 'monitor_maxt_1', 'monitor_max_1', 'monitor_mint_1', 'monitor_min_1',
                                   'rt_avg_1', 'rt_maxt_1', 'rt_max_1', 'rt_mint_1', 'rt_min_1',
                                   'tmp_avg_1', 'tmp_maxt_1', 'tmp_max_1', 'tmp_mint_1', 'tmp_min_1',
                                   'mem_avg_1', 'mem_maxt_1', 'mem_max_1', 'mem_mint_1', 'mem_min_1',
                                   'cpu_avg_2', 'cpu_maxt_2', 'cpu_max_2', 'cpu_mint_2', 'cpu_min_2',  # 创建空dataframe 存放merge之后的数据
                                   'boot_avg_2', 'boot_maxt_2', 'boot_max_2', 'boot_mint_2', 'boot_min_2',
                                   'home_avg_2', 'home_maxt_2', 'home_max_2', 'home_mint_2', 'home_min_2',
                                   'monitor_avg_2', 'monitor_maxt_2', 'monitor_max_2', 'monitor_mint_2', 'monitor_min_2',
                                   'rt_avg_2', 'rt_maxt_2', 'rt_max_2', 'rt_mint_2', 'rt_min_2',
                                   'tmp_avg_2', 'tmp_maxt_2', 'tmp_max_2', 'tmp_mint_2', 'tmp_min_2',
                                   'mem_avg_2', 'mem_maxt_2', 'mem_max_2', 'mem_mint_2', 'mem_min_2'
                                   ])
    # columns=['hostname', 'archour', 'cpu_max', 'cpu_min',  # 创建空dataframe 存放merge之后的数据
    #                                'boot_max', 'boot_min', 'home_max', 'home_min',
    #                                'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
    #                                'tmp_max', 'tmp_min', 'mem_max', 'mem_min',
    #                                'cpu_max_1', 'cpu_min_1', 'boot_max_1', 'boot_min_1',
    #                                'home_max_1', 'home_min_1', 'monitor_max_1', 'monitor_min_1',
    #                                'rt_max_1', 'rt_min_1', 'tmp_max_1', 'tmp_min_1', 'mem_max_1', 'mem_min_1',
    #                                'cpu_max_2', 'cpu_min_2', 'boot_max_2', 'boot_min_2',
    #                                'home_max_2', 'home_min_2', 'monitor_max_2', 'monitor_min_2',
    #                                'rt_max_2', 'rt_min_2', 'tmp_max_2', 'tmp_min_2', 'mem_max_2', 'mem_min_2'])
    print('host number = ', len(host_name_list))
    print(host_name_list)
    for h_name in host_name_list:            #遍历每个主机对应的文件list
        file_list = host_name_file_dict[h_name]
        f_list = file_filter(file_list)   #筛选需要的文件
        print (h_name)
        df = pd.DataFrame(columns = ['hostname','archour'])
        idx = 0
        for f_name in f_list:    #遍历筛选之后的文件
            prefix = str(get_prefix(f_name))    #获取对应的部件作为前缀
            file_path = os.path.join(origin_dir, f_name)
            data = pd.read_csv(file_path, sep=',', dtype=str, header=None,index_col=None)  #header=None设置列名为空，自动用0开头的数字替代
            data.columns = ['archour',prefix+'_avg',prefix+ '_maxt',prefix+ '_max', prefix+'_mint', prefix+'_min' ] #列名
            if(idx == 0):
                df = pd.merge(df, data, how='outer', on=['archour'])
                idx = 1
            else:
                df = pd.merge(df, data, on=['archour'])  #通过时间字段 对hostname的不同部件的max min值merge到同一个dataframe中
            # print (df)
        df['hostname'] = h_name
        #单个主机的特征矩阵，还未连接到df_all中
        df_1hour_before = df[[ 'cpu_avg', 'cpu_maxt', 'cpu_max', 'cpu_mint', 'cpu_min',  #创建空dataframe 存放merge之后的数据
                                   'boot_avg', 'boot_maxt', 'boot_max', 'boot_mint', 'boot_min',
                                   'home_avg', 'home_maxt', 'home_max', 'home_mint', 'home_min',
                                   'monitor_avg', 'monitor_maxt', 'monitor_max', 'monitor_mint', 'monitor_min',
                                   'rt_avg', 'rt_maxt', 'rt_max', 'rt_mint', 'rt_min',
                                   'tmp_avg', 'tmp_maxt', 'tmp_max', 'tmp_mint', 'tmp_min',
                                   'mem_avg', 'mem_maxt', 'mem_max', 'mem_mint', 'mem_min']][1:-1]     #前一个小时的特征

        df_2hour_before = df[['cpu_avg', 'cpu_maxt', 'cpu_max', 'cpu_mint', 'cpu_min',  #创建空dataframe 存放merge之后的数据
                                   'boot_avg', 'boot_maxt', 'boot_max', 'boot_mint', 'boot_min',
                                   'home_avg', 'home_maxt', 'home_max', 'home_mint', 'home_min',
                                   'monitor_avg', 'monitor_maxt', 'monitor_max', 'monitor_mint', 'monitor_min',
                                   'rt_avg', 'rt_maxt', 'rt_max', 'rt_mint', 'rt_min',
                                   'tmp_avg', 'tmp_maxt', 'tmp_max', 'tmp_mint', 'tmp_min',
                                   'mem_avg', 'mem_maxt', 'mem_max', 'mem_mint', 'mem_min']][0:-2]  # 前2个小时的特征

        df_1hour_before.columns = ['cpu_avg_1', 'cpu_maxt_1', 'cpu_max_1', 'cpu_mint_1', 'cpu_min_1',
                                   'boot_avg_1', 'boot_maxt_1', 'boot_max_1', 'boot_mint_1', 'boot_min_1',
                                   'home_avg_1', 'home_maxt_1', 'home_max_1', 'home_mint_1', 'home_min_1',
                                   'monitor_avg_1', 'monitor_maxt_1', 'monitor_max_1', 'monitor_mint_1', 'monitor_min_1',
                                   'rt_avg_1', 'rt_maxt_1', 'rt_max_1', 'rt_mint_1', 'rt_min_1',
                                   'tmp_avg_1', 'tmp_maxt_1', 'tmp_max_1', 'tmp_mint_1', 'tmp_min_1',
                                   'mem_avg_1', 'mem_maxt_1', 'mem_max_1', 'mem_mint_1', 'mem_min_1']
        df_2hour_before.columns = ['cpu_avg_2', 'cpu_maxt_2', 'cpu_max_2', 'cpu_mint_2', 'cpu_min_2',  # 创建空dataframe 存放merge之后的数据
                                   'boot_avg_2', 'boot_maxt_2', 'boot_max_2', 'boot_mint_2', 'boot_min_2',
                                   'home_avg_2', 'home_maxt_2', 'home_max_2', 'home_mint_2', 'home_min_2',
                                   'monitor_avg_2', 'monitor_maxt_2', 'monitor_max_2', 'monitor_mint_2', 'monitor_min_2',
                                   'rt_avg_2', 'rt_maxt_2', 'rt_max_2', 'rt_mint_2', 'rt_min_2',
                                   'tmp_avg_2', 'tmp_maxt_2', 'tmp_max_2', 'tmp_mint_2', 'tmp_min_2',
                                   'mem_avg_2', 'mem_maxt_2', 'mem_max_2', 'mem_mint_2', 'mem_min_2']


        df_1hour_before = df_1hour_before.reset_index(drop=True)    #重置索引列

        df = df.loc[2:] #从第三个时刻开始算起，去掉前两个时刻，从而构造时间窗口
        df = df.reset_index(drop=True)
        df = df.join(df_1hour_before,how = 'outer')
        df = df.join(df_2hour_before,how = 'outer')


        df_all = pd.concat([df_all, df])   #连接当前主机的df数据到df_all中
        print(df_all)

    print('yes')
    print(df_all)
    df_all.to_csv(history_data_file, sep=',', index=False)
    print(df_all.shape)  #643560*44
    print('done')

#去掉部分特征，只保留一种特征或两种特征
def delete_feature(origin_file, output_file):
    data = pd.read_csv(origin_file, sep=',', dtype = str)
    print(data)
    #要去掉的特征list
    drop_column_list =['cpu_maxt', 'cpu_mint',
                      'boot_avg','boot_maxt','boot_max',
                       'boot_mint','boot_min',
                       'home_avg','home_maxt', 'home_max', 'home_mint','home_min',
                      'monitor_avg','monitor_maxt','monitor_max', 'monitor_mint',  'monitor_min',
                       'rt_avg','rt_maxt','rt_max', 'rt_mint', 'rt_min',
                      'tmp_avg','tmp_maxt','tmp_max', 'tmp_mint', 'tmp_min',
                      'mem_maxt', 'mem_mint',
                       'cpu_maxt_1', 'cpu_mint_1',
                       'boot_avg_1', 'boot_maxt_1', 'boot_max_1',
                       'boot_mint_1', 'boot_min_1',
                       'home_avg_1', 'home_maxt_1', 'home_max_1', 'home_mint_1', 'home_min_1',
                       'monitor_avg_1', 'monitor_maxt_1', 'monitor_max_1', 'monitor_mint_1', 'monitor_min_1',
                       'rt_avg_1', 'rt_maxt_1', 'rt_max_1', 'rt_mint_1', 'rt_min_1',
                       'tmp_avg_1', 'tmp_maxt_1', 'tmp_max_1', 'tmp_mint_1', 'tmp_min_1',
                       'mem_maxt_1', 'mem_mint_1',
                       'cpu_maxt_2', 'cpu_mint_2',
                       'boot_avg_2', 'boot_maxt_2', 'boot_max_2',
                       'boot_mint_2', 'boot_min_2',
                       'home_avg_2', 'home_maxt_2', 'home_max_2', 'home_mint_2', 'home_min_2',
                       'monitor_avg_2', 'monitor_maxt_2', 'monitor_max_2', 'monitor_mint_2', 'monitor_min_2',
                       'rt_avg_2', 'rt_maxt_2', 'rt_max_2', 'rt_mint_2', 'rt_min_2',
                       'tmp_avg_2', 'tmp_maxt_2', 'tmp_max_2', 'tmp_mint_2', 'tmp_min_2',
                       'mem_maxt_2', 'mem_mint_2',
                      'boot_max_2', 'boot_min_2', 'home_max_2', 'home_min_2',
                      'monitor_max_2', 'monitor_min_2', 'rt_max_2', 'rt_min_2',
                      'tmp_max_2', 'tmp_min_2',
                       'cpu_dt', 'mem_dt', 'cpu_dt_1', 'mem_dt_1', 'cpu_dt_2', 'mem_dt_2', 'cpu_amm', 'mem_amm',
                       'cpu_amm_1', 'mem_amm_1','cpu_amm_2', 'mem_amm_2']
                      # 'mem_max_2', 'mem_min_2']
    data.drop(drop_column_list,axis=1,inplace=True)
    print(data)
    data.to_csv(output_file,sep=',',index=None)

#生成各个特征的六小时内数据
def generate_cluster_history_data(origin_dir, cluster_history_data_file):
    f_list = os.listdir(origin_dir)  # csv list
    host_name_file_dict = {}
    for file_name in f_list:
        host_name = get_host_name(file_name)
        # print (file_name)#创建host_name 对应的dict 每个host包含多个文件
        host_name_file_dict[host_name] = host_name_file_dict.get(host_name, [])
        host_name_file_dict[host_name].append(file_name)
    # 这里得到的主机数量是201，与原有告警文件中主机总数261差了60台
    host_name_list = host_name_file_dict.keys()
    # 创建空dataframe 存放merge之后的数据
    df_all = pd.DataFrame(columns=['hostname', 'archour', 'cpu_max', 'cpu_min',
                                   'boot_max', 'boot_min', 'home_max', 'home_min',
                                   'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                                   'tmp_max', 'tmp_min', 'mem_max', 'mem_min',
                                   'cpu_max_1', 'cpu_min_1', 'boot_max_1', 'boot_min_1',
                                   'home_max_1', 'home_min_1', 'monitor_max_1', 'monitor_min_1',
                                   'rt_max_1', 'rt_min_1', 'tmp_max_1', 'tmp_min_1', 'mem_max_1', 'mem_min_1',
                                   'cpu_max_2', 'cpu_min_2', 'boot_max_2', 'boot_min_2',
                                   'home_max_2', 'home_min_2', 'monitor_max_2', 'monitor_min_2',
                                   'rt_max_2', 'rt_min_2', 'tmp_max_2', 'tmp_min_2', 'mem_max_2', 'mem_min_2',
                                   'cpu_max_3', 'cpu_min_3', 'boot_max_3', 'boot_min_3',
                                   'home_max_3', 'home_min_3', 'monitor_max_3', 'monitor_min_3',
                                   'rt_max_3', 'rt_min_3', 'tmp_max_3', 'tmp_min_3', 'mem_max_3', 'mem_min_3',
                                   'cpu_max_4', 'cpu_min_4', 'boot_max_4', 'boot_min_4',
                                   'home_max_4', 'home_min_4', 'monitor_max_4', 'monitor_min_4',
                                   'rt_max_4', 'rt_min_4', 'tmp_max_4', 'tmp_min_4', 'mem_max_4', 'mem_min_4',
                                   'cpu_max_5', 'cpu_min_5', 'boot_max_5', 'boot_min_5',
                                   'home_max_5', 'home_min_5', 'monitor_max_5', 'monitor_min_5',
                                   'rt_max_5', 'rt_min_5', 'tmp_max_5', 'tmp_min_5', 'mem_max_5', 'mem_min_5'
                                   ])
    print('host number = ', len(host_name_list))
    print(host_name_list)
    for h_name in host_name_list:            #遍历每个主机对应的文件list
        file_list = host_name_file_dict[h_name]
        f_list = file_filter(file_list)   #筛选需要的文件
        print (h_name)
        df = pd.DataFrame(columns = ['hostname','archour'])
        idx = 0
        for f_name in f_list:    #遍历筛选之后的文件
            prefix = str(get_prefix(f_name))    #获取对应的部件作为前缀
            file_path = os.path.join(origin_dir, f_name)
            data = pd.read_csv(file_path, sep=',', dtype=str, header=None,index_col=None)  #header=None设置列名为空，自动用0开头的数字替代
            data.columns = ['archour',prefix+ '_max', prefix+'_min']  #列名
            if(idx == 0):
                df = pd.merge(df, data, how='outer', on=['archour'])
                idx = 1
            else:
                df = pd.merge(df, data, on=['archour'])  #通过时间字段 对hostname的不同部件的max min值merge到同一个dataframe中
            # print (df)
        df['hostname'] = h_name
        #单个主机的特征矩阵，还未连接到df_all中
        df_1hour_before = df[[ 'cpu_max', 'cpu_min',  # 创建空dataframe 存放merge之后的数据
                                   'boot_max', 'boot_min', 'home_max', 'home_min',
                                   'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                                   'tmp_max', 'tmp_min', 'mem_max', 'mem_min']][4:-1]     #前一个小时的特征
        df_2hour_before = df[[ 'cpu_max', 'cpu_min',
                                   'boot_max', 'boot_min', 'home_max', 'home_min',
                                   'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                                   'tmp_max', 'tmp_min', 'mem_max', 'mem_min']][3:-2]    #前两个小时的特征
        df_3hour_before = df[['cpu_max', 'cpu_min',
                              'boot_max', 'boot_min', 'home_max', 'home_min',
                              'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                              'tmp_max', 'tmp_min', 'mem_max', 'mem_min']][2:-3]  # 前一个小时的特征
        df_4hour_before = df[['cpu_max', 'cpu_min',
                              'boot_max', 'boot_min', 'home_max', 'home_min',
                              'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                              'tmp_max', 'tmp_min', 'mem_max', 'mem_min']][1:-4]  # 前一个小时的特征
        df_5hour_before = df[['cpu_max', 'cpu_min',
                              'boot_max', 'boot_min', 'home_max', 'home_min',
                              'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                              'tmp_max', 'tmp_min', 'mem_max', 'mem_min']][0:-5]  # 前两个小时的特征

        df_1hour_before.columns = ['cpu_max_1', 'cpu_min_1', 'boot_max_1', 'boot_min_1',
                                   'home_max_1', 'home_min_1','monitor_max_1', 'monitor_min_1',
                                   'rt_max_1', 'rt_min_1','tmp_max_1', 'tmp_min_1', 'mem_max_1', 'mem_min_1']
        df_2hour_before.columns = ['cpu_max_2', 'cpu_min_2', 'boot_max_2', 'boot_min_2',
                                   'home_max_2', 'home_min_2', 'monitor_max_2', 'monitor_min_2',
                                   'rt_max_2', 'rt_min_2', 'tmp_max_2', 'tmp_min_2', 'mem_max_2', 'mem_min_2']
        df_3hour_before.columns = ['cpu_max_3', 'cpu_min_3', 'boot_max_3', 'boot_min_3',
                                   'home_max_3', 'home_min_3', 'monitor_max_3', 'monitor_min_3',
                                   'rt_max_3', 'rt_min_3', 'tmp_max_3', 'tmp_min_3', 'mem_max_3', 'mem_min_3']
        df_4hour_before.columns = ['cpu_max_4', 'cpu_min_4', 'boot_max_4', 'boot_min_4',
                                   'home_max_4', 'home_min_4', 'monitor_max_4', 'monitor_min_4',
                                   'rt_max_4', 'rt_min_4', 'tmp_max_4', 'tmp_min_4', 'mem_max_4', 'mem_min_4']
        df_5hour_before.columns = ['cpu_max_5', 'cpu_min_5', 'boot_max_5', 'boot_min_5',
                                   'home_max_5', 'home_min_5', 'monitor_max_5', 'monitor_min_5',
                                   'rt_max_5', 'rt_min_5', 'tmp_max_5', 'tmp_min_5', 'mem_max_5', 'mem_min_5']
        df_1hour_before = df_1hour_before.reset_index(drop=True)    #重置索引列
        df_2hour_before = df_2hour_before.reset_index(drop=True)
        df_3hour_before = df_3hour_before.reset_index(drop=True)
        df_4hour_before = df_4hour_before.reset_index(drop=True)
        df_5hour_before = df_5hour_before.reset_index(drop=True)



        df = df.loc[5:] #从第三个时刻开始算起，去掉前两个时刻，从而构造时间窗口
        df = df.reset_index(drop=True)
        df = df.join(df_1hour_before,how = 'outer')
        df = df.join(df_2hour_before,how = 'outer')
        df = df.join(df_3hour_before, how='outer')
        df = df.join(df_4hour_before,how = 'outer')
        df = df.join(df_5hour_before, how='outer')

        df_all = pd.concat([df_all, df])   #连接当前主机的df数据到df_all中
        print(df_all)

    print('yes')
    print(df_all)
    df_all.to_csv(cluster_history_data_file, sep=',', index=False)
    print(df_all.shape)  #643560*44
    print('done')

#生成聚类所用的序列集合，每个集合元素为特征六小时的特征数据及告警等级和告警类型
def generate_cluster_data(cluster_history_data_file,alarm_file,cluster_merge_data_file):
    feature_data = pd.read_csv(cluster_history_data_file, sep=',', dtype=str)
    print(feature_data.shape)
    alarm_data= pd.read_csv(alarm_file, sep=',', dtype=str)
    print(alarm_data.shape)
    # 通过主机名和时间 左连接将告警事件match到对应的特征数据中,并包含level和content两列数据  共7046条
    merged_df = pd.merge(feature_data, alarm_data, on=['hostname', 'archour'], how="inner", left_index=False, right_index=False)
    print(merged_df)
    # merge之后，有7997条告警数据，639199条非告警数据
    merged_df.to_csv(cluster_merge_data_file, sep=',', index=False)
    # print(merged_df['content'].value)
    # print(merged_df[merged_df['event'] == '1'].shape)

def generate_history_label(merged_data_file):
    df = pd.read_csv(merged_data_file,sep=',', dtype=str)
    new_df = df[['hostname','archour','event']]

    new_df.columns = ['hostname','archour','pre_event']
    new_df.drop_duplicates(inplace=True)
    new_df['archour'] = new_df['archour'].apply(trans_date)
    merged_df = pd.merge(df, new_df, on=['hostname','archour'], how="inner", left_index=False,
                         right_index=False)
    print(merged_df)
    merged_df.to_csv(merged_data_file, sep=',', index=False)
    print(merged_df['pre_event'].value_counts())
    print(merged_df['event'].value_counts())
    print(df['event'].value_counts())

def trans_date(date_str):
    #2018-01-24 02:00:00
    hour = int(date_str[11:13])
    hour+=1
    if hour<=9 :
        str_hour = '0'+str(hour)
    else:
        str_hour = str(hour)
    new_str = date_str[:11] +str_hour+ date_str[13:]
    print(new_str)
    return new_str




