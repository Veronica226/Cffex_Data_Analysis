import random
from itertools import starmap
import math,os

import pandas as pd
import numpy as np
from numpy import array, zeros, argmin, inf, equal, ndim
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy
import sys
#

#使用DTW方法判断两个序列之间的距离，再作为后续聚类时的相似性度量指标
# def DTW_algorithm(a,b):
#
#     # M = [[distance(a[i],b[i]) for i in range(len(a))]  for j in range(len(b))]
#     # print(M)
#     l1 = len(a)
#     l2 = len(b)
#     D = [[0 for i in range(l1 + 1)] for i in range(l2 + 1)]
#     # D[0][0] = 0
#     for i in range(1, l1 + 1):
#         D[0][i] = sys.maxsize
#     for j in range(1, l2 + 1):
#         D[j][0] = sys.maxsize
#     for j in range(1, l2 + 1):
#         for i in range(1, l1 + 1):
#             D[j][i] = distance(a[i - 1], b[j - 1]) + min(D[j - 1][i], D[j][i - 1], D[j - 1][i - 1])
#             print(j, i, D[j][i])
#     print(D)
class bicluster:
    def __init__(self, vec, left=None, right=None,distance = 0.0,id=None,alarm_category=None,level= None):
        self.left = left
        self.right = right  # 每次聚类都是一对数据，left保存其中一个数据，right保存另一个
        self.vec = vec  # 保存两个数据聚类后形成新的中心
        self.distance = distance
        self.id = id    #保存序列编号
        self.alarm_category = alarm_category  #保存告警的内容编号
        self.level = level #保存告警事件的级别

def child(clust):
    if clust.left == None and clust.right == None :
        # print(clust.level)
        # return [clust.alarm_category]
        return [clust.level]
    return child(clust.left) + child(clust.right)



def Pearson(s1,s2):
    s1_mean = np.mean(s1)
    s2_mean = np.mean(s2)
    n = len(s1)
    sumTop = 0.0
    sumBottom = 0.0
    s1_pow = 0.0
    s2_pow = 0.0
    for i in range(n):
        sumTop += (s1[i]-s1_mean)*(s2[i]-s2_mean)
    for i in range(n):
        s1_pow += math.pow(s1[i]-s1_mean,2)
        s2_pow += math.pow(s2[i]-s2_mean,2)
    sumBottom = math.sqrt(s1_pow*s2_pow)
    p = sumTop/sumBottom
    return p

def DOT(s1,s2):
    return np.dot(s1,s2)

#生成聚类所用的数据，根据所选特征选择
def get_cluster_data(cluster_series_data_file):
    cluster_series_list = []
    column_list = ['cpu_max', 'cpu_min',
                   # 'boot_max', 'boot_min', 'home_max', 'home_min','monitor_max', 'monitor_min', 'rt_max', 'rt_min', 'tmp_max', 'tmp_min',
                   'mem_max', 'mem_min',
                   'cpu_max_1', 'cpu_min_1',
                   # 'boot_max_1', 'boot_min_1', 'home_max_1', 'home_min_1', 'monitor_max_1', 'monitor_min_1', 'rt_max_1', 'rt_min_1', 'tmp_max_1', 'tmp_min_1',
                   'mem_max_1', 'mem_min_1',
                   'cpu_max_2', 'cpu_min_2',
                   # 'boot_max_2', 'boot_min_2','home_max_2', 'home_min_2', 'monitor_max_2', 'monitor_min_2','rt_max_2', 'rt_min_2', 'tmp_max_2', 'tmp_min_2',
                   'mem_max_2', 'mem_min_2',
                    'cpu_max_3', 'cpu_min_3',
                   # 'boot_max_3', 'boot_min_3','home_max_3', 'home_min_3', 'monitor_max_3', 'monitor_min_3','rt_max_3', 'rt_min_3', 'tmp_max_3', 'tmp_min_3',
                   'mem_max_3', 'mem_min_3',
                   'cpu_max_4', 'cpu_min_4',
                   # 'boot_max_4', 'boot_min_4','home_max_4', 'home_min_4', 'monitor_max_4', 'monitor_min_4','rt_max_4', 'rt_min_4', 'tmp_max_4', 'tmp_min_4',
                   'mem_max_4', 'mem_min_4',
                    'cpu_max_5', 'cpu_min_5',
                   # 'boot_max_5', 'boot_min_5','home_max_5', 'home_min_5', 'monitor_max_5', 'monitor_min_5', 'rt_max_5', 'rt_min_5', 'tmp_max_5', 'tmp_min_5',
                   'mem_max_5', 'mem_min_5',
                  'category', 'event','content']
    df = pd.read_csv(cluster_series_data_file,usecols=column_list, sep=',', dtype=str)
    #
    # data = df.values
    # for d in data:
    #     floats = d.tolist()
    #     float_list = [float(i) for i in floats]
    #     cluster_series_list.append(float_list)
    # return cluster_series_list
    #print(df)
    return df
#层次聚类
def hierarchical_clusterting(cluster_series_data_file,n) :
    series_df = get_cluster_data(cluster_series_data_file)
    for category, df_category in series_df.groupby('category'):
        print(category + ':')
        data = (df_category.iloc[:, :-3].values).astype(np.float64)
        print(data)
        sample_num = data.shape[0]
        dis_mat = zeros((sample_num, sample_num))
        for i in range(sample_num):
            for j in range(i + 1, sample_num):
                dis_mat[i, j] = dis_mat[j, i] = DTW(data[i], data[j])
        print('YES')
        db = DBSCAN(eps=0.13, metric='precomputed', min_samples=3).fit(dis_mat)
        labels = db.labels_
        print(labels)
        print(df_category['content'].values)
        # data = series.values
        # kpi_series = []
        # for d in data:
        #     floats = d.tolist()
        #     float_list = [float(i) for i in floats]
        #     kpi_series.append(float_list)
        #
        # # kpi_series = random.sample(kpi_series,30)
        # biclusters = [ bicluster(vec = kpi_series[i][:-2], id = i,level = kpi_series[i][-2],alarm_category= kpi_series[i][-1] )
        #             for i in range(len(kpi_series)) ]   #存储序列的list
        #
        # # levels=[]  #存储聚类后的序列的level 作为验证
        # distances = {}     #存储各个序列间的距离
        # flag = None  #记录最相似的两个序列编号id
        # currentclusted = -1
        #
        # while(len(biclusters) > n) : #假设聚成n个类
        #     min_val = float('inf') #Python的无穷大应该是inf
        #     biclusters_len = len(biclusters)
        #     for i in range(biclusters_len-1) :
        #         for j in range(i + 1, biclusters_len) :
        #             print('('+str(i)+','+str(j)+')')
        #             if distances.get((biclusters[i].id,biclusters[j].id)) == None:
        #                 distances[(biclusters[i].id,biclusters[j].id)] = float(DTW(biclusters[i].vec,biclusters[j].vec))    #各个序列间的距离字典
        #             d = distances[(biclusters[i].id,biclusters[j].id)]
        #             if d < min_val :
        #                 min_val = d
        #                 flag = (i,j)     #更新最邻近的两个序列间距离，以及序列的编号

    # return biclusters,clusters
        break
#数据降维
def PCA(data):
    pass

def generate_hist_plot(cor_list,hostname,hist_plot_dir):
    if not os.path.exists(hist_plot_dir):
        os.makedirs(hist_plot_dir)

    fig = plt.figure()
    plt.title(hostname)
    plt.hist(cor_list)
    plt.xlabel('correlation of kpi lists')
    plt.ylabel('numbers')
    plt.show()

    plot_path = os.path.join(hist_plot_dir, hostname + '_hist_plot.png')
    fig.savefig(plot_path, dpi=100)


def get_correlation_by_hostname(merged_data_file,hist_plot_dir,alarm_content_file,content_pair_file):
    col_list = ['hostname','archour','cpu_max', 'cpu_min', 'mem_max', 'mem_min','event','content']
    data = pd.read_csv(merged_data_file, sep=',',usecols=col_list, dtype=str)
    correlation_dict = {}
    for hostname,df in data.groupby('hostname'):
        print(hostname)
        cor_list = []
        len = df.shape[0]
        for i in range(0,len):
            row = df.iloc[i]
            # print(row.values.tolist()[2:-1])
            if(row['content']!='0'):
                content_id_1 = int(row['content'])
                event_1 = int(row['event'])
                list_1 = row.values.tolist()[2:-2]
                kpi_list_1 = [float(i) for i in list_1]
                # print(kpi_list_1)
                if(i+1< len):
                    row_next = df.iloc[i+1,:]
                    if(row_next['content']!='0'):
                        content_id_2 = int(row_next['content'])
                        event_2 = int(row_next['event'])
                        print('Yes')
                        if correlation_dict.get((content_id_1,event_1,content_id_2,event_2)) == None:
                            correlation_dict[(content_id_1,event_1,content_id_2,event_2)] = 1
                        else:
                            correlation_dict[(content_id_1,event_1,content_id_2,event_2)] += 1
                        list_2 = row_next.values.tolist()[2:-2]
                        kpi_list_2 = [float(i) for i in list_2]
                        cor = Pearson(kpi_list_1,kpi_list_2)
                        cor_list.append(cor)

        # if(cor_list != []):
        #     print(cor_list)
        #     generate_hist_plot(cor_list,hostname,hist_plot_dir)

    print(correlation_dict)
    df = generate_content_event_pair(correlation_dict,alarm_content_file)
    df.to_csv(content_pair_file,sep=',',encoding='gb2312')

def generate_content_event_pair(correlation_dict,alarm_content_file):
    df_content = pd.read_csv(alarm_content_file, sep=',', encoding = "gb2312")
    content_dict = dict(zip(df_content['id'], df_content['alarm_content']))
    key_list = list(correlation_dict.keys())
    value_list = list(correlation_dict.values())
    content_id1_list = []
    alarm_content_1_list = []
    event_1_list = []
    content_id2_list = []
    alarm_content_2_list = []
    event_2_list = []
    for i in key_list:
        key_list_split = str(i).split(', ')

        content_id1 = int(key_list_split[0][1:])
        content_id1_list.append(content_id1 )
        alarm_content_1 = content_dict[content_id1]
        alarm_content_1_list.append(alarm_content_1)

        event_1_list.append(int( key_list_split[1]))

        content_id2 =int(key_list_split[2])
        content_id2_list.append(content_id2)
        alarm_content_2 = content_dict[content_id2]
        alarm_content_2_list.append(alarm_content_2)

        event_2_list.append(int(key_list_split[3][0]))
        # break

    data = {'i':content_id1_list,
            'content1':alarm_content_1_list,
            'event1':event_1_list,
            'j':content_id2_list,
            'content2':alarm_content_2_list,
            'event2':event_2_list,
            'count':value_list}
    df = pd.DataFrame(data)
    print(df)
    return df





