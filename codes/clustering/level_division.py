import random
from itertools import starmap
import math

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
        return [clust.alarm_category]
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

    # return biclusters,clusters

#数据降维
def PCA(data):
    pass


def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 计算该轴上的统计值（0为列，1为行）
    newData = dataMat - meanVal
    return newData, meanVal

# def pca(dataMat, percent=0.99):
#
#     newData, meanVal = zeroMean(dataMat)
#     covMat = np.cov(newData, rowvar=0)
#     eigVals, eigVects = np.linalg.eig(np.mat(covMat))
#     n = percentage2n(eigVals, percent)  # 要达到percent的方差百分比，需要前n个特征向量
#     print(str(n) + u"vectors")
#     eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
#     n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
#     n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
#     lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
#     reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
#     return reconMat, lowDDataMat

def percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num
