from itertools import starmap
from math import *
from numpy import array, zeros, argmin, inf, equal, ndim
from sklearn.metrics.pairwise import manhattan_distances
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
        print(clust.id)
        return [clust.id]
    return child(clust.left) + child(clust.right)

def DTW(s1,s2):
    r, c = len(s1), len(s2)
    D0 = zeros((r + 1, c + 1))
    D1 = D0[1:, 1:]

    # 使用曼哈顿距离 生成原始距离矩阵
    for i in range(r):
        for j in range(c):
            D1[i, j] = manhattan_distances(s1[i], s2[j])

    # 动态计算最短距离
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
    distance = D1[-1,-1]
    print(D1) # 累积距离矩阵
    print(distance) # 序列距离
    return distance

#层次聚类


def hierarchical_clusterting(kpi_series,n) :
    biclusters = [ bicluster(vec = kpi_series[i][:-1], id = i,level = kpi_series[i][-1] ) for i in range(len(kpi_series)) ]   #存储序列的list
    levels=[]  #存储聚类后的序列的level 作为验证
    distances = {}     #存储各个序列间的距离
    flag = None  #记录最相似的两个序列编号id
    currentclusted = -1

    while(len(biclusters) > n) : #假设聚成n个类
        min_val = float('inf') #Python的无穷大应该是inf
        biclusters_len = len(biclusters)
        for i in range(biclusters_len-1) :
            for j in range(i + 1, biclusters_len) :
                if distances.get((biclusters[i].id,biclusters[j].id)) == None:
                    distances[(biclusters[i].id,biclusters[j].id)] = float(DTW(biclusters[i].vec,biclusters[j].vec))    #各个序列间的距离字典
                d = distances[(biclusters[i].id,biclusters[j].id)]
                if d < min_val :
                    min_val = d
                    flag = (i,j)     #更新最邻近的两个序列间距离，以及序列的编号

        bic1,bic2 = flag
        newvec = [(biclusters[bic1].vec[i] + biclusters[bic2].vec[i])/2 for i in range(len(biclusters[bic1].vec))] #形成新的类中心，平均
        newbic = bicluster(newvec, left=biclusters[bic1], right=biclusters[bic2], distance=min_val, id = currentclusted) #二合一
        currentclusted -= 1
        del biclusters[bic2] #删除聚成一起的两个数据，由于这两个数据要聚成一起
        del biclusters[bic1]
        biclusters.append(newbic)#补回新聚类中心
        clusters = [child(biclusters[i]) for i in range(len(biclusters))] #深度优先搜索叶子节点，用于输出显示
        # levels = [yezi(biclusters[i]).level for i in range(len(biclusters))]
    return biclusters,clusters



if __name__ == '__main__':
    a = [1,2,3]
    b = [1,2,3]

    # DTW(a,b)
    C = [[1,2,3,0],[2,3,4,1],[1,2,3,0]]
    biclusters,clusters= hierarchical_clusterting(C,2)
    print(biclusters)
    print(clusters)
    # print (levels)