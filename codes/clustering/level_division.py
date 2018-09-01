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

    print(D1) # 累积距离矩阵
    print(D1[-1, -1]) # 序列距离


if __name__ == '__main__':
    a = [1,2,3]
    b = [1,2,3]
    
    DTW(a,b)