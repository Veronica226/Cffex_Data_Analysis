from libc.string cimport memset
from sklearn.metrics.pairwise import manhattan_distances

cdef DTW(s1,s2):
    cdef int r = len(s1), c = len(s2)
    cdef double D[r + 1][c + 1]
    memset(D, 0, sizeof(D))

    # 使用曼哈顿距离 生成原始距离矩阵
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            D[i, j] = manhattan_distances(s1[i - 1], s2[j - 1])

    # 动态计算最短距离
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            D[i, j] += min(D[i, j], D[i, j - 1], D[i - 1, j])
    distance = D[r,c]
    #print(D1) # 累积距离矩阵
    #print(distance) # 序列距离
    return distance

#data表示矩阵，n表示矩阵的行，m表示矩阵的列
cdef calc_dtw_dist(double data[][], int n, int m):
    cdef double dis_mat[n][n]
    for i in range(n):
        for j in range(i + 1, n):
            dis_mat[i, j] = dis_mat[j, i] = DTW(data[i], data[j])
    return dis_mat

def calc_dtw_distance(data, n, m):
    return calc_dtw_dist(data, n, m)