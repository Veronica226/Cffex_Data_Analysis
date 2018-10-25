import random
from itertools import starmap
import math,os
import scipy
from scipy import stats
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
from sklearn.cluster import Birch,KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

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
        # return [clust.level]
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

def Spearman(s1,s2):
    return scipy.stats.spearmanr(s1,s2)[0]

def spearman_corr(x,y):
	assert len(x) == len(y)
	n = len(x)
	assert n > 0
	xrank = scipy.stats.rankdata(x)
	yrank = scipy.stats.rankdata(y)
	avgx = scipy.average(xrank)
	avgy =  scipy.average(yrank)
	diffmult= 0
	xdiff2 = 0
	ydiff2 = 0
	for i in range(n):
		 xdiff = xrank[i] - avgx
		 ydiff =  yrank[i] - avgy
		 diffmult += xdiff * ydiff
		 xdiff2 += xdiff * xdiff
		 ydiff2 += ydiff * ydiff
	return diffmult / math.sqrt(xdiff2 * ydiff2)

def DOT(s1,s2):
    return np.dot(s1,s2)


def DTW(s1, s2):
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


    distance = D1[-1, -1]
    # print(D1)  # 累积距离矩阵
    # print(distance)  # 序列距离
    # print(D1) # 累积距离矩阵
    # print(distance) # 序列距离
    return distance

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


def hierarchical_clusterting(cluster_series_data_file, n):
    series_df = get_cluster_data(cluster_series_data_file)
    for category, series in series_df.groupby('category'):
        print(str(category) + ':'+str(len(series)))
        data = series.values
        kpi_series = []
        for d in data:
            floats = d.tolist()
            float_list = [float(i) for i in floats]
            kpi_series.append(float_list)
         # kpi_series = random.sample(series,30)
        biclusters = [ bicluster(vec = kpi_series[i][:-2], id = i,level = kpi_series[i][-2],alarm_category= kpi_series[i][-1] )
                    for i in range(len(kpi_series)) ]   #存储序列的list
         # levels=[]  #存储聚类后的序列的level 作为验证
        distances = {}     #存储各个序列间的距离
        flag = None  #记录最相似的两个序列编号id
        currentclusted = -1
        while(len(biclusters) > n) : #假设聚成n个类
            min_val = float('inf') #Python的无穷大应该是inf
            biclusters_len = len(biclusters)
            for i in range(biclusters_len-1) :
                for j in range(i + 1, biclusters_len) :
                    # print('('+str(i)+','+str(j)+')')
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

            print(clusters)
    # return biclusters,clusters

#层次聚类
def hierarchical_clusterting_1(cluster_series_data_file,n) :
    series_df = get_cluster_data(cluster_series_data_file)
    for category, df_category in series_df.groupby('category'):
        print(category + ':')
        data = (df_category.iloc[:, :-3].values).astype(np.float64)
        print(data)
        sample_num = data.shape[0]
        dis_mat = zeros((sample_num, sample_num))
        for i in range(sample_num):
            for j in range(i + 1, sample_num):
                dis_mat[i, j] = dis_mat[j, i] = Pearson(data[i], data[j])
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



###########################################################################
#correlation analysis

#生成每个主机的关联告警相关性直方图
def generate_hist_plot(cor_list,hostname,hist_plot_dir):
    if not os.path.exists(hist_plot_dir):
        os.makedirs(hist_plot_dir)

    fig = plt.figure()
    plt.title(hostname)
    plt.hist(cor_list)
    plt.xlabel('correlation of kpi lists')
    plt.ylabel('numbers')
    plt.show()

    # plot_path = os.path.join(hist_plot_dir, hostname + '_spearman_hist_plot.png')
    # fig.savefig(plot_path, dpi=100)

def get_correlation_by_hostname(merged_data_file,hist_plot_dir,alarm_content_file,multiclass_data_dir):
    col_list = ['hostname','archour','cpu_max', 'cpu_min', 'mem_max', 'mem_min','event','content','alertgroup']
    data = pd.read_csv(merged_data_file, sep=',',usecols=col_list, dtype=str)
    df_content = pd.read_csv(alarm_content_file, sep=',', encoding="gb2312")
    content_dict = dict(zip(df_content['id'], df_content['alarm_content']))
    for alertgroup,all_df in data.groupby('alertgroup'):
        if alertgroup != 'Net':
            print(alertgroup)
            correlation_dict = {}
            all_alarm_content_list = []    #存储每个主机的告警list
            for hostname,df in all_df.groupby('hostname'):
                cor_list = []
                len_df = df.shape[0]
                # 每个主机内的所有告警作为dataset的一项
                # for i in range(0,len_df):
                #     row = df.iloc[i]
                #     # print(row.values.tolist()[2:-1])
                #     if(row['content']!='0'):
                #         content_id_1 = int(row['content'])
                #         # alarm_content_list.append(content_id_1)
                #         event_1 = int(row['event'])
                #         list_1 = row.values.tolist()[2:-2]
                #         kpi_list_1 = [float(i) for i in list_1]
                #         # print(kpi_list_1)
                #         if(i+1< len_df):
                #             row_next = df.iloc[i+1,:]
                #             if(row_next['content']!='0'):
                #                 content_id_2 = int(row_next['content'])
                #                 event_2 = int(row_next['event'])
                #                 # print('Yes')
                #                 if correlation_dict.get((content_id_1,event_1,content_id_2,event_2)) == None:
                #                     correlation_dict[(content_id_1,event_1,content_id_2,event_2)] = 1
                #                 else:
                #                     correlation_dict[(content_id_1,event_1,content_id_2,event_2)] += 1
                #                 list_2 = row_next.values.tolist()[2:-2]
                #                 kpi_list_2 = [float(i) for i in list_2]
                #                 cor = spearman_corr(kpi_list_1,kpi_list_2)
                #                 cor_list.append(cor)
                # if(cor_list != []):
                #     print(hostname+str(cor_list))
                #     generate_hist_plot(cor_list,hostname,hist_plot_dir)

            # 对每个主机的告警划分时间段，使每一项告警list内前后两个告警的时间间隔在1小时
            # cor analysis
                alarm_contents= df['content'].values.tolist()
                alarm_content = [float(i) for i in alarm_contents]
                i=0
                while(i < len_df-1):
                    k=1
                    if alarm_content[i]!=0:
                        sub_alarm_content_list = []
                        sub_alarm_content_list.append(alarm_content[i])
                        while(i+k < len_df - 1 and alarm_content[i+k]!=0):
                            sub_alarm_content_list.append(alarm_content[i+k])
                            # j+=1
                            k+=1
                        if len(sub_alarm_content_list) > 1:
                            all_alarm_content_list.append(sub_alarm_content_list)
                        # print(sub_alarm_content_list)
                    i+=k

            print(all_alarm_content_list)
            cor_analysis_1(all_alarm_content_list,content_dict)
            # cor_analysis end




        # print(correlation_dict)
        # df = generate_content_event_pair(correlation_dict,alarm_content_file)
        # print(df)
        # new_content_pair_file = os.path.join(multiclass_data_dir,alertgroup + '_correlation_pair.csv')
        # df.to_csv(new_content_pair_file,sep=',',encoding='gb2312')

#导出关联告警的pair
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




####################################################################
#########################第一种

def create_C1(data_set):
    """
    Create frequent candidate 1-itemset C1 by scaning data set.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
    Returns:
        C1: A set which contains all frequent candidate 1-itemsets
    """
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1


def is_apriori(Ck_item, Lksub1):
    """
    Judge whether a frequent candidate k-itemset satisfy Apriori property.
    Args:
        Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                 candidate k-itemsets.
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
    Returns:
        True: satisfying Apriori property.
        False: Not satisfying Apriori property.
    """
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    """
    Create Ck, a set which contains all all frequent candidate k-itemsets
    by Lk-1's own connection operation.
    Args:
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
        k: the item number of a frequent itemset.
    Return:
        Ck: a set which contains all all frequent candidate k-itemsets.
    """
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    """
    Generate Lk by executing a delete policy from Ck.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        Ck: A set which contains all all frequent candidate k-itemsets.
        min_support: The minimum support.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    Returns:
        Lk: A set which contains all all frequent k-itemsets.
    """
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk


def generate_L(data_set, k, min_support):
    """
    Generate all frequent itemsets.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        k: Maximum number of items for all frequent itemsets.
        min_support: The minimum support.
    Returns:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    """
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data


def generate_big_rules(L, support_data, min_conf):
    """
    Generate big rules from frequent itemsets.
    Args:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
        min_conf: Minimal confidence.
    Returns:
        big_rule_list: A list which contains all big rules. Each big rule is represented
                       as a 3-tuple.
    """
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list


def cor_analysis_1(data_set,content_dict):
    L, support_data = generate_L(data_set, k=4, min_support=0.1)
    big_rules_list = generate_big_rules(L, support_data, min_conf=0.2)
    for Lk in L:
        if len(Lk) != 0:
            print(Lk)
            print("=" * 50)
            print("frequent " + str(len(list(Lk)[0])) + "-itemsets\t\tsupport")
            print("=" * 50)
            for freq_set in Lk:
                print(freq_set, support_data[freq_set])

    print("Big Rules")
    for item in big_rules_list:
        print(item[0], "=>", item[1], "conf: ", item[2])
        new_item_0 = [content_dict[i]for i in item[0]]
        new_item_1 = [content_dict[j] for j in item[1]]
        print(new_item_0, "=>", new_item_1, "conf: ", item[2])



###########################################################################
##################第二种（有先后顺序）#####################################
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))

#将候选集Ck转换为频繁项集Lk
#D：原始数据集
#Cn: 候选集项Ck
#minSupport:支持度的最小值
def scanD(D, Ck, minSupport):
    #候选集计数
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys(): ssCnt[can] = 1
                else: ssCnt[can] += 1

    numItems = float(len(D))
    Lk= []     # 候选集项Cn生成的频繁项集Lk
    supportData = {}    #候选集项Cn的支持度字典
    #计算候选项集的支持度, supportData key:候选项， value:支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            Lk.append(key)
        supportData[key] = support
    return Lk, supportData

#连接操作，将频繁Lk-1项集通过拼接转换为候选k项集
def aprioriGen(Lk_1, k):
    Ck = []
    lenLk = len(Lk_1)
    for i in range(lenLk):
        L1 = list(Lk_1[i])[:k - 2]
        L1.sort()
        for j in range(i + 1, lenLk):
            #前k-2个项相同时，将两个集合合并
            L2 = list(Lk_1[j])[:k - 2]
            L2.sort()
            if L1 == L2:
                Ck.append(Lk_1[i] | Lk_1[j])

    return Ck

def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Lk_1 = L[k-2]
        Ck = aprioriGen(Lk_1, k)
        print("ck:",Ck)
        Lk, supK = scanD(dataSet, Ck, minSupport)
        supportData.update(supK)
        print("lk:", Lk)
        L.append(Lk)
        k += 1

    return L, supportData

#生成关联规则
#L: 频繁项集列表
#supportData: 包含频繁项集支持数据的字典
#minConf 最小置信度
def generateRules(L, supportData, minConf):
    #包含置信度的规则列表
    bigRuleList = []
    #从频繁二项集开始遍历
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 计算是否满足最小可信度
def calcConf(freqSet, H, supportData, brl, minConf):
    prunedH = []
    #用每个conseq作为后件
    for conseq in H:
        # 计算置信度
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            # 元组中的三个元素：前件、后件、置信度
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)

    #返回后件列表
    return prunedH


# 对规则进行评估
def rulesFromConseq(freqSet, H, supportData, brl, minConf):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
       # print(1,H, Hmp1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 0):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def cor_analysis_2(dataset):
    L, support_data = apriori(dataset, minSupport=0.01)
    for Lk in L:
        if len(Lk) != 0:
            print(Lk)
            print("=" * 50)
            print("frequent " + str(len(list(Lk)[0])) + "-itemsets\t\tsupport")
            print("=" * 50)
            for freq_set in Lk:
                print(freq_set, support_data[freq_set])
    print('=====================bigrules======================')

    big_rules_list = generateRules(L, support_data, minConf=0.2)
    # print(big_rules_list)
    # for item in big_rules_list:
    #     print(item[0], "=>", item[1], "conf: ", item[2])


def get_cluster_plot(cluster_data_file,cluster_plot_dir):
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
                   # 'category',
                   # 'event',
                   'content',
                   'alertgroup']
    df = pd.read_csv(cluster_data_file, usecols=column_list, sep=',', dtype=str)
    for alertgroup, new_df in df.groupby('alertgroup'):
        dataSet = np.array(new_df)
        new_dataSet = []
        labels = []
        for i in dataSet:
            new_dataSet.append([float(j) for j in i[:-2]])
            labels.append(float(i[-2]))
        dataSet = np.array(new_dataSet)

        y_pred = KMeans(n_clusters=5).fit_predict(dataSet)
        print(y_pred)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=80)
        norm1 = matplotlib.colors.Normalize(vmin=0, vmax=4)
        X_pca = TSNE(learning_rate=500, n_components=2, random_state=0).fit_transform(dataSet)
        fig = plt.figure()
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=2, c=y_pred, cmap='cool')
        plt.colorbar(sc)
        plt.title(alertgroup + '_KMeans_result_plot_cluser')
        plt.xlim((-120, 100))
        plt.ylim((-100, 100))
        plt.show()
        plot_path = os.path.join(cluster_plot_dir, 'KMeans_cluter.png')
        fig.savefig(plot_path, dpi=100)

        content_list = new_df['content'].tolist()
        X = pd.DataFrame(X_pca)
        L = pd.DataFrame(y_pred)
        X['cluster'] = y_pred
        X['content'] = content_list
        for cluster, c_df in X.groupby('cluster'):
            X_pca_1 = np.array(c_df)
            y = c_df['content'].tolist()
            y1 = c_df['cluster'].tolist()
            fig = plt.figure()
            sc = plt.scatter(X_pca_1[:, 0], X_pca_1[:, 1], s=2, c=y, cmap='Set1', norm=norm)
            plt.colorbar(sc)
            plt.title(alertgroup + '_KMeans_result_plot_cluser:' + str(cluster))
            plt.xlim((-120, 100))
            plt.ylim((-100, 100))
            plt.show()
            plot_path = os.path.join(cluster_plot_dir, str(cluster) + '_cluter_content.png')
            fig.savefig(plot_path, dpi=100)

            fig = plt.figure()
            sc = plt.scatter(X_pca_1[:, 0], X_pca_1[:, 1], s=2, c=y1, cmap='cool', norm=norm1)
            plt.colorbar(sc)
            plt.title(alertgroup + '_KMeans_result_plot_cluser:' + str(cluster))
            plt.xlim((-120, 100))
            plt.ylim((-100, 100))
            plt.show()
            plot_path = os.path.join(cluster_plot_dir, str(cluster) + '_cluter.png')
            fig.savefig(plot_path, dpi=100)

        # y_pred = Birch(n_clusters = None).fit_predict(X)
        # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        # plt.show()
        #
        # print("CH指标:", metrics.calinski_harabaz_score(X, y_pred))


