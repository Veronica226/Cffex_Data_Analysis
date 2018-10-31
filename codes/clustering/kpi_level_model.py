#coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from settings import cluster_data_dir
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import metrics
import os

def create_kNN_model(data_dir):
    column_list = ['cpu_max', 'cpu_min', 'mem_max', 'mem_min', 'cpu_max_1', 'cpu_min_1', 'mem_max_1', 'mem_min_1',
        'cpu_max_2', 'cpu_min_2', 'mem_max_2', 'mem_min_2', 'event', 'alertgroup']
    cluster_training_file_path = os.path.join(data_dir, 'cluster_series_data.csv')
    df = pd.read_csv(cluster_training_file_path, usecols=column_list)
    df['alertgroup'].fillna('Other', inplace=True)
    # print(df['event'].value_counts())
    df['event'] = df['event'].map({0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3})
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=50366)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    print(df_test['alertgroup'].unique())
    test_sample_num = df_test.shape[0]

    alert_groups = df_train.groupby('alertgroup')
    biz_data = alert_groups.get_group('Biz')
    print('Start create knn models!')
    biz_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto', leaf_size=30) #, metric='correlation') #biz knn model的event level是0,1,3,4
    biz_knn_model.fit(biz_data.drop(['alertgroup', 'event'], axis=1), biz_data['event'])
    print('biz event:\n', biz_data['event'].value_counts())
    biz_predict_results = biz_knn_model.predict(df_test.drop(['alertgroup', 'event'], axis=1))
    print('biz predict:\n', pd.Series(biz_predict_results).value_counts())

    mon_data = alert_groups.get_group('Mon')
    mon_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto', leaf_size=30) #, metric='correlation') #mon knn model的event level是0,1,3,4,5
    mon_knn_model.fit(mon_data.drop(['alertgroup','event'], axis=1), mon_data['event'])
    print('mon event:\n', mon_data['event'].value_counts())
    mon_predict_results = mon_knn_model.predict(df_test.drop(['alertgroup', 'event'], axis=1))
    print('mon predict:\n', pd.Series(mon_predict_results).value_counts())

    ora_data = alert_groups.get_group('Ora')
    ora_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto', leaf_size=30) #, metric='correlation')  #ora knn model的event level是0,1,3,4
    ora_knn_model.fit(ora_data.drop(['alertgroup','event'], axis=1), ora_data['event'])
    print('ora event:\n', ora_data['event'].value_counts())
    ora_predict_results = ora_knn_model.predict(df_test.drop(['alertgroup', 'event'], axis=1))
    print('ora predict:\n', pd.Series(ora_predict_results).value_counts())

    trd_data = alert_groups.get_group('Trd')
    trd_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto', leaf_size=30) #, metric='correlation')  #trd knn model的event level是0,2,4
    trd_knn_model.fit(trd_data.drop(['alertgroup','event'], axis=1), trd_data['event'])
    print('trd event:\n', trd_data['event'].value_counts())
    trd_predict_results = trd_knn_model.predict(df_test.drop(['alertgroup', 'event'], axis=1))
    print('trd predict:\n', pd.Series(trd_predict_results).value_counts())

    other_data = alert_groups.get_group('Other')
    other_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto', leaf_size=30)  # , metric='correlation')  #trd knn model的event level是0,2,4
    other_knn_model.fit(other_data.drop(['alertgroup', 'event'], axis=1), other_data['event'])
    print('other event:\n', other_data['event'].value_counts())
    other_predict_results = other_knn_model.predict(df_test.drop(['alertgroup', 'event'], axis=1))
    print('other predict:\n', pd.Series(other_predict_results).value_counts())

    print('Finish create knn models!')
    print(test_sample_num)
    predict_results = np.zeros(test_sample_num)
    df_res = pd.DataFrame(columns=['mode', 'max', 'group', 'predict', 'real'])

    for idx in range(test_sample_num):
        sample_predict_vec = np.array([np.round(biz_predict_results[idx]), np.round(mon_predict_results[idx]), np.round(ora_predict_results[idx]),
                                       np.round(trd_predict_results[idx]), np.round(other_predict_results[idx])])
        mode_prediction_res = stats.mode(sample_predict_vec)[0][0] #5个模型预测结果的众数
        max_prediction_res = sample_predict_vec[np.argmax(sample_predict_vec)] #5个模型预测结果的最大值
        alert_group_name = df_test.loc[idx, 'alertgroup']
        mapping_dict = {'Biz': 0, 'Mon': 1, 'Ora': 2, 'Trd': 3, 'Other': 4}
        group_prediction_res = sample_predict_vec[mapping_dict[alert_group_name]] #group_prediction_val <= max_prediction_val， 该条数据对应的业务模型预测的结果
        if(mode_prediction_res <= 2 and max_prediction_res <= 2):
            predict_results[idx] = group_prediction_res
        else:
            predict_results[idx] = max_prediction_res
        df_res.loc[idx] = pd.Series([mode_prediction_res, max_prediction_res, group_prediction_res, predict_results[idx], df_test.loc[idx, 'event']],
                                    index=['mode', 'max', 'group', 'predict', 'real'])
    #df_res.to_csv('predict_res.csv', sep=',', index=False)
    df_predict_res = pd.Series(predict_results)
    #df_predict_res.to_csv('predict_res_level.csv', sep=',', index=False)
    df_label_res = df_test['event'].copy()
    #df_label_res.to_csv('test_res_level.csv', sep=',', index=False)

    # print(df_res)
    print(df_predict_res.value_counts())

    print(df_label_res.value_counts())

    # generate_barplot(df_label_res,df_predict_res)
    # generate_sample_plot(df_res['real'][150:200],df_res['predict'][150:200])
    print((df_predict_res.values != df_label_res).sum())
    print((df_predict_res.values != df_label_res).sum()/test_sample_num)
    print(metrics.mean_absolute_error(df_predict_res, df_label_res))
    print(metrics.mean_squared_error(df_predict_res, df_label_res))
    return

#按业务划分最终的kNN model
def create_grouped_kNN_model(data_dir):
    return

def generate_barplot(label,result):
    label_list =(label.value_counts()).tolist()
    result_list = (result.value_counts()).tolist()
    name_list = ['level_1','level_2','level_3']
    x = list(range(len(label_list)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, label_list, width=width, label='real', fc='y')
    for i, j in zip(x, label_list):
        plt.text(i, j + 0.1, str(j), ha='center')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, result_list, width=width, label='predict', tick_label=name_list, fc='r')
    for i, j in zip(x, result_list):
        plt.text(i, j + 0.1, str(j), ha='center')
    plt.title('level predict result')
    plt.legend()
    plt.show()

def generate_sample_plot(label,result):
    axix = range(0,len(label),1)
    plt.title('level predict result')
    plt.plot(axix,label,color = 'red',label='real')
    plt.plot(axix, result, color='lightgreen', label='predict')
    plt.legend()
    plt.show()


def predict():
    feature_list = ['cpu_avg', 'cpu_max', 'cpu_min', 'mem_avg', 'mem_max', 'mem_min', 'cpu_avg_1', 'cpu_max_1', 'cpu_min_1', 'mem_avg_1', 'mem_max_1', 'mem_min_1',
        'cpu_avg_2', 'cpu_max_2', 'cpu_min_2', 'mem_avg_2', 'mem_max_2', 'mem_min_2']
    return

def test_KNN_model(data_dir):

    column_list = ['cpu_max', 'cpu_min', 'mem_max', 'mem_min', 'cpu_max_1', 'cpu_min_1', 'mem_max_1', 'mem_min_1',
                       'cpu_max_2', 'cpu_min_2', 'mem_max_2', 'mem_min_2', 'event', 'alertgroup']
    cluster_training_file_path = os.path.join(data_dir, 'cluster_series_data.csv')
    df = pd.read_csv(cluster_training_file_path, usecols=column_list)
    df['alertgroup'].fillna('Other', inplace=True)
    df['event'] = df['event'].map({0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3})
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=50366)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    print(df_test['alertgroup'].unique())
    test_sample_num = df_test.shape[0]

    alert_groups = df_train.groupby('alertgroup')
    biz_data = alert_groups.get_group('Biz')
    print('Start create knn models!')
    biz_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto',
                                            leaf_size=30)  # , metric='correlation') #biz knn model的event level是0,1,3,4
    biz_knn_model.fit(biz_data.drop(['alertgroup', 'event'], axis=1), biz_data['event'])


    mon_data = alert_groups.get_group('Mon')
    mon_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto',
                                            leaf_size=30)  # , metric='correlation') #mon knn model的event level是0,1,3,4,5
    mon_knn_model.fit(mon_data.drop(['alertgroup', 'event'], axis=1), mon_data['event'])


    ora_data = alert_groups.get_group('Ora')
    ora_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto',
                                            leaf_size=30)  # , metric='correlation')  #ora knn model的event level是0,1,3,4
    ora_knn_model.fit(ora_data.drop(['alertgroup', 'event'], axis=1), ora_data['event'])

    trd_data = alert_groups.get_group('Trd')
    trd_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto',
                                            leaf_size=30)  # , metric='correlation')  #trd knn model的event level是0,2,4
    trd_knn_model.fit(trd_data.drop(['alertgroup', 'event'], axis=1), trd_data['event'])


    other_data = alert_groups.get_group('Other')
    other_knn_model = KNeighborsRegressor(n_neighbors=3, algorithm='auto',
                                              leaf_size=30)  # , metric='correlation')  #trd knn model的event level是0,2,4
    other_knn_model.fit(other_data.drop(['alertgroup', 'event'], axis=1), other_data['event'])

    return biz_knn_model,mon_knn_model,ora_knn_model,trd_knn_model,other_knn_model


if __name__ == '__main__':
    create_kNN_model(cluster_data_dir)