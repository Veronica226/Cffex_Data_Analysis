import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, gc
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from scipy import stats

kpi_list = ['cpu_max', 'cpu_min', 'boot_max', 'boot_min','home_max', 'home_min', 'monitor_max', 'monitor_min',
            'rt_max', 'rt_min', 'tmp_max', 'tmp_min','mem_max', 'mem_min']
threshhold = 0.05
max_tolerate_num = 2

#这里接收到的item为一个pandas timestamp对象



def get_week_hour(item):
    week_day_index = item.weekday()
    return week_day_index * 24 + item.hour

#这里接收到的item为groupby之后的dataframe对象
def get_week_hour_model(item):
    wkhour = item['weekhour'].unique()[0]  #获取每个group的weekhour
    result_list = []
    index_list = []
    for kpi in kpi_list:
        index_list.append(kpi + '_model')
        kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(item[kpi + '_N'].values.reshape(-1, 1))
        result_list.append(kde)
    series = pd.Series(result_list, index=index_list)
    return series

#这里接收到的item为df_test的每一行series对象
def predict_on_test_set(item, df_model=None):
    week_hour = item['weekhour']
    event = item['event']
    result_list = [week_hour, event]
    index_list = ['weekhour', 'event']
    num = 0
    has_anomaly = 0
    for kpi in kpi_list:
        detection_model = df_model.loc[week_hour, kpi + '_model']
        kpi_val = item[kpi + '_N']
        prob = np.exp(detection_model.score_samples([[kpi_val]])[0])
        if(prob < threshhold or 1.0 - prob < threshhold):
            num += 1
        result_list.append(prob)
        index_list.append(kpi + 'anomaly_prob')
    #if(2 * num > len(kpi_list)):
    if(num >= max_tolerate_num):
        has_anomaly = 1
    result_list.append(has_anomaly)
    index_list.append('anomaly')
    series = pd.Series(result_list, index=index_list)
    return series

#把当前次随机划分的该主机
def save_host_train_test_data(df_train, df_test, df_train_results, df_test_results, train_data_path, test_data_path, train_resuls_path, test_result_path):
    df_train.to_csv(train_data_path, sep=',', index=False)
    df_test.to_csv(test_data_path, sep=',', index=False)
    df_train_results.to_csv(train_resuls_path, sep=',', index=False)
    df_test_results.to_csv(test_result_path, sep=',', index=False)

def generate_host_time_series_decomposition_model(host_data_dir, host_data_file_name, is_save=True):
    host_data_file_path = os.path.join(host_data_dir, host_data_file_name).replace('\\', '/')
    host_name = '_'.join(os.path.splitext(host_data_file_name)[0].split('_')[:-1])
    df_host = pd.read_csv(host_data_file_path, sep=',', dtype=str)  #会自动将archour识别为pandas内置的timestamp对象,但有时候又不好使，所以还是强制改成str
    df_host['archour'] = df_host['archour'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df_host['event'] = df_host['event'].apply(np.int64)
    column_list = list(df_host.iloc[:,2:].columns)
    column_list.remove('event')
    df_host.loc[:, column_list] = df_host.loc[:, column_list].applymap(np.float64)
    df_host['weekhour'] = df_host['archour'].apply(get_week_hour)
    df_train, df_test = train_test_split(df_host, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_anomaly_model = df_train.groupby(['weekhour']).apply(get_week_hour_model)
    df_train_results = df_train.apply(predict_on_test_set, df_model=df_anomaly_model, axis=1)
    df_test_results = df_test.apply(predict_on_test_set, df_model=df_anomaly_model, axis=1)
    if(is_save):
        train_data_path = os.path.join(host_data_dir, host_name + '_train_data.csv')
        test_data_path = os.path.join(host_data_dir, host_name + '_test_data.csv')
        train_results_path = os.path.join(host_data_dir, host_name + '_train_results.csv')
        test_results_path = os.path.join(host_data_dir, host_name + '_test_results.csv')
        save_host_train_test_data(df_train, df_test, df_train_results, df_test_results, train_data_path, test_data_path, train_results_path, test_results_path)
    return


def generate_model_result_score(host_data_dir, test_result_file_name):
    test_result_file_path = os.path.join(host_data_dir, test_result_file_name).replace('\\', '/')
    host_name = '_'.join(os.path.splitext(test_result_file_name)[0].split('_')[:-2])
    df_test_result = pd.read_csv(test_result_file_path, sep=',', dtype=str)
    df_test_result[['weekhour', 'event', 'anomaly']] = df_test_result[['weekhour', 'event', 'anomaly']].apply(lambda x: np.int64(np.float64(x)))
    kpi_column_list = list(df_test_result.columns)[2:-1]
    df_test_result[kpi_column_list] = df_test_result[kpi_column_list].apply(np.float64)
    TP = ((df_test_result['event'] == 1).values & (df_test_result['anomaly'] == 1).values).sum()
    FP = ((df_test_result['event'] == 0).values & (df_test_result['anomaly'] == 1).values).sum()
    FN = ((df_test_result['event'] == 1).values & (df_test_result['anomaly'] == 0).values).sum()
    TN = ((df_test_result['event'] == 0).values & (df_test_result['anomaly'] == 0).values).sum()
    if(TP + FP == 0):
        precision = 0
    else:
        precision = 1.0 * TP / (TP + FP)
    if(TP + FN == 0):
        recall = 0
    else:
        recall = 1.0 * TP / (TP + FN)
    accuracy = 1.0 * (TP + TN) / (TP + TN + FP + FN)
    assert TP + TN + FP + FN == df_test_result.shape[0]
    if(precision == 0 and recall == 0):
        F_half_score = 0
        F_score = 0
    else:
        F_half_score = 1.25 * (precision * recall) / (0.25 * precision + recall)
        F_score = 2.0 * precision * recall / (precision + recall)
    return [host_name, precision, recall, accuracy, F_half_score, F_score]

#获取全部主机的异常检测模型及相关指标
def generate_time_series_decomposition_model(input_file_dir):
    host_index_file = os.path.join(input_file_dir, 'host_data_index.txt').replace('\\', '/')
    with open(host_index_file, 'r') as f:
        content = [item.strip() for item in f.readlines()]
        for host_data_file in content:
            host_name = '_'.join(os.path.splitext(host_data_file)[0].split('_')[:-1])
            host_data_dir = os.path.join(input_file_dir, host_name).replace('\\', '/')
            generate_host_time_series_decomposition_model(host_data_dir, host_data_file)
            print('process ' + host_name + ' model finished!')
    return

#计算模型的预测准确度等指标
def calc_model_evaluation_score(input_file_dir):
    host_index_file = os.path.join(input_file_dir, 'host_data_index.txt').replace('\\', '/')
    index_list = ['hostname', 'precision', 'recall', 'accuracy', 'F0.5-score', 'F1-score']
    df_train_score = pd.DataFrame(columns=index_list)
    df_test_score = pd.DataFrame(columns=index_list)
    idx = 0
    with open(host_index_file, 'r') as f:
        content = [item.strip() for item in f.readlines()]
        for host_data_file in content:
            host_name = '_'.join(os.path.splitext(host_data_file)[0].split('_')[:-1])
            host_data_dir = os.path.join(input_file_dir, host_name).replace('\\', '/')
            train_result_file = host_name + '_train_results.csv'
            test_result_file = host_name + '_test_results.csv'
            df_train_score.loc[idx] = pd.Series(generate_model_result_score(host_data_dir, train_result_file), index=index_list)
            df_test_score.loc[idx] = pd.Series(generate_model_result_score(host_data_dir, test_result_file), index=index_list)
            idx += 1
    train_score_file_path = os.path.join(input_file_dir, 'host_train_socre.csv')
    test_score_file_path = os.path.join(input_file_dir, 'host_test_socre.csv')
    df_train_score.to_csv(train_score_file_path, sep=',', index=False)
    df_test_score.to_csv(test_score_file_path, sep=',', index=False)
    print('output host score file finished!')


def calc_host_ave_model_evaluation_score(input_file_dir):
    train_score_file_path = os.path.join(input_file_dir, 'host_train_socre.csv')
    test_score_file_path = os.path.join(input_file_dir, 'host_test_socre.csv')
    df_train_score = pd.read_csv(train_score_file_path)
    df_test_score = pd.read_csv(test_score_file_path)
    evaluation_score_list = ['precision', 'recall', 'accuracy', 'F0.5-score', 'F1-score']
    train_ave_result_series = df_train_score[evaluation_score_list].mean()  #默认axis=0,即计算列的均值
    test_ave_result_series = df_test_score[evaluation_score_list].mean()
    df_train_ave_result = pd.DataFrame(columns=evaluation_score_list)
    df_test_ave_result = pd.DataFrame(columns=evaluation_score_list)
    df_train_ave_result.loc[0] = train_ave_result_series
    df_test_ave_result.loc[0] = test_ave_result_series

    train_ave_score_file_path = os.path.join(input_file_dir, 'host_train_ave_socre.csv')
    test_ave_score_file_path = os.path.join(input_file_dir, 'host_test_ave_socre.csv')
    df_train_ave_result.to_csv(train_ave_score_file_path, sep=',', index=False)
    df_test_ave_result.to_csv(test_ave_score_file_path, sep=',', index=False)

if __name__ == '__main__':
    time = datetime.now()
    time = time + timedelta(days=2)
    print(time.weekday())