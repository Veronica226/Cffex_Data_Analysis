import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import stats

from codes.timeseries_prediction import timeseries_prediction_model
from codes.preprocessing import data_preprocessing
from codes.model import predict_model
from codes.clustering import kpi_level_model

from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from datetime import datetime,timedelta
from settings import output_dir
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from settings import output_dir,alarm_data_dir,new_predict_data_dir,cluster_data_dir
# 预测调用函数样例
def run_test_predict(out_file_name):
	predict_result = pd.DataFrame()
	# 可选预测模型
	predict_model = [
		# 基线模型，用于对比，优于该模型则说明建模有效
		'baseline',
		# 滑动平均模型
		'MA',
		# 指数平滑模型
		'ES',
		# 指数平滑趋势调整模型
		'ES_Trend',
		# 指数加权移动平均模型
		'EWMA',
		# 小波变换分解+ARMA
		'Wavelet',
		# 随机森林回归
		'RFR',
		# 支持向量机回归
		'SVR',
		# K邻近回归
		'KNN',
		# 长短期记忆神经网络
		'LSTM'
	]

	# 时间序列预测部分
	csv_dir = os.path.join(output_dir,'cffex-host-info-cpu-mem')
	dateparse = lambda dates: pd.datetime.strptime(dates,'%Y%m%d%H')
	f_list = os.listdir(csv_dir)
	for i in range(len(f_list)):
		if i%2==1:
			continue
		else:
			# 文件名处理
			filename_split = f_list[i].split('.')
			host_name = filename_split[0][5:-11]
			filename_split = filename_split[0].split('_')
			filename_split.pop()
			filename = '_'
			filename = filename.join(filename_split)
			print(filename +' done')
			# cpu数据准备
			cpu_data = pd.read_csv(os.path.join(csv_dir,f_list[i]),encoding='UTF-8',parse_dates=['archour'],index_col='archour',date_parser=dateparse)
			# mem数据准备
			mem_data = pd.read_csv(os.path.join(csv_dir,f_list[i+1]),encoding='UTF-8',parse_dates=['archour'],index_col='archour',date_parser=dateparse)
			# 进行预测
			predict_result = predict_result.append(timeseries_prediction_model.predict(host_name,cpu_data, mem_data, result_length=50, model='SVR'))
	# timeseries_prediction_model.predict()
	# 参数列表：
	# 		必须参数：
	# 			cpu_data：主机cpu数据
	# 			mem_data：主机mem数据
	# 			result_length：预期返回的数据期数，默认30
	# 			model：预期使用的预测模型，默认基线模型
	# 		可选参数：
	# 			MA_window：移动平均模型移动窗口大小，默认12
	# 			ES_factor：指数平滑模型平滑系数，默认0.7
	# 			ES_Trand_factor：指数平滑趋势调整系数，默认0.5
	# 			EWMA_factor：指数加权移动平均系数，默认0.6
	# 			RFR_tree_num：随机森林树数量，默认20
	# 			LSTM_term_num：长短期记忆神经网络时间点数量，默认1500
	# 			LSTM_neuron_num：长短期记忆神经网络神经元数量，默认5
	# 样例调用：
	# predict_result = timeseries_prediction_model.predict('2018_vcsdb1_hourly',cpu_data, mem_data, result_length=50, model='SVR')
	predict_result.to_csv(out_file_name, index = 0, encoding = 'UTF-8')

# 分类调用函数样例
def test_classifier_model(out_file_name,predict_result_file):
	traing_data_file = os.path.join(new_predict_data_dir, "training_data.csv")
	alertgroup_file = os.path.join(alarm_data_dir, 'alertgroup.csv')
	data_preprocessing.get_alertgroup_by_hostname(alertgroup_file, out_file_name)
	data = pd.read_csv(out_file_name, sep=',', dtype=str)
	all_df = pd.DataFrame(columns=['alertgroup','classifier', 'hostname', 'predict_event'])

	for alertgroup,group in data.groupby('alertgroup'):
		if alertgroup!='Net':
			print(str(alertgroup)+' start')
			col_list = [
				'cpu_avg',
				'cpu_max',
				'cpu_min',
				'mem_avg',
				'mem_max',
				'mem_min',
				'cpu_avg_1',
				'cpu_max_1',
				'cpu_min_1',
				'mem_avg_1',
				'mem_max_1',
				'mem_min_1',
				'cpu_avg_2',
				'cpu_max_2',
				'cpu_min_2',
				'mem_avg_2',
				'mem_max_2',
				'mem_min_2']
			test_classifiers_list = ['RF',
									 'GBDT',
									 'KNN',
									 'DT']
			data = group[col_list]
			# data.replace(-np.inf, np.nan)
			# data.fillna(0)
			data = data.convert_objects(convert_numeric=True)
			for classifier in test_classifiers_list:
				print(str(classifier)+' start')
				model = predict_model.test_classifier_for_prediction(traing_data_file,alertgroup,classifier)
				predict = model.predict(data)

				new_df = pd.DataFrame(columns=['alertgroup', 'classifier', 'hostname', 'predict_event'])
				new_df['hostname'] = group['hostname']
				new_df['predict_event'] = predict
				new_df['classifier'] = classifier
				new_df['alertgroup'] = alertgroup
				new_df = new_df.join(data,how='outer')
				print(new_df['predict_event'].value_counts())
				all_df = pd.concat([all_df, new_df])

		all_df.to_csv(predict_result_file,sep=',',index=False)

#告警级别判定调用函数样例
def test_kpi_level_model(predict_result_file,final_result_file):
	df = pd.read_csv(predict_result_file, sep=',',dtype=str)
	df = df[df['predict_event']=='1']
	mapping_dict = {'Biz': 0, 'Mon': 1, 'Ora': 2, 'Trd': 3, 'Other': 4}
	knn_model_list = []
	knn_model_list = kpi_level_model.test_KNN_model(cluster_data_dir)
	all_df = pd.DataFrame(columns=['alertgroup', 'classifier', 'hostname', 'predict_event','predict_level'])
	for alertgroup,group in df.groupby('alertgroup'):
		column_list = ['cpu_max', 'cpu_min', 'mem_max', 'mem_min', 'cpu_max_1', 'cpu_min_1', 'mem_max_1', 'mem_min_1',
				   'cpu_max_2', 'cpu_min_2', 'mem_max_2', 'mem_min_2']
		data = group[column_list]
		kpi_predict_result = []
		for i in knn_model_list:
			kpi_predict_result.append(i.predict(data))
		print(kpi_predict_result)
		predict_results = np.zeros(len(group))
		df_res = pd.DataFrame(columns=['predict_level'])
		for idx in range(len(group)):
			sample_predict_vec = np.array([np.round(kpi_predict_result[0][idx]), np.round(kpi_predict_result[1][idx]),
										   np.round(kpi_predict_result[2][idx]),
										   np.round(kpi_predict_result[3][idx]), np.round(kpi_predict_result[4][idx])])
			# print(sample_predict_vec)
			mode_prediction_res = stats.mode(sample_predict_vec)[0][0]  # 5个模型预测结果的众数
			print(mode_prediction_res)
			max_prediction_res = sample_predict_vec[np.argmax(sample_predict_vec)]  # 5个模型预测结果的最大值
			print(max_prediction_res)
			group_prediction_res = sample_predict_vec[mapping_dict[alertgroup]]  # group_prediction_val <= max_prediction_val， 该条数据对应的业务模型预测的结果
			print(group_prediction_res)
			if (mode_prediction_res <= 2 and max_prediction_res <= 2):
				predict_results[idx] = group_prediction_res
			else:
				predict_results[idx] = max_prediction_res
			df_res.loc[idx] = int(predict_results[idx])

		new_df = group[['alertgroup', 'classifier', 'hostname', 'predict_event']].reset_index(drop=True).join(df_res,how = 'outer')
		all_df = pd.concat([all_df, new_df])

	print(all_df)
	all_df.to_csv(final_result_file, sep=',', index=False)

def run_tests():
	csv_dir = r'E:\HomeMadeSoftware\Cffex_Data_Analysis\output_data\cffex-host-info-cpu-mem'
	dateparse = lambda dates: pd.datetime.strptime(dates,'%Y%m%d%H')
	f_list = os.listdir(csv_dir)

	predict_result = pd.DataFrame(columns=['file','feature','baseline_TS_RMSE',
		'MA_TS_RMSE','ES_TS_RMSE','EST_TS_RMSE','EWMA_TS_RMSE',
		# 'wavelet_ARMA_TS_RMSE',
		'RFR_TS_RMSE','SVR_TS_RMSE','KNN_TS_RMSE'])
	feature = ['maxvalue','minvalue','avgvalue']
	for i in f_list:
		data = pd.read_csv(os.path.join(csv_dir,i),encoding='UTF-8',parse_dates=['archour'],index_col='archour',date_parser=dateparse)
		for j in feature:
			timeseries = data[j]
			result_length = 50

			baseline_TS = timeseries_prediction_model.baseline_model(timeseries, result_length)
			MA_TS = timeseries_prediction_model.moving_average_model(timeseries, 12, result_length)
			ES_TS = timeseries_prediction_model.exponential_smoothing_model(timeseries, 0.7, result_length)
			EST_TS = timeseries_prediction_model.exponential_smoothing_trend_adjustment_model(timeseries, 0.7, 0.5, result_length)
			EWMA_TS = timeseries_prediction_model.exponential_weight_moving_average_model(timeseries, 0.6, result_length)
			RFR_TS = timeseries_prediction_model.random_forest_regressor_model(timeseries, result_length)
			SVR_TS = timeseries_prediction_model.surpport_vector_regressor_model(timeseries, result_length)
			KNN_TS = timeseries_prediction_model.k_neighbors_regressor_model(timeseries, result_length)
			
			compare_TS = timeseries_prediction_model.add_predict_term_to_timeseries(timeseries, timeseries.tail(1).iloc[0])
			compare_TS = compare_TS[-result_length:]
			baseline_TS_RMSE = sqrt(mean_squared_error(compare_TS,baseline_TS))
			MA_TS_RMSE = sqrt(mean_squared_error(compare_TS,MA_TS))
			ES_TS_RMSE = sqrt(mean_squared_error(compare_TS,ES_TS))
			EST_TS_RMSE = sqrt(mean_squared_error(compare_TS,EST_TS))
			EWMA_TS_RMSE = sqrt(mean_squared_error(compare_TS,EWMA_TS))
			RFR_TS_RMSE = sqrt(mean_squared_error(compare_TS,RFR_TS))
			SVR_TS_RMSE = sqrt(mean_squared_error(compare_TS,SVR_TS))
			KNN_TS_RMSE = sqrt(mean_squared_error(compare_TS,KNN_TS))

			try:
				wavelet_ARMA_TS = timeseries_prediction_model.wavelet_ARMA_model(timeseries, result_length)
			except:
				wavelet_ARMA_TS_RMSE = 0
			else:
				wavelet_ARMA_TS_RMSE = sqrt(mean_squared_error(compare_TS,wavelet_ARMA_TS))
			

			print(i)
			print(j)
			print('baseline_TS_RMSE:%f' % baseline_TS_RMSE)
			print('MA_TS_RMSE:%f' % MA_TS_RMSE)
			print('ES_TS_RMSE:%f' % ES_TS_RMSE)
			print('EST_TS_RMSE:%f' % EST_TS_RMSE)
			print('EWMA_TS_RMSE:%f' % EWMA_TS_RMSE)
			print('wavelet_ARMA_TS_RMSE:%f' % wavelet_ARMA_TS_RMSE)
			print('RFR_TS_RMSE:%f' % RFR_TS_RMSE)
			print('SVR_TS_RMSE:%f' % SVR_TS_RMSE)
			print('KNN_TS_RMSE:%f' % KNN_TS_RMSE)

			predict_result = predict_result.append({'file':os.path.splitext(i)[0],
                                            'feature':j,
                                            'baseline_TS_RMSE':baseline_TS_RMSE,
                                            'MA_TS_RMSE':MA_TS_RMSE,
                                            'ES_TS_RMSE':ES_TS_RMSE,
                                            'EST_TS_RMSE':EST_TS_RMSE,
                                            'EWMA_TS_RMSE':EWMA_TS_RMSE,
                                            'wavelet_ARMA_TS_RMSE':wavelet_ARMA_TS_RMSE,
                                            'RFR_TS_RMSE':RFR_TS_RMSE,
                                            'SVR_TS_RMSE':SVR_TS_RMSE,
                                            'KNN_TS_RMSE':KNN_TS_RMSE,
                                           },ignore_index = True)
		predict_result.to_csv('output_data\\TS_predict_result_RMSE.csv', index = 0, encoding = 'UTF-8')

if __name__ == '__main__':
    # run_tests()

   model_save_file = os.path.join(output_dir,'classifier_model.csv')
   out_file_name = os.path.join(output_dir,'TS_predict_result.csv')
   predict_result_file = os.path.join(output_dir,'alarm_prediction.csv')
   final_result_file = os.path.join(output_dir,'final_prediction_result.csv')
   run_test_predict(out_file_name)   #预测时间序列
   test_classifier_model(out_file_name,predict_result_file)  #分类器预测
   test_kpi_level_model(predict_result_file,final_result_file) #告警级别划分