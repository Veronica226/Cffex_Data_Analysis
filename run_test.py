import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from codes.timeseries_prediction import timeseries_prediction_model

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

from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors

# 单独调用样例
from settings import output_dir


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

# 预测调用函数样例
def run_test_predict():
	dateparse = lambda dates: pd.datetime.strptime(dates,'%Y%m%d%H')
	cpu_data = pd.read_csv(os.path.join(output_dir,'cffex-host-info','2018_vcsdb1_hourly_cpu.csv'),encoding='UTF-8',parse_dates=['archour'],index_col='archour',date_parser=dateparse)
	mem_data = pd.read_csv(os.path.join(output_dir,'cffex-host-info','2018_vcsdb1_hourly_mem.csv'),encoding='UTF-8',parse_dates=['archour'],index_col='archour',date_parser=dateparse)
	
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
	predict_result = timeseries_prediction_model.predict(cpu_data, mem_data, result_length=50, model='SVR')
	
	predict_result.to_csv('output_data\\TS_predict_result.csv', index = 0, encoding = 'UTF-8')


if __name__ == '__main__':
    # run_tests()
    run_test_predict()