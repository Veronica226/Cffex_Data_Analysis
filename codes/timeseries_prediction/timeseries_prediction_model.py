import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from datetime import datetime,timedelta

import pywt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA

from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors

def predict(timeseries, result_length, model):
	if model == 'baseline':
		1
	elif model == 'MA':
		2
	elif model == 'ES':
		3
	elif model == 'ES_Trend':
		4
	elif model == 'EWMA':
		5
	elif model == 'Wavelet':
		6
	elif model == 'RFR':
		7
	elif model == 'SVR':
		8
	elif model == 'KNN':
		9
	elif model == 'LSTM':
		10
	

# 基线模型
def baseline_model(timeseries, result_length):
	baseline_TS = timeseries.copy(deep=True)
	baseline_TS[baseline_TS!=0] = 0.0
	baseline_TS[0] = timeseries[0]
	baseline_TS = add_predict_term_to_timeseries(baseline_TS, 0.0)
	for i in range(1, len(baseline_TS)):
		baseline_TS[i] = timeseries[i-1]
	return baseline_TS[-result_length:]

# 滑动平均模型
def moving_average_model(timeseries, required_window, result_length):
	timeseries = add_predict_term_to_timeseries(timeseries, 0.0)
	MA_TS = timeseries.rolling(window = required_window).mean()
	return MA_TS[-result_length:]

# 指数平滑模型
def exponential_smoothing_model(timeseries, smoothing_factor, result_length):
	ES_TS = timeseries.copy(deep=True)
	ES_TS[ES_TS!=0] = 0.0
	ES_TS[0] = timeseries[0]
	ES_TS = add_predict_term_to_timeseries(ES_TS, 0.0)
	for i in range(1,len(ES_TS)):
		ES_TS[i] = smoothing_factor*timeseries[i-1]+(1-smoothing_factor)*ES_TS[i-1]
	return ES_TS[-result_length:]

# 指数平滑趋势调整模型
def exponential_smoothing_trend_adjustment_model(timeseries, smoothing_factor, trend_factor, result_length):
    ES_TS = timeseries.copy(deep=True)
    ES_TS[ES_TS!=0] = 0.0
    ES_TS[0] = timeseries[0]
    ES_TS = add_predict_term_to_timeseries(ES_TS, 0.0)
    ES_Trend = ES_TS.copy(deep=True)
    ES_Final = ES_TS.copy(deep=True)
    for i in range(1, len(ES_TS)):
        ES_TS[i] = smoothing_factor*timeseries[i-1] + (1-smoothing_factor)*ES_TS[i-1]
        ES_Trend[i] = (1-trend_factor)*ES_Trend[i-1] + trend_factor*(ES_TS[i]-ES_TS[i-1])
        ES_Final[i] = ES_TS[i] + ES_Trend[i]
    return ES_Final[-result_length:]

# 指数加权移动平均模型
def exponential_weight_moving_average_model(timeseries, weight_factor, result_length):
	EWMA_TS = timeseries.copy(deep=True)
	EWMA_TS[EWMA_TS!=0] = 0.0
	EWMA_TS[0] = timeseries[0]
	EWMA_TS = add_predict_term_to_timeseries(EWMA_TS, 0.0)
	for i in range(1, len(EWMA_TS)):
		weight = weight_factor/(i+weight_factor-1)
		EWMA_TS[i] = weight*timeseries[i-1]+(1-weight)*EWMA_TS[i-1]
	return EWMA_TS[-result_length:]

# 小波变换分解+ARMA模型
def wavelet_ARMA_model(timeseries, result_length):
	timeseries = add_predict_term_to_timeseries(timeseries, 0.0)

	index_list = np.array(timeseries)[:-result_length]
	date_list1 = np.array(timeseries.index)[:-result_length]

	index_for_predict = np.array(timeseries)[-result_length:]
	date_list2 = np.array(timeseries.index)[-result_length:]

    #分解
	A2,D2,D1 = pywt.wavedec(index_list,'db4',mode='sym',level=2)
	coeff=[A2,D2,D1]

    # 对每层小波系数求解模型系数
	order_A2 = sm.tsa.arma_order_select_ic(A2,ic='aic')['aic_min_order']
	order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']
	order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']

    #对每层小波系数构建ARMA模型
	model_A2 = ARMA(A2,order=order_A2)
	model_D2 = ARMA(D2,order=order_D2)
	model_D1 = ARMA(D1,order=order_D1)

	results_A2 = model_A2.fit()
	results_D2 = model_D2.fit()
	results_D1 = model_D1.fit()

	A2_all,D2_all,D1_all = pywt.wavedec(np.array(timeseries),'db4',mode='sym',level=2)
	delta = [len(A2_all)-len(A2),len(D2_all)-len(D2),len(D1_all)-len(D1)]

	pA2 = model_A2.predict(params=results_A2.params,start=1,end=len(A2)+delta[0])
	pD2 = model_D2.predict(params=results_D2.params,start=1,end=len(D2)+delta[1])
	pD1 = model_D1.predict(params=results_D1.params,start=1,end=len(D1)+delta[2])

	coeff_new = [pA2,pD2,pD1]
	denoised_index = pywt.waverec(coeff_new,'db4')

	temp_data_wt = {'pre_value':denoised_index[-result_length:]}
	Wavelet_TS = pd.DataFrame(temp_data_wt,index=date_list2,columns=['pre_value'])

	return Wavelet_TS['pre_value']

# 随机森林回归模型
def random_forest_regressor_model(timeseries, result_length):
	train_set,varify_set,predict_set = construct_mechine_learning_set(timeseries, result_length)
	RFR = ensemble.RandomForestRegressor(n_estimators=40)#用20个决策树
	RFR.fit(train_set,varify_set)
	RFR_TS = RFR.predict(predict_set)
	return RFR_TS

# 支持向量机回归模型
def surpport_vector_regressor_model(timeseries, result_length):
	train_set,varify_set,predict_set = construct_mechine_learning_set(timeseries, result_length)
	SVR = svm.SVR()
	SVR.fit(train_set,varify_set)
	SVR_TS = SVR.predict(predict_set)
	return SVR_TS

# KNN回归模型
def k_neighbors_regressor_model(timeseries, result_length):
	train_set,varify_set,predict_set = construct_mechine_learning_set(timeseries, result_length)
	KNN = neighbors.KNeighborsRegressor()
	KNN.fit(train_set,varify_set)
	KNN_TS = KNN.predict(predict_set)
	return KNN_TS

# LSTM模型训练
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model
 
# LSTM模型做出一步预测
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]



###############################################################################################################
# 辅助函数
# 时间序列转监督学习
def timeseries_to_supervised(timeseries, lag = 1):
	df = pd.DataFrame(timeseries)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# 构建机器学习训练集、验证集、测试集
def construct_mechine_learning_set(timeseries, predict_set_length):
	train_set = timeseries[:-predict_set_length]
	temp = timeseries_to_supervised(train_set, 1)
	train_set = np.array(train_set).reshape(-1,1)
	varify_set = temp.iloc[:,0]
	predict_set = np.array(timeseries[-predict_set_length:]).reshape(-1,1)
	return train_set,varify_set,predict_set

# 在时间序列末端插入一行
def add_predict_term_to_timeseries(timeseries, predict_value):
	last_term_date = timeseries.tail(1).index.tolist()
	new_term_date = last_term_date[0] + timedelta(hours = 1)
	new_term = pd.Series(predict_value, index = [new_term_date])
	timeseries = timeseries.append(new_term)
	return timeseries

# 差分
def difference(timeseries, interval=1):
    diff = list()
    for i in range(interval, len(timeseries)):
        value = timeseries[i] - timeseries[i - interval]
        diff.append(value)
    return pd.Series(diff)

# 反向差分
def inverse_difference(history, timeseries_result, interval=1):
    return timeseries_result + history[-interval]

# 缩放预测值到[0,1]
def scale(train, test):
    # fit scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
 
# 反缩放预测值
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]