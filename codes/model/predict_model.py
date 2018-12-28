import time
import numpy as np
import pandas as pd
# import xgboost as xgb
import matplotlib.pyplot as plt
import random
# import lightgbm as lgb


from sklearn.metrics import roc_curve, auc, precision_recall_curve, fbeta_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit,RandomizedSearchCV
from sklearn import svm,datasets,metrics
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.externals import joblib
from settings import *
from sklearn.model_selection import KFold
# from imblearn.combine import SMOTEENN
import pickle
import pandas as pd
# from xgboost import *

######################################################################################
#Author: 王靖文

#几种分类器模型
# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    arr_x = train_x.values
    arr_y = train_y.values
    kf = KFold(n_splits=3)
    max_acc = 0
    max_fs = 0
    best_model = None
    for train_index, test_index in kf.split(arr_x):
        model = KNeighborsClassifier()
        train_x = arr_x[train_index]
        train_y = arr_y[train_index]
        test_x = arr_x[test_index]
        test_y = arr_y[test_index]
        model.fit(train_x, train_y)
        predict = model.predict(test_x)
        acc = metrics.accuracy_score(test_y, predict)
        fbetascore = fbeta_score(test_y, predict, 0.5)
        print('acc:'+ str(acc)+'  f0.5score:'+str(fbetascore))
        if fbetascore > max_fs:
            max_fs = fbetascore
            best_model = model
    return best_model



# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(C=10, penalty='l2',dual=False,class_weight={0:0.2,1:0.8},solver='sag')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    arr_x = train_x.values
    arr_y = train_y.values
    kf = KFold(n_splits=5)
    max_acc = 0
    max_fs = 0
    best_model = None
    for train_index, test_index in kf.split(arr_x):
        model = RandomForestClassifier()
        train_x = arr_x[train_index]
        train_y = arr_y[train_index]
        test_x = arr_x[test_index]
        test_y = arr_y[test_index]
        model.fit(train_x, train_y)
        predict = model.predict(test_x)
        acc = metrics.accuracy_score(test_y, predict)
        fbetascore = fbeta_score(test_y, predict, 0.5)
        print('acc:' + str(acc) + '  f0.5score:' + str(fbetascore))
        if fbetascore > max_fs:
            max_fs = fbetascore
            best_model = model
    return best_model
    # arr_x = train_x.values
    # arr_y = train_y.values
    # kf = KFold(n_splits = 5)
    # max_acc=0
    # max_fs = 0
    # best_model = None
    # for train_index,test_index in kf.split(arr_x):
    #     param_dist = {
    #         'n_estimators': range(80, 201, 20),
    #         'max_depth': range(10, 15, 1),
    #         'min_samples_leaf':range(10,101,10)
    #     }
    #     model = RandomizedSearchCV(RandomForestClassifier(),param_dist,cv=5,n_iter = 300,n_jobs = -1)
        # model = RandomForestClassifier(n_estimators=8,max_depth=13, n_jobs=-1) # max_depth > 10
        # model = RandomForestClassifier(oob_score=True, random_state=10)
        # train_x = arr_x[train_index]
        # train_y = arr_y[train_index]
        # test_x = arr_x[test_index]
        # test_y = arr_y[test_index]
        # model.fit(train_x, train_y)
        # print(model.best_score_)
        # print(model.best_estimator_)
        # print(model.best_params_)
        # # predict = model.predict(test_x)
        # # acc = metrics.accuracy_score(test_y,predict)
        # # fbetascore = fbeta_score(test_y, predict, 0.5)
        # # print('acc:' + str(acc) + '  f0.5score:' + str(fbetascore))
        # # if fbetascore > max_fs:
        # #     max_fs = fbetascore
        # #     best_model = model
        #
        #
        # return model.best_estimator_
        #




    # alert_estimator_dict = {'Biz':100,'Mon':190,'Ora':150,'Trd:120}
    # depth_dict = {'Biz':19,'Mon':19,'Ora':17,'Trd':17}
    # split_dict= {'Biz':30,'Mon':10,'Ora':10,'Trd':60}'
    #默认参数
    # print(model.feature_importances_)
    # print(model.oob_score_)
    # y_pre = model.predict_proba(train_x)[:,1]
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, y_pre))
    #找最佳迭代次数 n_estimator
    # param_test1 = {'n_estimators': range(10, 201, 10)}
    # model = RandomForestClassifier(n_estimators= alert_estimator_dict[alertgroup],min_samples_split=60,
    #              min_samples_leaf = 20, max_depth = depth_dict[alertgroup], max_features = 'sqrt', random_state = 10,oob_score=True)
    #
    # model.fit(train_x, train_y)
    # print(model.oob_score_)
    # y_pre = model.predict_proba(train_x)[:,1]
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, y_pre))
    # return model

    # 找max_depth 和 min_samples_split
    # param_test2 = {'max_depth': range(13,21,2), 'min_samples_split': range(10,51,10)}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=alert_estimator_dict[alertgroup],
    #                                                          min_samples_leaf=20, max_features='sqrt', oob_score=True,
    #                                                          random_state=10),
    #                         param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)

    #找min_samples_leaf
    # print('find max feature')
    # param_test3 = {'max_features':range(3,12,2)}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=alert_estimator_dict[alertgroup], max_depth=depth_dict[alertgroup],
    #         min_samples_split=split_dict[alertgroup], min_samples_leaf=10,
    #          oob_score = True, random_state = 10),
    #         param_grid = param_test3, scoring = 'roc_auc', iid = False, cv = 5)
    # gsearch1.fit(train_x, train_y)
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    #
    # return gsearch1


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    arr_x = train_x.values
    arr_y = train_y.values
    kf = KFold(n_splits=3)
    max_acc = 0
    max_fs = 0
    best_model = None
    for train_index, test_index in kf.split(arr_x):
        model  = tree.DecisionTreeClassifier()
        train_x = arr_x[train_index]
        train_y = arr_y[train_index]
        test_x = arr_x[test_index]
        test_y = arr_y[test_index]
        model.fit(train_x, train_y)
        predict = model.predict(test_x)
        acc = metrics.accuracy_score(test_y, predict)
        fbetascore = fbeta_score(test_y, predict, 0.5)
        print('acc:' + str(acc) + '  f0.5score:' + str(fbetascore))
        if fbetascore > max_fs:
            max_fs = fbetascore
            best_model = model
    return best_model


# GBDT(Gradient Boosting Decision Tree) Classifier

def gradient_boosting_classifier(train_x, train_y):
    # alert_estimater_dict = {'Biz':320,'Mon':370,'Ora':320,'Trd':180}
    #
    # # param_test1 = {'n_estimators': range(200, 401, 10)}
    # # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
    # #                                                              min_samples_leaf=20, max_depth=8, max_features='sqrt',
    # #                                                              subsample=0.8, random_state=10),
    # #                         param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    #
    # param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    # gsearch2 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=alert_estimater_dict[alertgroup], min_samples_leaf=20,
    #                                          max_features='sqrt', subsample=0.8, random_state=10),
    #     param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    # gsearch2.fit(train_x, train_y)
    # print(gsearch2.grid_scores_)
    # print(gsearch2.best_params_)
    # print(gsearch2.best_score_)
    # return gsearch2
    #


    # model = GradientBoostingClassifier(n_estimators=200,subsample=0.8)
    arr_x = train_x.values
    arr_y = train_y.values
    kf = KFold(n_splits=3)
    max_acc = 0
    max_fs = 0
    best_model = None
    for train_index, test_index in kf.split(arr_x):
        model = GradientBoostingClassifier(n_estimators=200)
        train_x = arr_x[train_index]
        train_y = arr_y[train_index]
        test_x = arr_x[test_index]
        test_y = arr_y[test_index]
        model.fit(train_x, train_y)
        predict = model.predict(test_x)
        acc = metrics.accuracy_score(test_y, predict)
        fbetascore = fbeta_score(test_y, predict, 0.5)
        print('acc:' + str(acc) + '  f0.5score:' + str(fbetascore))
        if fbetascore > max_fs:
            max_fs = fbetascore
            best_model = model
    return best_model

# def xgboost_classifier(train_x,train_y):
#     dtrain = xgb.Dmatrix(train_x,train_y)
#     params = {'booster': 'gbtree',
#               'objective': 'binary:logistic',
#               'eval_metric': 'auc',
#               'max_depth': 4,
#               'lambda': 10,
#               'subsample': 0.75,
#               'colsample_bytree': 0.75,
#               'min_child_weight': 2,
#               'eta': 0.025,
#               'seed': 0,
#               'nthread': 8,
#               'silent': 1}
#     watchlist = [(dtrain, 'train')]
#     bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
#     return bst

# def lightgbm_classifier(train_x,train_y):
#     lgb_train = lgb.Dataset(train_x,train_y,free_raw_data=False)
#     params = {
#     'boosting_type': 'gbdt',
#     'boosting': 'dart',
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'learning_rate': 0.01,
#     'num_leaves': 25,
#     'max_depth': 3,
#     'max_bin': 10,
#     'min_data_in_leaf': 8,
#     'feature_fraction': 0.6,
#     'bagging_fraction': 1,
#     'bagging_freq': 0,
#     'lambda_l1': 0,
#     'lambda_l2': 0,
#     'min_split_gain': 0}
#     gbm = lgb.train(params,lgb_train,num_boost_round=2000,       # 迭代次数
#                valid_sets=lgb_eval,        # 验证集
#                early_stopping_rounds=30)   # 早停系数)



# SVM Classifier
def svm_classifier(train_x, train_y):
    model = SVC(kernel='rbf', class_weight='balanced' , max_iter=1000, probability=True,  random_state=2018)  #max_iter=5000，而且计算概率，要跑20min
    #model = SVC(kernel='rbf', class_weight="balanced", max_iter=200, probability=True, random_state=2018)
    scaler = MinMaxScaler()
    train_x_standard = scaler.fit_transform(train_x)  #支持向量机要对数据做归一化或者标准化处理，这里将数据归一化到0-1区间
    model.fit(train_x_standard, train_y)
    print("fit success")
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

#从文件中读取并返回模型所用的数据
def get_data(data_df,split):
    # 创建空dataframe 存放merge之后的数据
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
                'mem_min_2',
                # 'alarm_count',
                'event']
                # 'pre_event']

    data = data_df[col_list]
    # data.replace(-np.inf, np.nan)
    # data.fillna(0)
    data = data.convert_objects(convert_numeric=True)
    # print(data)
    feature_data = data.drop('event', axis=1)
    label_data = data.event

    if split==True:
        # train_x, valid_test_x,train_y,valid_test_y = train_test_split(feature_data,label_data,test_size=0.4)
        # valid_x,test_x,valid_y,test_y = train_test_split(valid_test_x,valid_test_y,test_size=0.5)
        #
        # return train_x,valid_x,test_x,train_y,valid_y,test_y
        return train_test_split(feature_data,label_data,test_size=0.4)
    else:
        return feature_data, label_data

def read_data(data_file,split):
    # data = pd.read_csv(data_file, sep=',',usecols=['cpu_max', 'cpu_min',       #创建空dataframe 存放merge之后的数据
    #                                 'boot_max', 'boot_min','home_max', 'home_min',
    #                                'monitor_max', 'monitor_min','rt_max', 'rt_min',
    #                                 'tmp_max', 'tmp_min','mem_max', 'mem_min','event'],dtype=np.float64)
    # 创建空dataframe 存放merge之后的数据
    data = pd.read_csv(data_file, sep=',', usecols=['cpu_max', 'cpu_min',
                                                    # 'boot_max', 'boot_min', 'home_max', 'home_min',
                                                    # 'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                                                    # 'tmp_max', 'tmp_min',
                                                      'mem_max', 'mem_min',
                                                     'cpu_max_1', 'cpu_min_1',
                                                    # 'boot_max_1', 'boot_min_1','home_max_1', 'home_min_1',
                                                    # 'monitor_max_1', 'monitor_min_1','rt_max_1', 'rt_min_1',
                                                    # 'tmp_max_1', 'tmp_min_1',
                                                     'mem_max_1', 'mem_min_1',
                                                      'cpu_max_2', 'cpu_min_2',
                                                    # 'boot_max_2', 'boot_min_2', 'home_max_2', 'home_min_2',
                                                    # 'monitor_max_2', 'monitor_min_2', 'rt_max_2', 'rt_min_2',
                                                    # 'tmp_max_2', 'tmp_min_2',
                                                      'mem_max_2', 'mem_min_2',
                                                    'event','alertgroup'], dtype=np.float64)

    # train = data[:int(len(data) * 0.8)]         #划分训练数据和测试数据
    # test = data[int(len(data) * 0.8):]
    # train_y = train.event
    # train_x = train.drop('event', axis=1)
    # test_y = test.event
    # test_x = test.drop('event', axis=1)
    feature_data = data.drop('event', axis=1)
    label_data = data.event

    # positive_label_number = len(data[data.event == 1]) #正样本数量
    # positive_index_list = np.array(data[data.event==1].index)   #正样本索引值
    # nagetive_index_list = np.array(data[data.event==0].index)   #负样本索引值
    # random_nagetive_indices = np.random.choice(nagetive_index_list, positive_label_number*3,
    #                                          replace=False)  # 随机采样，并不对原始dataframe进行替换
    # random_nagetive_indices = np.array(random_nagetive_indices)  # 转换成numpy的array格式转换成矩阵
    # under_sample_indices = np.concatenate([positive_index_list, random_nagetive_indices])  # 将两组索引数据连接成新的数据索引
    # under_sample_data = data.iloc[under_sample_indices, :]            #定位到真正数据，iloc通过行号索引行数据
    #
    # feature_data = under_sample_data.loc[:, under_sample_data.columns != 'event']
    # label_data = under_sample_data.loc[:, under_sample_data.columns == 'event']
    #
    #
    # test_feature_data = data.drop(['event'], axis=1)
    # test_label_data = data.loc[:, 'event']

    if split==True:
        return train_test_split(feature_data,label_data,test_size=0.2,random_state=800)
        #return feature_data, test_feature_data, label_data, test_label_data
    else:
        return feature_data, label_data


#画图
def generate_ROC_plot(test_y, predict,classifier_name):
    if not os.path.exists(history_metric_figures_dir):
        os.makedirs(history_metric_figures_dir)
    FP, TP, thresholds = roc_curve(test_y, predict)
    ROC_auc = auc(FP, TP)
    fig = plt.figure()
    plt.title(classifier_name+'- ROC CURVE')
    plt.plot(FP, TP, 'b', label='AUC = %0.2f' % ROC_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    roc_plot_path = os.path.join(history_metric_figures_dir, classifier_name + 'nocpu_ROC_CURVE.png')
    fig.savefig(roc_plot_path, dpi=100)
    #plt.show()


def generate_PR_plot(test_y, predict,classifier_name):
    if not os.path.exists(history_metric_figures_dir):
        os.makedirs(history_metric_figures_dir)
    precision, recall, thresholds = precision_recall_curve(test_y, predict)
    fig = plt.figure()
    plt.title(classifier_name+'- PR CURVE')
    plt.plot(precision, recall, 'b')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    pr_plot_path = os.path.join(history_metric_figures_dir, classifier_name + '_PR_CURVE.png')
    fig.savefig(pr_plot_path, dpi=100)
    #plt.show()

def generate_learning_curve(data_file,model,classifier_name):
    if not os.path.exists(history_metric_figures_dir):
        os.makedirs(history_metric_figures_dir)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    X, Y = read_data(data_file, split=False)
    print('start drawing...')
    train_sizes, train_loss, test_loss = learning_curve(
        model, X, Y, cv=cv)
    print('finish drawing...')
    # 平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
    train_loss_mean = np.mean(train_loss, axis=1)
    test_loss_mean = np.mean(test_loss, axis=1)
    fig = plt.figure()
    plt.plot(train_sizes, train_loss_mean, 'o-', color="r",label="Training")
    plt.plot(train_sizes, test_loss_mean, 'o-', color="g",label="Cross-validation")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.title(classifier_name + '- LEARNING CURVE')
    pr_plot_path = os.path.join(history_metric_figures_dir, classifier_name + '_learning-curve.png')
    fig.savefig(pr_plot_path, dpi=100)
    #plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

def generate_compared_curve(test_y,predict_proba,classifier_name):
    if not os.path.exists(metric_figures_dir):
        os.makedirs(metric_figures_dir)
    fig = plt.figure()
    test_y = test_y[1:50]
    predict_proba = predict_proba[1:50]
    x = np.arange(1, len(predict_proba)+1)
    plt.plot(x,test_y, marker='.', color="lightpink",label="practice")
    plt.plot(x, predict_proba, marker ='+',color="lightblue",label="predict")
    plt.xlabel("Training examples")
    plt.ylabel("label")
    plt.legend(loc="best")
    plt.title(classifier_name + '- compared')
    plt.show()
    pr_plot_path = os.path.join(metric_figures_dir, classifier_name + '_compared-curve.png')
    fig.savefig(pr_plot_path, dpi=100)


def plot_confusion_matrix(confusion_mat):
    '''将混淆矩阵画图并显示出来'''
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.spring)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(confusion_mat.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_time_series_data(ts_result_file):
    data = pd.read_csv(ts_result_file, sep=',', dtype=str)
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

    host_d = data['hostname']
    ts_d = data[col_list]
    ts_d = ts_d.convert_objects(convert_numeric=True)
    return host_d,ts_d

def classifiers_for_prediction(data_file,model_save_file,result_file,roc_plot_data_dir):

    model_save = {}
    test_classifiers_list = [ 'RF',
                             'GBDT',
                              'LR',
                               'KNN',
                               'DT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   # 'XGB':xgboost_classifier
                   }


    all_data =  pd.read_csv(data_file, sep=',', dtype=str)
    for alertgroup,group in all_data.groupby('alertgroup'):
        if alertgroup != 'Net':
            print(alertgroup)
            # print(group['event'].value_counts())
            print('reading training and testing data...')

            train_x, test_x, train_y, test_y = get_data(group,split=True)
            # print(test_y.value_counts())
            # sm = SMOTEENN()
            # X_resampled, y_resampled = sm.fit_sample(train_x, train_y)
            # train_x = pd.DataFrame(X_resampled.tolist())
            # train_y = pd.DataFrame(y_resampled)
            # train_x, test_x, train_y, test_y = read_data(data_file,split=True)
            all_df = pd.DataFrame(columns=['classifier','hostname', 'predict_event'])
            roc_df = pd.DataFrame(columns=['real'])
            roc_df['real'] = test_y

            for classifier in test_classifiers_list:
                print('******************* %s ********************' % classifier)
                start_time = time.time()
                model = classifiers[classifier](train_x, train_y)
                print('training took %fs!' % (time.time() - start_time))

                # predict = model.predict(test_x)

                # print((predict.sum())/len(predict))
                # predict_proba = model.predict(test_x)
                predict_proba = model.predict_proba(test_x)[:,1]
                roc_df[classifier] = predict_proba
                # if model_save_file != None:
                #     model_save[alertgroup][classifier] = model


                #画图
                # generate_ROC_plot(test_y, predict_proba, classifier
                # generate_PR_plot(test_y, predict_proba, classifier)
                # generate_learning_curve(data_file, model, classifier)
                # generate_compared_curve(test_y,predict_proba,classifier)

                # predict_proba[predict_proba >= 0.5] = 1
                # predict_proba[predict_proba < 0.5] = 0
                # predict_proba = predict_proba.astype(np.int64)
                #print(predict_proba)

#评价指标
#000000
                #
                # print(len([v1 for (v1,v2) in zip(test_y,predict) if v1 != v2]))
                #
                # precision = metrics.precision_score(test_y, predict)
                # recall = metrics.recall_score(test_y, predict)
                # fbetascore = fbeta_score(test_y, predict, 0.5)
                # accuracy = metrics.accuracy_score(test_y, predict)
                # model_score = model.score(test_x, test_y)
                # print('precision: %.6f' % (100 *precision))
                # print('recall: %.6f' % (100 * recall))
                # print('f0.5score: %.6f' % (100 * fbetascore))
                # print('model score: %.6f' % (100*model_score))
                # print('accuracy: %.6f%%' % (100 * accuracy))

                # alertgroup_list.append(alertgroup)
                # classifier_list.append(classifier)
                # precision_list.append(precision)
                # recall_list.append(recall)
                # result_list.append([alertgroup,classifier,precision,recall,fbetascore,accuracy,model_score])

            # if model_save_file != None:
            #     pickle.dump(model_save, open(model_save_file, 'wb'))

            # all_df.to_csv(result_file,sep=',',index=False)

            print(roc_df)
            roc_df_file = os.path.join(roc_plot_data_dir, alertgroup + '_roc_data_proba.csv')
            roc_df.to_csv(roc_df_file,sep=',',index=False)
    # result_df = pd.DataFrame(result_list,
    #                          columns=['alertgroup','classifier','precision','recall','fbetascore','accuracy','model_score'])
    # print(result_df)
    # result_df.to_csv(result_file,sep=',',index=False)


#根据业务返回已经训练好的具体分类器模型
def test_classifier_for_prediction(data_file,alertgroup_name,classifier):
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   # 'XGB':xgboost_classifier
                   }
    all_data = pd.read_csv(data_file, sep=',', dtype=str)
    for alertgroup, group in all_data.groupby('alertgroup'):
        if alertgroup == alertgroup_name:
            train_x, test_x, train_y, test_y = get_data(group, split=True)
            model = classifiers[classifier](train_x, train_y)

    return model
