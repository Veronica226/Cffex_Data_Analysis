import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


from sklearn.metrics import roc_curve, auc, precision_recall_curve, fbeta_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn import svm,datasets,metrics
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.externals import joblib
from settings import *
import pickle
import pandas as pd

######################################################################################
#Author: 王靖文


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(C=10, penalty='l2',class_weight='balanced')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


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


def read_data(data_file,split):
    # data = pd.read_csv(data_file, sep=',',usecols=['cpu_max', 'cpu_min',       #创建空dataframe 存放merge之后的数据
    #                                 'boot_max', 'boot_min','home_max', 'home_min',
    #                                'monitor_max', 'monitor_min','rt_max', 'rt_min',
    #                                 'tmp_max', 'tmp_min','mem_max', 'mem_min','event'],dtype=np.float64)

    data = pd.read_csv(data_file, sep=',', usecols=['cpu_max', 'cpu_min',  # 创建空dataframe 存放merge之后的数据
                                                    'boot_max', 'boot_min', 'home_max', 'home_min',
                                                    'monitor_max', 'monitor_min', 'rt_max', 'rt_min',
                                                    'tmp_max', 'tmp_min', 'mem_max', 'mem_min',
                                                     'cpu_max_1', 'cpu_min_1', 'boot_max_1', 'boot_min_1',
                                                      'home_max_1', 'home_min_1', 'monitor_max_1', 'monitor_min_1',
                                                     'rt_max_1', 'rt_min_1', 'tmp_max_1', 'tmp_min_1', 'mem_max_1', 'mem_min_1',
                                                      'cpu_max_2', 'cpu_min_2', 'boot_max_2', 'boot_min_2',
                                                      'home_max_2', 'home_min_2', 'monitor_max_2', 'monitor_min_2',
                                                     'rt_max_2', 'rt_min_2', 'tmp_max_2', 'tmp_min_2', 'mem_max_2', 'mem_min_2','event'], dtype=np.float64)

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
    roc_plot_path = os.path.join(history_metric_figures_dir, classifier_name + '_ROC_CURVE.png')
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

    pr_plot_path = os.path.join(metric_figures_dir, classifier_name + '_compared-curve.png')
    fig.savefig(pr_plot_path, dpi=100)
    #plt.show()

def classifiers_for_prediction(data_file, model_save_file,predict_proba_file):
    model_save = {}

    test_classifiers_list = ['GBDT','KNN','LR','RF','DT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    train_x,  test_x,train_y, test_y = read_data(data_file,split=True)

    for classifier in test_classifiers_list:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        # predict = model.predict(test_x)
        #predict_proba = model.predict(test_x)
        if(classifier == 'SVM'):
            test_x = MinMaxScaler().fit_transform(test_x)
        predict_proba = model.predict_proba(test_x)[:,1]
        if model_save_file != None:
            model_save[classifier] = model

        generate_ROC_plot(test_y, predict_proba, classifier)
        generate_PR_plot(test_y, predict_proba, classifier)
        #generate_learning_curve(data_file, model, classifier)

        generate_compared_curve(test_y,predict_proba,classifier)

        predict_proba[predict_proba >= 0.5] = 1
        predict_proba[predict_proba < 0.5] = 0
        predict_proba = predict_proba.astype(np.int64)
        #print(predict_proba)
        precision = metrics.precision_score(test_y, predict_proba)
        recall = metrics.recall_score(test_y, predict_proba)
        fbetascore = fbeta_score(test_y, predict_proba, 0.5)
        print('precision: %.6f%%, recall: %.6f%%, f0.5score: %.6f%%' % (100 * precision, 100 * recall, 100 * fbetascore))
        print('model score: %.6f' % (model.score(test_x, test_y)))
        accuracy = metrics.accuracy_score(test_y, predict_proba)
        print('accuracy: %.6f%%' % (100 * accuracy))
        print('predict proba 1 = {0}%'.format(100*(predict_proba[predict_proba == 1].sum() / predict_proba.size)))
        print('test 1 = {0}%'.format(100 * (test_y[test_y == 1].sum() / test_y.size)))
        # np.savetxt(predict_proba_file,predict_proba)

        # generate_ROC_plot(test_y, predict_proba,classifier)
        # generate_PR_plot(test_y, predict_proba, classifier)
        # generate_learning_curve(data_file, model, classifier)


    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))