import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


from sklearn.metrics import roc_curve, auc, precision_recall_curve, fbeta_score,confusion_matrix,classification_report
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
    model = LogisticRegression(C=10, penalty='l2',dual=False,class_weight={0:0.2,1:0.8},solver='sag')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    # alert_estimator_dict = {'Biz':100,'Mon':190,'Ora':150,'Trd:120}
    # depth_dict = {'Biz':19,'Mon':19,'Ora':17,'Trd':17}
    # split_dict= {'Biz':30,'Mon':10,'Ora':10,'Trd':60}'
    #默认参数
    model = RandomForestClassifier(n_estimators=8)
    # model = RandomForestClassifier(oob_score=True, random_state=10)
    model.fit(train_x, train_y)
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
    return model

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
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


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
    model = GradientBoostingClassifier(n_estimators=200)

    # model = GradientBoostingClassifier(n_estimators=200,subsample=0.8)
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

def get_data(data_df,split):
    # 创建空dataframe 存放merge之后的数据
    # col_list = [
    #            'cpu_avg', 'cpu_maxt', 'cpu_max', 'cpu_mint', 'cpu_min',
    #            # 'boot_avg', 'boot_maxt', 'boot_max', 'boot_mint', 'boot_min',
    #            # 'home_avg', 'home_maxt', 'home_max', 'home_mint', 'home_min',
    #            # 'monitor_avg', 'monitor_maxt', 'monitor_max', 'monitor_mint', 'monitor_min',
    #            # 'rt_avg', 'rt_maxt', 'rt_max', 'rt_mint', 'rt_min',
    #            # 'tmp_avg', 'tmp_maxt', 'tmp_max', 'tmp_mint', 'tmp_min',
    #            'mem_avg', 'mem_maxt', 'mem_max', 'mem_mint', 'mem_min',
    #            'cpu_avg_1', 'cpu_maxt_1', 'cpu_max_1', 'cpu_mint_1', 'cpu_min_1',
    #            # 'boot_avg_1', 'boot_maxt_1', 'boot_max_1', 'boot_mint_1', 'boot_min_1',
    #            # 'home_avg_1', 'home_maxt_1', 'home_max_1', 'home_mint_1', 'home_min_1',
    #            # 'monitor_avg_1', 'monitor_maxt_1', 'monitor_max_1', 'monitor_mint_1', 'monitor_min_1',
    #            # 'rt_avg_1', 'rt_maxt_1', 'rt_max_1', 'rt_mint_1', 'rt_min_1',
    #            # 'tmp_avg_1', 'tmp_maxt_1', 'tmp_max_1', 'tmp_mint_1', 'tmp_min_1',
    #            'mem_avg_1', 'mem_maxt_1', 'mem_max_1', 'mem_mint_1', 'mem_min_1',
    #            'cpu_avg_2', 'cpu_maxt_2', 'cpu_max_2', 'cpu_mint_2', 'cpu_min_2',  # 创建空dataframe 存放merge之后的数据
    #            # 'boot_avg_2', 'boot_maxt_2', 'boot_max_2', 'boot_mint_2', 'boot_min_2',
    #            # 'home_avg_2', 'home_maxt_2', 'home_max_2', 'home_mint_2', 'home_min_2',
    #            # 'monitor_avg_2', 'monitor_maxt_2', 'monitor_max_2', 'monitor_mint_2', 'monitor_min_2',
    #            # 'rt_avg_2', 'rt_maxt_2', 'rt_max_2', 'rt_mint_2', 'rt_min_2',
    #            # 'tmp_avg_2', 'tmp_maxt_2', 'tmp_max_2', 'tmp_mint_2', 'tmp_min_2',
    #            'mem_avg_2', 'mem_maxt_2', 'mem_max_2', 'mem_mint_2', 'mem_min_2',
    #             'alarm_count','event']
    col_list = ['cpu_max', 'cpu_min',
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
                     'event']
    data = data_df[col_list]
    data = data.convert_objects(convert_numeric=True)
    # for col in col_list:
    #     data[col] = pd.to_numeric(data[col], errors='coerce')
    feature_data = data.drop('event', axis=1)
    label_data = data.event
    if split==True:
        return train_test_split(feature_data,label_data,test_size=0.2,random_state=800)
        #return feature_data, test_feature_data, label_data, test_label_data
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


def classifiers_for_prediction(data_file, model_save_file,predict_proba_file,result_file):
    model_save = {}
    test_classifiers_list = ['GBDT',
                              'KNN',
                             'LR',
                             'RF',
                             'DT'
                             ]
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    result_list = []
    # ignored_list = []
    all_data =  pd.read_csv(data_file, sep=',', dtype=str)
    for alertgroup,group in all_data.groupby('alertgroup'):
        if alertgroup != 'Net' :
        # if alertgroup == 'Ora' :
            print(alertgroup)
            print(group['event'].value_counts())
            print('reading training and testing data...')

            train_x, test_x, train_y, test_y = get_data(group,split=True)
            # train_x, test_x, train_y, test_y = read_data(data_file,split=True)

            for classifier in test_classifiers_list:
                print('******************* %s ********************' % classifier)
                start_time = time.time()
                model = classifiers[classifier](train_x, train_y)
                print('training took %fs!' % (time.time() - start_time))

                predict = model.predict(test_x)
                print(predict)
                # predict_proba = model.predict(test_x)
                if(classifier == 'SVM'):
                    test_x = MinMaxScaler().fit_transform(test_x)
                # predict_proba = model.predict_proba(test_x)[:,1]
                if model_save_file != None:
                    model_save[classifier] = model


                #画图
                # generate_ROC_plot(test_y, predict_proba, classifier
                # generate_PR_plot(test_y, predict_proba, classifier)
                # generate_learning_curve(data_file, model, classifier)
                # generate_compared_curve(test_y,predict_proba,classifier)

                # predict_proba[predict_proba >= 0.5] = 1
                # predict_proba[predict_proba < 0.5] = 0
                # predict_proba = predict_proba.astype(np.int64)
                #print(predict_proba)
 #多分类混淆矩阵
                # confusion_mat = confusion_matrix(test_y,predict)
                # print(confusion_mat)
                # confusion_mat[1,1] = 0
                # plot_confusion_matrix(confusion_mat)
                # print(classification_report(test_y,predict))


#评价指标
#000000

                precision = metrics.precision_score(test_y, predict)
                recall = metrics.recall_score(test_y, predict)
                fbetascore = fbeta_score(test_y, predict, 0.5)
                accuracy = metrics.accuracy_score(test_y, predict)
                model_score = model.score(test_x, test_y)
                print('precision: %.6f' % (100 *precision))
                print('recall: %.6f' % (100 * recall))
                print('f0.5score: %.6f' % (100 * fbetascore))
                print('model score: %.6f' % (100*model_score))
                print('accuracy: %.6f%%' % (100 * accuracy))

                # alertgroup_list.append(alertgroup)
                # classifier_list.append(classifier)
                # precision_list.append(precision)
                # recall_list.append(recall)
                result_list.append([alertgroup,classifier,precision,recall,fbetascore,accuracy,model_score])


                #多分类评价指标

#0000000

                # precision = metrics.precision_score(test_y, predict, average="micro")
                # recall = metrics.recall_score(test_y, predict, average="micro")
                # fbetascore = fbeta_score(test_y, predict, 0.5, average="micro")
                # print(
                #     'precision: %.6f%%, recall: %.6f%%, f0.5score: %.6f%%' % (100 * precision, 100 * recall, 100 * fbetascore))
                # print('model score: %.6f' % (model.score(test_x, test_y)))
                # accuracy = metrics.accuracy_score(test_y, predict)
                # print('accuracy: %.6f%%' % (100 * accuracy))

                # print('predict proba 1 = {0}%'.format(100*(predict[predict == 1].sum() / predict.size)))
                # print('test 1 = {0}%'.format(100 * (test_y[test_y == 1].sum() / test_y.size)))


                # np.savetxt(predict_proba_file,predict_proba)



    #

            if model_save_file != None:
                pickle.dump(model_save, open(model_save_file, 'wb'))

    result_df = pd.DataFrame(result_list,
                             columns=['alertgroup','classifier','precision','recall','fbetascore','accuracy','model_score'])
    print(result_df)
    result_df.to_csv(result_file,sep=',',index=False)

