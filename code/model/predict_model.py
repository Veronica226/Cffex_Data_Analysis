import time

import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import svm,datasets,metrics
from sklearn.externals import joblib
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
    model = LogisticRegression(penalty='l2',class_weight="balanced")
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
    model = SVC(kernel='rbf', class_weight="balanced" ,probability=True)
    model.fit(train_x, train_y)
    print("fit scuuess")
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


def read_data(data_file):
    data = pd.read_csv(data_file, sep=',',usecols=['cpu_max', 'cpu_min',       #创建空dataframe 存放merge之后的数据
                                    'boot_max', 'boot_min','home_max', 'home_min',
                                   'monitor_max', 'monitor_min','rt_max', 'rt_min',
                                    'tmp_max', 'tmp_min','mem_max', 'mem_min','event'],dtype=float)
    train = data[:int(len(data) * 0.9)]         #划分训练数据和测试数据
    test = data[int(len(data) * 0.9):]
    train_y = train.event
    train_x = train.drop('event', axis=1)
    test_y = test.event
    test_x = test.drop('event', axis=1)
    return train_x, train_y, test_x, test_y

def generate_ROC_plot(test_y, predict,classifier):
    FP, TP, thresholds = roc_curve(test_y, predict)
    ROC_auc = auc(FP, TP)
    plt.title(classifier+'- ROC CURVE')
    plt.plot(FP, TP, 'b', label='AUC = %0.2f' % ROC_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def generate_PR_plot(test_y, predict,classifier):
    precision, recall, thresholds = precision_recall_curve(test_y, predict)
    plt.title(classifier+'- PR CURVE')
    plt.plot(precision, recall, 'r')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.show()



def classifiers_for_prediction(data_file, model_save_file):
    model_save = {}

    test_classifiers = [ 'SVM']
    classifiers = {#'NB': naive_bayes_classifier,
                   # 'KNN': knn_classifier,
                   # 'LR': logistic_regression_classifier,
                   # 'RF': random_forest_classifier,
                   # 'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   # 'SVMCV': svm_cross_validation,
                   # 'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data(data_file)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        # predict = model.predict(test_x)
        predict_proba = model.predict_proba(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        precision = metrics.precision_score(test_y, predict_proba)
        recall = metrics.recall_score(test_y, predict_proba)
        f1score = f1_score(test_y, predict_proba)
        print('precision: %.2f%%, recall: %.2f%%, f1score: %.2f%%' % (100 * precision, 100 * recall, 100 * f1score))
        accuracy = metrics.accuracy_score(test_y, predict_proba)
        print('accuracy: %.2f%%' % (100 * accuracy))
        generate_ROC_plot(test_y, predict_proba,classifier)
        generate_PR_plot(test_y, predict_proba, classifier)


    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))