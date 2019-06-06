import pandas as pd
import numpy as np
import ca_data_utils
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV

def load_auto_data():
    lags = 10
    length = 2 * lags
    fn = '../data/autocorr_len20_lag10.npy'
    X = np.load(fn)
    labels = ca_data_utils.load_labels()[9:39992:2]
    labels = labels[length-1:]
    print(X.shape)
    print(labels.shape)
    max_lens = np.linspace(3,8,6)
    learning_rates = np.logspace(-2,2,5)
    n_estimators = np.linspace(100,200,5)
    boosters = ['gbtree', 'gblinear','dart']
    param_dict = dict(max_depth=max_lens, learning_rate=learning_rates,n_estimators=n_estimators,booster=boosters)
    grid = GridSearchCV(xgb.XGBClassifier(), param_grid=param_dict, n_jobs=20)

    print('start to train...')
    grid.fit(X, labels)
    print('finished')
    df = pd.DataFrame.from_dict(grid.cv_results_)
    filename = '../data/clf_results/autocorr_xgbooster'
    df.to_csv(filename)




def load_data(step):
    v = ca_data_utils.load_v_matrix().T[9:39992:step]
    labels = ca_data_utils.load_labels()[9:39992:step]
    return v, labels

def one_iter(length, lags):
    X, labels = load_data(2)
    X = X[:,::4]
    print(X.shape)
    # length = 10
    # lags = 5
    # dataset = np.fromfunction(lambda i, j: pd.Series(X[i]).autocorr(lag=j+1), (len(X), lags))
    # for i in range(X.shape[0]-length):
    #     series = pd.Series(X[:,i])
    #     for j in range(lags):
    #         corr = series.autocorr(lag=j+1)
    #         dataset[i, j] = corr

    if lags == 6:
        dataset = np.load('../data/autocorr_len12_lag6.npy')
        
    else:
        dataset = np.zeros((X.shape[0]-length+1, X.shape[1]*lags))
        for i in range(dataset.shape[0]):
            x = X[i:i+length]
            for p in range(X.shape[1]):
                series = pd.Series(x[:,p])
                for j in range(lags):
                    corr = series.autocorr(lag=j+1)
                    dataset[i, p*lags+j] = corr
        print(dataset.shape)
        fn = '../data/autocorr_len' + str(length) + '_lag' + str(lags)
        np.save(fn, dataset)
        print(dataset.shape)

    labels = labels[length-1:]
    total_len = len(dataset)
    cut = int(3 * total_len / 4)

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=0.001),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        xgb.XGBClassifier()]
    
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "XGBoost"]

    scores = []
    for name, clf in zip(names, classifiers):
        print('starting ', name)
        clf.fit(dataset[:cut], labels[:cut])
        score = clf.score(dataset[cut:], labels[cut:])
        scores.append(score)
        print('---> test accuracy is ', str(score))
    fn = '../data/clf_results/autocorr_len' + str(length) + '_lag' + str(lags)
    np.save(fn, scores)

if __name__ == "__main__":
    load_auto_data()
    # for i in range(6, 11):
    #     print('lags = ', i)
    #     one_iter(i*2, i)