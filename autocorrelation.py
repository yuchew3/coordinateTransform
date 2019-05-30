import pandas as pd
import numpy as np
import ca_data_utils
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

def load_data(step):
    v = ca_data_utils.load_v_matrix().T[9:39992:step]
    labels = ca_data_utils.load_labels()[9:39992:step]
    return v, labels
    


if __name__ == "__main__":
    X, labels = load_data(2)
    # X = X[:,::4]
    # print(X.shape)
    length = 10
    # lags = 5
    # # dataset = np.fromfunction(lambda i, j: pd.Series(X[i]).autocorr(lag=j+1), (len(X), lags))
    # # for i in range(X.shape[0]-length):
    # #     series = pd.Series(X[:,i])
    # #     for j in range(lags):
    # #         corr = series.autocorr(lag=j+1)
    # #         dataset[i, j] = corr

    # dataset = np.zeros((X.shape[0]-length+1, X.shape[1]*lags))
    # for i in range(dataset.shape[0]):
    #     x = X[i:i+length]
    #     for p in range(X.shape[1]):
    #         series = pd.Series(x[:,p])
    #         for j in range(lags):
    #             corr = series.autocorr(lag=j+1)
    #             dataset[i, p*lags+j] = corr
    # print(dataset.shape)
    labels = labels[length-1:]
    dataset = np.load('../data/autocorr_2.npy')
    total_len = len(dataset)
    cut = int(3 * total_len / 4)

    # np.save('../data/autocorr_2', dataset)
    print(dataset.shape)

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    for name, clf in zip(names, classifiers):
        print('starting ', name)
        clf.fit(dataset[:cut], labels[:cut])
        score = clf.score(dataset[cut:], labels[cut:])
        print('---> test accuracy is ', str(score))