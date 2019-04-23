import numpy as np
import matplotlib.pyplot as plt
import ca_data_utils

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def main():
    vid = ca_data_utils.load_v_matrix()
    labels = ca_data_utils.load_labels()
    X_train, X_test, y_train, y_test = train_test_split(vid.T, labels, test_size=0.25) # random state?
    classifiers = [
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier()
    ]
    names = ['linear SVM', 'RBF SVM', 'Random Forest', 'AdaBoost']
    train = []
    test = []
    for name, clf in zip(names, classifiers):
        print('starting to train with ', name)
        clf.fit(X_train, y_train)
        print('---> calculating training set accuracy:')
        train_accu = clf.score(X_train, y_train)
        train.append(train_accu)
        print('---> training set accuracy: ', train_accu)
        print('---> calculating testing set accuracy:')
        test_accu = clf.score(X_test, y_test)
        test.append(test_accu)
        print('---> testing set accuracy: ', test_accu)
    np.save('../data/clf_result/clf_names', names)
    np.save('../data/clf_results/train_accuracy', train)
    np.save('../data/clf_results/test_accuracy', test)


if __name__ == '__main__':
    main()