import numpy as np
import matplotlib.pyplot as plt
import ca_data_utils

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

def select_models():
    vid = ca_data_utils.load_v_matrix()[8:39992]
    labels = ca_data_utils.load_labels()[8:39992]
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
    np.save('../data/clf_results/clf_names', names)
    np.save('../data/clf_results/train_accuracy', train)
    np.save('../data/clf_results/test_accuracy', test)

def tune_rbf_svm():
    vid = ca_data_utils.load_v_matrix()[:,8:39992]
    labels = ca_data_utils.load_labels()[8:39992]

    scaler = StandardScaler()
    X = scaler.fit_transform(vid.T)

    C_range = np.logspace(-2, 10, 13)
    print('C range: ', C_range)
    gamma_range = np.logspace(-9, 3, 13)
    print('gamma range: ', gamma_range)
    param_grid = dict(gamma=gamma_range, C=C_range)

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    print('start to train...')
    grid.fit(X, labels)

    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    
    print('The results are:')
    print(grid.cv_results_)



if __name__ == '__main__':
    tune_rbf_svm()