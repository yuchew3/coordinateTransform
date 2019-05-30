import numpy as np
import matplotlib.pyplot as plt
import ca_data_utils
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

def select_models():
    vid = ca_data_utils.load_v_matrix().T[8:39992]
    labels = ca_data_utils.load_labels()[8:39992]
    X_train, X_test, y_train, y_test = train_test_split(vid, labels, test_size=0.25) # random state?
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
    vid = ca_data_utils.load_v_matrix()[:,9:39992]
    labels = ca_data_utils.load_labels()[9:39992]

    scaler = StandardScaler()
    X = scaler.fit_transform(vid.T)

    C_range = np.logspace(-1, 2, 4)
    print('C range: ', C_range)
    gamma_range = np.logspace(-3, 3, 7) * 1. / X.shape[1] #(-3,3,7)
    print('gamma range: ', gamma_range)
    param_grid = dict(gamma=gamma_range, C=C_range)

    cv = StratifiedShuffleSplit(test_size=0.25, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=20)
    print('start to train...')
    grid.fit(X, labels)

    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    
    print('The results are:')
    print(grid.cv_results_)

def kernel_pca():
    X = ca_data_utils.load_v_matrix().T[8:39992]
    labels = ca_data_utils.load_labels()[8:39992]
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, n_jobs=20)
    print('starting kpca')
    X_kpca = kpca.fit_transform(X)
    print('finished')
    # pca = PCA()
    # X_pca = pca.fit_transform(X)

    sleep = labels == 1
    wake1 = labels == 2
    wake2 = labels == 3

    plt.figure()
    plt.subplot(1, 2, 1, aspect='equal')
    plt.title("Original space")
    plt.scatter(X[sleep, 0], X[sleep, 1], c="red",
            s=20, edgecolor='k')
    plt.scatter(X[wake1, 0], X[wake1, 1], c="blue",
            s=20, edgecolor='k')
    plt.scatter(X[wake2, 0], X[wake2, 1], c='green',
            s=20, edgecolors='k')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.subplot(1, 2, 2, aspect='equal')
    plt.scatter(X_kpca[sleep, 0], X_kpca[sleep, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(X_kpca[wake1, 0], X_kpca[wake1, 1], c="blue",
                s=20, edgecolor='k')
    plt.scatter(X_kpca[wake2, 0], X_kpca[wake2, 1], c='green',
                s=20, edgecolor='k')
    plt.title("Projection by KPCA")
    plt.xlabel(r"1st principal component in space induced by $\phi$")
    plt.ylabel("2nd component")
    plt.tight_layout()
    plt.savefig('../data/clf_results/kpca')

def xgboost(gamma):
    X = ca_data_utils.load_v_matrix().T[8:39992:5]
    labels = ca_data_utils.load_labels()[8:39992:5]
    clf = xgb.XGBClassifier(gamma=gamma, learning_rate=0.1)
    total_len = len(labels)
    cut = int(3 * total_len / 4)
    clf.fit(X[:cut], labels[:cut])
    score = clf.score(X[cut:], labels[cut:])
    print(score)

if __name__ == '__main__':
    # kernel_pca()
    for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        print('for gamma = ', i)
        xgboost(i)