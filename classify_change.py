import numpy as np
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
import ca_data_utils

def load_data():
    X = ca_data_utils.load_v_matrix().T[9:39992]
    labels = ca_data_utils.load_labels()[9:39992]
    labels[labels==3] = 2
    labels = [labels[k+1] - labels[k] for k in range(len(labels)-1)]
    labels = np.array(labels)
    X = [X[k+1] - X[k] for k in range(len(X) - 1)]
    labels[labels!=0] = 1
    return X, labels

if __name__ == '__main__':
    X, labels = load_data()
    x = np.linspace(0, 39750, 250, dtype=np.int64)
    print(x.shape)
    b = np.where(labels==1)[0]
    print(b.shape)
    x = np.concatenate((x, b))
    x = np.sort(x)
    total_len = len(x)
    cut = int(3 * total_len / 4)
    print(len(x))
    X = np.array(X)[x]
    labels = labels[x]

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
        clf.fit(X[:cut, labels[:cut]])
        score = clf.score(X[cut:], labels[cut:])
        print('---> test accuracy is ', str(score))
