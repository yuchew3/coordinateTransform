import numpy as np
from sklearn.svm import SVC
import ca_data_utils

def load_data():
    X = ca_data_utils.load_v_matrix().T[9:39992]
    labels = ca_data_utils.load_labels()[9:39992]
    labels[labels==3] = 2
    print(np.where(labels == 1))
    print(len(np.where(labels == 2)))
    labels = [labels[k+1] - labels[k] for k in range(len(labels)-1)]
    X = [X[k+1] - X[k] for k in range(len(X) - 1)]
    labels[labels != 0] = 1
    return X, labels

if __name__ == '__main__':
    X, labels = load_data()
    clf = SVC(C=0.1, gamma=0.01)
    clf.fit(X[:30000], labels[:30000])
    score = clf.score(X[30000:], labels[30000:])
    a = np.where(labels==0)
    print(len(a[0]))
    print(len(labels))
    print(score)
