import numpy as np
from sklearn.svm import SVC
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
    x = np.linspace(0, 39750, 250, dtype=np.int16)
    print(np.dtype(x))
    print(np.where(labels==1)[0])
    x = np.concatenate(x, np.where(labels==1)[0])
    x = np.sort(np.array(x))
    total_len = len(x)
    cut = int(3 * total_len / 4)
    print(len(x))
    X = X[x]
    labels = labels[x]
    clf = SVC(C=0.1, gamma=0.01)
    clf.fit(X[:cut], labels[:cut])
    score = clf.score(X[cut:], labels[cut:])
    print(score)
