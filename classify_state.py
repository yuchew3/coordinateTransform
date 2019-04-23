import numpy as np
import matplotlib.pyplot as plt
import ca_data_utils

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def main():
    vid = ca_data_utils.load_v_matrix()
    labels = ca_data_utils.load_labels()
    X_train, X_test, y_train, y_test = train_test_split(vid.T, labels, test_size=0.25) # random state?
    clf = SVC(kernel="linear", C=0.025)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)


if __name__ == '__main__':
    main()