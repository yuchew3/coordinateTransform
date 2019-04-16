import scipy.io
import numpy as np

def load_labels():
    mat = io.loadmat('../data/frameStates.mat')
    mat = np.array(mat['b'])
    return mat[0][:-1]
if __name__ == '__main__':
    labels = load_labels()
    print(labels.shape)