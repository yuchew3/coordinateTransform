import scipy.io
import numpy as np
import skimage.io

def trunk_data_by_states(vid, labels):
    state_1 = np.where(labels==1)
    state_2 = np.where(labels==2)
    state_3 = np.where(labels==3)
    print(state_1[0].shape)
    print(state_2[0].shape)
    print(state_3[0].shape)
    sleep = vid[state_1[0]]
    wake_1 = vid[state_2[0]]
    wake_2 = vid[state_3[0]]
    np.save('../data/sleep',sleep)
    np.save('../data/wake_1',wake_1)
    np.save('../data/wake_2',wake_2)

def load_labels():
    mat = scipy.io.loadmat('../data/frameStates.mat')
    mat = np.array(mat['b'])
    return mat[0][:-1]

def load_vid():
    vid = skimage.io.imread('../data/vid.tif')
    vid = np.transpose([np.asarray(x).flatten() for x in vid])
    return vid

if __name__ == '__main__':
    labels = load_labels()
    vid = load_vid()
    print(vid.shape)
    trunk_data_by_states(vid, labels)

