import numpy as np
import scipy.io
import skimage.io

def load_vid():
    # load whole original video
    # size of video: (40000, 180, 240)
    # size of returned, flattened matrix: (43200, 40000)
    vid = skimage.io.imread('../data/vid.tif')
    vid = np.transpose([np.asarray(x).flatten() for x in vid])
    return vid

def load_v_matrix():
    # load the V matrix produced by doing svd on the original video
    # size of V matrix: (1000, 40000)
    v = np.load('../data/final_svd/vmat.npy')
    return v

def load_labels():
    # load the labels to each frame of the video
    # I don't know why but there are 17 0s [0,1,2,3,4,5,6,7,39992,39993,39994,39995,39996,39997,39998,39999]
    # 1: sleep
    # 2: wake with movement
    # 3: wake without movement
    mat = scipy.io.loadmat('../data/frameStates.mat')
    mat = np.array(mat['b'])
    return mat[0][:-1]

def load_sleep_vid():
    # load flattened video for all sleep frames
    # size: (43200, 16603)
    vid = np.load('../data/byState/sleep.npy')
    return vid

def load_wake_1_vid():
    # load flattened video for all wake and move frames
    # size: (43200, 13754)
    vid = np.load('../data/byState/wake_1.npy')
    return vid

def load_wake_2_vid():
    # load flattened video for all wake and not move frames
    # size: (43200, 9626)
    vid = np.load('../data/byState/wake_2.npy')
    return vid

def load_wake_vid():
    # load flattened video for all wake frames (move and not move)
    # size: (43200, 23380)
    vid = np.load('../data/byState/wake.npy')
    return vid