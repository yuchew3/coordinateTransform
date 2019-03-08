import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from skimage import io
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import cv2

def run_svd():
    matrix = np.load('../data/converted_matrix.npy')
    print('done loading data')
    U, s, V = randomized_svd(matrix, 
                              n_components=500)
    np.save('../data/spatial', U)
    np.save('../data/temporal',V)
    np.save('../data/singular_v', s)
    


def generate_video_with_coords():
    vid = io.imread('../data/vid.tif')
    print(vid.shape)
    matrix = np.load('../data/matrix.npy')
    compare = np.load("../data/flat_cortex_template.npy")
    x = np.load('../data/map/x_coords.npy')
    y = np.load('../data/map/y_coords.npy')
    con_vid = np.zeros((len(x), vid.shape[0]))
    for i in range(vid.shape[0]):
        if i % 100 == 0:
            print('converting frame ' + str(i))
        frame = vid[i]
        converted_frame = cv2.warpPerspective(frame, matrix, compare.T.shape, borderValue=np.nan)
        flatten = np.zeros(len(x))
        for j in range(len(x)):
            flatten[j] = converted_frame[x[j]][y[j]]
        con_vid[:,i] = flatten
    np.save('../data/converted_matrix', con_vid)

    rank = 500
    U, s, V = np.linalg.svd(con_vid)
    rank = min(r, U.shape[1])
    V = V.conj().T
    U_r = U[:, :rank]
    V_r = V[:, :rank]
    s = S[:rank]
    np.save('../data/spatial', U)
    np.save('../data/temporal',V)
    np.save('../data/singular_v', s)



def generate_coords():
    vid = io.imread('short_vid.tif')
    first = vid[0]
    matrix = np.load('matrix.npy')
    compare = np.load("../data/flat_cortex_template.npy")
    converted_first = cv2.warpPerspective(first,matrix,compare.T.shape,borderValue=np.nan)
    intersection = converted_first
    x, y = np.where(np.isnan(compare))
    for i in range(len(x)):
        intersection[x[i]][y[i]] = np.nan

    x, y = np.where(~np.isnan(intersection))
    np.save('../data/map/x_coords', x)
    np.save('../data/map/y_coords', y)
    
def isnan(frame):
    compare = np.load("../data/flat_cortex_template.npy")
    return not(np.isnan(compare))

if __name__ == '__main__':
    run_svd()