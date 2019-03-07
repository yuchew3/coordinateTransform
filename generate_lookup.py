import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2


def generate_video_with_coords():
    vid = io.imread('vid.tif')
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
        for j in len(x):
            flatten[j] = converted_frame[x[j]][y[j]]
        con_vid[:,i] = flatten
    np.save('../data/converted_matrix', con_vid)

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
    generate_video_with_coords()