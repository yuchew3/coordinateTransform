import numpy as np
import cv2
import matplotlib.pyplot as plt

# This file is for all testing, just to see what it is/does, functions

def learn_flat_map():
    img = np.load("../data/flat_cortex_template.npy")
    brain_slice = np.load("slice.npy")
    heatmap = cv2.cvtColor(brain_slice, cv2.COLOR_GRAY2BGR)
    print(heatmap.shape)
    heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_WINTER)
    heatmap = heatmap/255
    cv2.imshow("image", heatmap)
    cv2.waitKey(0)

if __name__ == "__main__":
    learn_flat_map()
