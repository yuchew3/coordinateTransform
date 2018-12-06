import numpy as np
import cv2
import matplotlib.pyplot as plt

# This file is for all testing, just to see what it is/does, functions

def learn_flat_map():
    img = np.load("../data/flat_cortex_template.npy")
    cv2.imshow(img)
    print(img.shape)


if __name__ == "__main__":
    learn_flat_map()