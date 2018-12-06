import numpy as np
import matplotlib.pyplot as plt

# This file is for all testing, just to see what it is/does, functions

def learn_flat_map():
    img = np.load("../data/flat_cortex_template.npy")
    plt.imshow(img)
    plt.show()
    print(img.shape)


if __name__ == "__main__":
    learn_flat_map()
