import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.linspace(0,1,5)
    y = np.linspace(0,5,5)
    plt.scatter(x, y)
    plt.savefig('test.png')