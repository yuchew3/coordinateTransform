import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from pydmd import DMD

def main():
    # when on cylon to test on full vid fdir = '../data/vid.tif'
    vid = io.imread('../data/vid.tif') 
    flatten = np.transpose([np.asarray(x).flatten() for x in vid])
    X1 = flatten[:,:-1]
    X2 = flatten[:,1:]

    print('start dmd')
    dmd = DMD(svd_rank=10)
    dmd.fit(flatten.T)
    print('finished fitting data')

    # for eig in dmd.eigs:
    #     print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
    # dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    diff = dmd.reconstructed_data.T - flatten
    diff = diff.reshape(180, 240, 40000)
    np.save('../data/dmd/reconstructed', dmd.reconstructed_data.T.reshape((180,240,40000)))
    np.save('../data/dmd/difference', diff)


# use the original video for test
if __name__ == '__main__':
    main()