import matplotlib.pyplot as plt
import numpy as np
import ca_data_utils

from pydmd import DMD

def main():
    # when on cylon to test on full vid fdir = '../data/vid.tif'
    # when local test use 'short_vid.tif'
    vid = ca_data_utils.load_vid()
    X1 = flatten[:,:-1]
    X2 = flatten[:,1:]

    print('start dmd')
    dmd = DMD(svd_rank=10)
    dmd.fit(flatten.T)
    print('finished fitting data')

    for eig in dmd.eigs:
        print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
    dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    self_re = np.load('reconstructed.npy')
    diff = dmd.reconstructed_data.T - self_re

    print(len(np.where(np.abs(diff)>1e-3)[0]))
    print(diff.max())
    print(diff.min())


# use the original video for test
if __name__ == '__main__':
    main()