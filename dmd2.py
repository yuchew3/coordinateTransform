import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from pydmd import DMD

def main():
    vid = io.imread('short_vid.tif') 
    flatten = np.transpose([np.asarray(x).flatten() for x in vid])
    X1 = flatten[:,:-1]
    X2 = flatten[:,1:]

    dmd = DMD(svd_rank=10)
    dmd.fit(flatten.T)

    # for eig in dmd.eigs:
    #     print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
    # dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    other_recon = np.load('reconstructed.npy')
    print(other_recon.shape)
    diff = dmd.reconstructed_data.T - flatten
    print(vid.shape)
    diff = diff.reshape(180, 240, 400)
    print(np.amax(diff))
    print(np.amax(other_recon - flatten))



# use the original video for test
if __name__ == '__main__':
    main()