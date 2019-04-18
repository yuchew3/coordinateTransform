import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import ca_data_utils
from past.utils import old_div
import plot_utils
import power_svd
import time


def dmd(X1, X2, r, dt):
    # U, S, V = np.linalg.svd(X1, full_matrices=False)
    print('start power iteration')
    startt = time.time()
    U, s, V = power_svd.svd(X1, r)
    V = V.conj().T
    print('get a result in ' + str(time.time() - startt) + 's')
    # rank = min(r, U.shape[1])

    # U_r = U[:, :rank]
    # V_r = V[:, :rank]
    # s = S[:rank]

    Atilde = U.T.conj().dot(X2).dot(V) * np.reciprocal(s)

    eigenvalues_low, eigenvectors_low = np.linalg.eig(Atilde)

    eigenvalues_high = eigenvalues_low
    eigenvectors_high = (X2.dot(V) * np.reciprocal(s)).dot(eigenvectors_low)

    b = np.linalg.lstsq(eigenvectors_high, X1.T[0], rcond=False)[0]

    omega = old_div(np.log(eigenvalues_high), dt)
    ######
    timesteps = np.arange(0, 39999 + dt, dt)
    temp = np.exp(np.multiply(*np.meshgrid(omega, timesteps)))
    time_dynamics = (temp * b).T
    Xdmd = eigenvectors_high.dot(time_dynamics)


def main():
    flatten = ca_data_utils.load_vid()
    X1 = flatten[:,:-1]
    X2 = flatten[:,1:]

    eigenvectors_high, omega, eigenvalues_high, b, Xdmd = dmd(X1, X2, 10, 1)

    # plot_utils.plot_eigenvalues(eigenvalues_high)
    # plot_utils.plot_modes(eigenvectors_high, vid[0].shape, eigenvalues_high)

    print(Xdmd.shape)
    np.save('reconstructed', Xdmd)
    np.save('eigenvalue_high', eigenvalues_high)
    np.save('eigenvectors_high', eigenvectors_high)


# use the original video for test
if __name__ == '__main__':
    main()