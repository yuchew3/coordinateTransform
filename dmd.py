import numpy as np
from skimage import io
from past.utils import old_div
import plot_utils


def dmd(X1, X2, r, dt):
    U, S, V = np.linalg.svd(X1, full_matrices=False)
    V = V.conj().T
    rank = min(r, U.shape[1])

    U_r = U[:, :rank]
    V_r = V[:, :rank]
    s = S[:rank]

    Atilde = U_r.T.conj().dot(X2).dot(V_r) * np.reciprocal(s)

    eigenvalues_low, eigenvectors_low = np.linalg.eig(Atilde)

    eigenvalues_high = eigenvalues_low
    eigenvectors_high = X2.dot(V_r) * np.reciprocal(s).dot(eigenvectors_low)

    b = np.linalg.lstsq(eigenvectors_high, X1.T[0], rcond=False)[0]

    omega = old_div(np.log(eigenvalues_high), dt)
    ######
    timesteps = np.arange(0, 399 + dt, dt)
    temp = np.exp(np.multiply(*np.meshgrid(omega, timesteps)))
    time_dynamics = (temp * b).T
    Xdmd = eigenvectors_high.dot(time_dynamics)

    return eigenvectors_high, omega, eigenvalues_high, b, Xdmd


def main():
    vid = io.imread('short_vid.tif')
    flatten = np.transpose([np.asarray(x).flatten() for x in vid])
    X1 = flatten[:,:-1]
    X2 = flatten[:,1:]

    eigenvectors_high, omega, eigenvalues_high, b, Xdmd = dmd(X1, X2, 10, 2)

    plot_utils.plot_eigenvalues(eigenvalues_high)
    plot_utils.plot_modes(eigenvectors_high, vid[0].shape)

    print(Xdmd.shape)
    np.save('reconstructed', Xdmd)


# use the original video for test
if __name__ == '__main__':
    main()