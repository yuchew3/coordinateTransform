import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
import power_svd
from skimage import io
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

def reconstruction_test():
    X = np.random.randint(5, size=(70,20))
    u,s,v = power_svd.svd(X, 20)
    epsilon = 1e-10
    diff = X - np.dot(u, np.dot(np.diag(s), v))

    print(np.where(diff > epsilon))

def svd_tolerance():
    # matrix = np.load('../data/converted_matrix.npy')
    matrix = io.imread('../data/vid.tif')
    matrix = np.transpose([np.asarray(x).flatten() for x in matrix])
    print('done loading data')
    U, s, V = randomized_svd(matrix, 
                              n_components=500)
    n_oversamples = [50, 100, 150, 200] # from 5 - 20
    n_iters = [40, 70, 100, 200, 300] # from 4, 7, 

    norms = []
    for n in n_oversamples:
        print('start doing n_oversample = ', n)
        norm = check_n_oversamples(matrix, n)
        norms.append(norm)
        print('done doing n_oversample = ', n)
    # plt.plot(n_oversamples, norms)
    # plt.savefig('n_oversamples')
    np.save('n_oversamples', norms)

    norms = []
    for n in n_iters:
        print('starting doing n_iters = ', n)
        norm = check_n_iters(matrix, n)
        norms.append(norm)
        print('done doing n_iters = ', n)
    # plt.plot(n_iters, norms)
    # plt.savefig('n_iters')
    np.save('n_iters', norms)

def svd_tune_rank():
    matrix = io.imread('../data/vid.tif')
    matrix = np.transpose([np.asarray(x).flatten() for x in matrix])
    print('done loading data')
    ranks = np.linspace(500,10000, 20)
    norms = []
    for r in ranks:
        print('starting rank = ', r)
        U, s, V = randomized_svd(matrix, n_oversamples=150, n_iter=, n_components=r)
        diff = matrix - np.dot(U, np.dot(np.diag(s), V))
        norms.append(np.linalg.norm(diff))
        print('done rank = ', r)
    norms = norms / 15452.521
    np.save('ranks', norms)
    



                            
def check_n_oversamples(matrix, n):
    U, s, V = randomized_svd(matrix, n_oversamples=n, n_components=500)
    diff = matrix - np.dot(U, np.dot(np.diag(s), V))
    return np.linalg.norm(diff)

def check_n_iters(matrix, n):
    U, s, V = randomized_svd(matrix, n_iter=n, n_components=500)
    diff = matrix - np.dot(U, np.dot(np.diag(s), V))
    return np.linalg.norm(diff)

if __name__ == '__main__':
    svd_tolerance()
    svd_tune_rank()