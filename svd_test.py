import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
import power_svd
import ca_data_utils
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
    matrix = ca_data_utils.load_vid()
    print('done loading data')
    U, s, V = randomized_svd(matrix, 
                              n_components=500)
    n_oversamples = [50, 100, 150, 200] # from 5 - 20
    n_iters = [40, 70, 100] # from 4, 7, 

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
    matrix = ca_data_utils.load_vid()
    print('done loading data')
    ranks = np.linspace(500,1000,5)
    ranks = ranks.astype(int)
    norms = []
    for r in ranks:
        print('starting rank = ', r)
        U, s, V = randomized_svd(matrix, n_oversamples=150, n_iter=30, n_components=r)
        diff = matrix - np.dot(U, np.dot(np.diag(s), V))
        norms.append(np.linalg.norm(diff))
        print('done rank = ', r)
    norms = np.array(norms) / 15452.521
    np.save('ranks', norms)
    

def make_svd_matrices():
    matrix = ca_data_utils.load_vid()
    print('done loading data')
    U, s, V = randomized_svd(matrix, n_oversamples=10, n_iter=20, n_components=1000)
    np.save('umat', U)
    np.save('svmat', np.dot(np.diag(s), V))

                            
def check_n_oversamples(matrix, n):
    U, s, V = randomized_svd(matrix, n_oversamples=n, n_components=500)
    diff = matrix - np.dot(U, np.dot(np.diag(s), V))
    return np.linalg.norm(diff)

def check_n_iters(matrix, n):
    U, s, V = randomized_svd(matrix, n_iter=n, n_components=500)
    diff = matrix - np.dot(U, np.dot(np.diag(s), V))
    return np.linalg.norm(diff)

if __name__ == '__main__':
    #svd_tolerance()
    # svd_tune_rank()
    make_svd_matrices()