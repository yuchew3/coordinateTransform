import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
import power_svd
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

def reconstruction_test():
    X = np.random.randint(5, size=(70,20))
    u,s,v = power_svd.svd(X, 20)
    epsilon = 1e-10
    diff = X - np.dot(u, np.dot(np.diag(s), v))

    print(np.where(diff > epsilon))

def svd_tolerance():
    matrix = np.load('../data/converted_matrix.npy')
    print('done loading data')
    U, s, V = randomized_svd(matrix, 
                              n_components=500)
    n_oversamples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    norms = []
    for n in n_oversamples:
        norm = check_n_oversamples(matrix, n)
        norms.append(norm)
    plt.plot(n_oversamples, norms)
    plt.show()
    plt.savefig('n_oversamples')

                            
def check_n_oversamples(matrix, n):
    U, s, V = randomized_svd(matrix, n_oversamples=n)
    diff = matrix - np.dot(U, np.dot(np.diag(s), V))
    return np.linalg.norm(diff)

if __name__ == '__main__':
    svd_tolerance()