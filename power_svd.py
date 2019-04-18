import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from random import normalvariate
from numpy.linalg import norm
from math import sqrt


def power_iteration(A, epsilon=1e-10):
    B = np.dot(A.T, A)

    print('generating random vector...')
    unnormalized = np.random.rand(B.shape[0])
    v = unnormalized / np.linalg.norm(unnormalized)
    print('done generating random vector!')

    iterations = 0
    while True:
        iterations += 1
        print(iterations)
        Bv = np.dot(B, v)
        v_new = Bv / np.linalg.norm(Bv)

        if abs(np.dot(v_new, v)) > 1 - epsilon:
            break
        v = v_new
    return v_new


def svd(A, k, epsilon=1e-10):
    # assume n >= m
    n, m = A.shape
    svdSoFar = []

    for i in range(k):
        print('starting number ' + str(i))

        v = power_iteration(A, epsilon=epsilon)
        Av = np.dot(A, v)
        sigma = np.linalg.norm(Av)
        u = Av / sigma

        svdSoFar.append((sigma, u, v))
        A -= sigma * np.outer(u, v)

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return us.T, singularValues, vs