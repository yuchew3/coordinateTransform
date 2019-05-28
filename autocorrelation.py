import pandas as pd
import numpy as np
import ca_data_utils

def load_data(step):
    v = ca_data_utils.load_v_matrix().T[9:39992:step]
    labels = ca_data_utils.load_labels()[9:39992:step]
    return v, labels
    


if __name__ == "__main__":
    X, labels = load_data(2)
    # X = X[:,0]
    lags = 10
    # dataset = np.fromfunction(lambda i, j: pd.Series(X[i]).autocorr(lag=j+1), (len(X), lags))
    dataset = np.zeros(X.shape[1], lags))
    for i in range(X.shape[1]):
        series = pd.Series(X[:,i])
        for j in range(lags):
            corr = series.autocorr(lag=j+1)
            dataset[i, j] = corr
    np.save('../data/autocorr', dataset)
    print(dataset.shape)